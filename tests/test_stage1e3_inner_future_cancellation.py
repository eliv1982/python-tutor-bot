"""
Stage 1E.3 regression tests: `utils/helpers.py:await_worker()`'s handling of
an INNER Future that is itself cancelled, as opposed to the OUTER caller
being cancelled.

A third independent Codex audit found that Stage 1E.2's `await_worker()`
loop:

    while True:
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            if first_cancellation is None:
                first_cancellation = exc
            continue
        ...

busy-loops forever if `future` itself is (or becomes) cancelled: once
`future.cancelled()` is True, `asyncio.shield(future)` returns `future`
directly (it's already done) and awaiting it raises `CancelledError` again
immediately — with nothing about the loop's state ever changing, so
`continue` just re-enters the same branch forever, spinning the CPU without
ever suspending back to the event loop.

The fix distinguishes, inside the `except CancelledError` branch, whether
`future.cancelled()` is True (the Future itself is terminal — no worker left
to wait for, so the loop must stop immediately) from False (the CancelledError
came from the outer `asyncio.shield()` wrapper, worker still alive — the
original Stage 1E.2 behavior of recording-and-looping is correct and
unchanged).

These tests exercise `await_worker()` directly with hand-controlled
`asyncio.Future` objects — no real executor thread involved, since the
whole point is to deterministically construct a Future that is already (or
becomes) cancelled, which is not something a real executor job can be made
to do reliably by timing alone. Per Stage 1E.3's brief, `await_worker()`
intentionally accepts any object satisfying `asyncio.Future`'s async
protocol (`.cancelled()` plus awaitability) — a `loop.create_future()` is a
legitimate, fully-representative stand-in for the `loop.run_in_executor()`
Future it's normally called with.
"""

import asyncio

import pytest

from utils.helpers import await_worker


# ---------------------------------------------------------------------------
# A. Inner Future cancelled BEFORE any outer cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_await_worker_terminates_immediately_when_future_already_cancelled(monkeypatch):
    """A Future that is already `cancelled()` before `await_worker()` is
    ever called on it (e.g. legitimately cancelled before its executor
    callable began running) must make `await_worker()` raise
    `CancelledError` immediately, in a single loop iteration — never spin.

    The deterministic proof is the `asyncio.shield` call count, not timing:
    a busy-loop regression would drive that count arbitrarily high (and,
    since the loop never actually suspends back to the event loop in the
    broken version, it would hang outright rather than merely being slow —
    the bounded `wait_for` below is only a best-effort safety net for that
    case, not the correctness mechanism)."""
    import utils.helpers as helpers_module

    shield_calls = {"n": 0}
    real_shield = asyncio.shield

    def counting_shield(aw):
        shield_calls["n"] += 1
        return real_shield(aw)

    monkeypatch.setattr(helpers_module.asyncio, "shield", counting_shield)

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    future.cancel()
    assert future.cancelled()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(await_worker(future), timeout=0.5)

    assert shield_calls["n"] == 1, (
        f"await_worker() looped instead of terminating immediately: "
        f"asyncio.shield() was called {shield_calls['n']} times"
    )


@pytest.mark.asyncio
async def test_await_worker_does_not_attempt_reconciliation_for_already_cancelled_future():
    """No worker outcome exists to reconcile when the Future was cancelled
    before it ever ran — `await_worker()` must not call `.result()` or
    `.exception()` on it (both raise `CancelledError` themselves on a
    cancelled Future, which would just be a confusing secondary failure
    mode instead of the clean CancelledError propagation this proves)."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    future.cancel()

    with pytest.raises(asyncio.CancelledError):
        await await_worker(future)

    # Sanity: the Future's own terminal state is untouched by await_worker()
    # — still exactly "cancelled", nothing further attempted on it.
    assert future.cancelled()


# ---------------------------------------------------------------------------
# B. Inner Future cancelled AFTER an outer cancellation was already recorded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_await_worker_keeps_original_outer_cancellation_when_future_later_cancelled(monkeypatch):
    """Ordering: the OUTER caller is cancelled first (while `future` is
    still genuinely pending), `await_worker()` correctly records that and
    keeps waiting — then `future` itself transitions to cancelled (e.g. a
    caller passed in a Future that was itself cancelled out from under
    reconciliation). This must terminate the loop (not spin) AND must
    surface the ORIGINAL first outer cancellation as the caller-visible
    result — never a fresh one manufactured from the Future's own
    cancellation.

    Distinguishing "the original outer CancelledError" from "a new one" by
    identity/message (via `Task.cancel(msg=...)`) is the deterministic
    proof — not relative timing."""
    import utils.helpers as helpers_module

    shield_calls = {"n": 0}
    real_shield = asyncio.shield

    def counting_shield(aw):
        shield_calls["n"] += 1
        return real_shield(aw)

    monkeypatch.setattr(helpers_module.asyncio, "shield", counting_shield)

    loop = asyncio.get_running_loop()
    future = loop.create_future()  # deliberately left pending

    async def caller():
        await await_worker(future)

    outer = asyncio.create_task(caller())
    await asyncio.sleep(0)  # let `outer` actually reach `await asyncio.shield(future)`
    assert shield_calls["n"] == 1

    outer.cancel(msg="ORIGINAL-OUTER-CANCEL")
    await asyncio.sleep(0)  # deliver + record the first cancellation, loop back
    assert not outer.done(), "outer finished after the first cancel — reconciliation loop broken"
    assert shield_calls["n"] == 2, "await_worker() did not re-shield after absorbing the first cancel"
    assert not future.cancelled()  # the worker Future itself is still untouched

    # Now the Future itself becomes cancelled — distinct from, and arriving
    # after, the outer cancellation above.
    future.cancel(msg="UNRELATED-INNER-FUTURE-CANCEL")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(outer, timeout=1)

    assert exc_info.value.args and exc_info.value.args[0] == "ORIGINAL-OUTER-CANCEL", (
        f"caller-visible cancellation was not the original outer one: "
        f"{exc_info.value.args!r}"
    )
    # The Future's cancellation is delivered through the ALREADY-PENDING
    # shield call (#2, still awaiting `future`) — shield's own internal
    # done-callback cancels that same outer wrapper when its inner Future
    # is cancelled, so no THIRD `asyncio.shield()` call is ever made. The
    # count staying at 2 (not climbing further) is the proof the loop
    # terminated here instead of re-shielding again.
    assert shield_calls["n"] == 2


# ---------------------------------------------------------------------------
# C. Repeated outer cancellation while the Future stays genuinely alive
#    (Stage 1E.2 behavior, must be unaffected by the Stage 1E.3 fix)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_await_worker_still_survives_repeated_outer_cancellation_of_live_future():
    """Companion to the Stage 1E.3 fix: proves the ordinary repeated-outer-
    cancellation path (the Future never itself becomes cancelled, only the
    outer caller is cancelled twice) is completely unaffected — the loop
    keeps waiting for the Future's genuine result both times."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    async def caller():
        return await await_worker(future)

    outer = asyncio.create_task(caller())
    await asyncio.sleep(0)

    outer.cancel()
    await asyncio.sleep(0)
    assert not outer.done()

    outer.cancel()
    await asyncio.sleep(0)
    assert not outer.done()
    assert not future.cancelled()

    future.set_result("worker-result")

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(outer, timeout=1)

    # The Future itself completed normally and is available for the
    # caller's own reconciliation — never cancelled, never abandoned.
    assert future.done()
    assert not future.cancelled()
    assert future.result() == "worker-result"
