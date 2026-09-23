"""
Stage 7A-3 runtime-topology prerequisite — service_main.py's unified
Telegram+web composition root.

Every test here uses fakes/mocks at the exact seams service_main.py
exposes for this purpose (run_telegram/run_uvicorn/stop_telegram/
stop_uvicorn/close_resources, or the individual _default_*/_build_*
helpers plus the rag.index/db.engine module-attribute boundaries) —
never a real bound socket, real Telegram polling, a real provider call,
or a real Qdrant/PostgreSQL instance. See service_main.py's own module
docstring for why a hand-rolled asyncio.wait(FIRST_COMPLETED) supervisor
is used instead of raw asyncio.TaskGroup despite Python 3.12 supporting
it (TaskGroup's automatic hard-cancel-on-exception would skip
uvicorn.Server's own cooperative should_exit shutdown path), and for the
exact cancellation-resilient algorithm `_supervise()`/`_run_uncancellable()`
implement (corrective pass — independent-audit MAJOR 1/2 + MINOR
findings).

Section mapping:
  - test_normal_lifecycle_* — normal lifecycle.
  - test_telegram_startup_failure_* / test_uvicorn_startup_failure_* /
    test_telegram_runtime_failure_* — adapter failure.
  - test_simultaneous_adapter_failures_* — deterministic primary +
    secondary-exception retrieval under simultaneous failure.
  - test_telegram_unexpected_normal_exit_* / test_uvicorn_unexpected_
    normal_exit_* — unexpected normal return (UnifiedServiceAdapterExited).
  - test_external_cancellation_* — cancellation during the initial wait.
  - test_cancellation_during_already_in_progress_sibling_teardown_* —
    cancellation while settlement is already active (MAJOR 1 regression).
  - test_graceful_stop_force_cancels_* — forced-cancel fallback.
  - test_uvicorn_forced_cancellation_during_hanging_lifespan_startup_* —
    no leaked uvicorn LifespanOn.main child task (MAJOR 2 regression).
  - test_cleanup_failure_does_not_mask_* — cleanup-masking precedence.
  - test_uvicorn_system_exit_* — SystemExit propagation.
  - test_shutdown_requested_while_telegram_still_in_its_startup_path_* —
    Telegram startup-phase shutdown.
  - test_no_unobserved_task_exceptions_* — no "Task exception was never
    retrieved".
  - test_create_app_owns_db_lifecycle_* — real FastAPI lifespan behavior
    for both `owns_db_lifecycle` values.

Second corrective pass additions:
  - test_settle_telegram_pending_tasks_* / test_default_run_telegram_
    settles_* / test_production_wiring_settles_telegram_pending_task_* —
    MAJOR: Telegram handler child tasks (`AsyncTeleBot._pending_tasks`)
    settled before "Telegram adapter settled".
  - test_run_uncancellable_* — MAJOR: `_run_uncancellable()`'s own
    protected operation is now BaseException-safe (SystemExit/
    KeyboardInterrupt/self-raised CancelledError), kept structurally
    distinct from external caller cancellation.
  - test_cleanup_systemexit_*/test_cleanup_keyboardinterrupt_* — the same
    cleanup-masking precedence as test_cleanup_failure_does_not_mask_*,
    now proven for scheduler-special BaseExceptions too.
  - test_*_self_raised_cancelled_error_* — MINOR: an adapter cancelled
    BEFORE the composition root ever asked it to stop is a non-successful
    UnifiedServiceAdapterCancelled, not a silently-skipped exit.
  - test_uvicorn_is_pinned_*/test_pytelegrambotapi_is_pinned_*/
    test_*_installed_version_matches_the_pin — MINOR: exact dependency
    pins for the two private-implementation integration points above.

Third corrective pass additions (independent re-audit — MAJOR: "adapter
settlement is not transitive to reachable worker/request descendants"):
  - test_settle_uvicorn_request_tasks_* — unit tests for the new
    `_settle_uvicorn_request_tasks()` helper against a fake server/
    server_state stand-in, mirroring test_settle_telegram_pending_tasks_*.
  - test_uvicorn_request_task_cannot_outlive_* — a REAL uvicorn.Server
    bound to 127.0.0.1:0 (loopback, OS-assigned ephemeral port — never a
    fixed/external one), proving a pure-async in-flight ASGI request task
    cannot survive `_build_uvicorn_adapter()`'s own `run()` closure ending
    (Section 12).
  - test_telegram_handler_resource_worker_cannot_outlive_* /
    test_web_request_resource_worker_cannot_outlive_* — Sections 10/11:
    a REAL OS-thread worker (submit_worker()/await_worker(), controlled by
    `threading.Event`, never merely a fake asyncio child) submitted from a
    Telegram handler task / a real ASGI request task must settle before
    that task settles, and shared cleanup must not begin until it has.
    Each fails against plain `asyncio.to_thread()` semantics and passes
    through submit_worker()/await_worker().
  - test_full_transitive_ordering_* — Section 13's central acceptance
    invariant end to end through run_unified_service(): every adapter-
    owned task AND every resource-sensitive worker descendant (Telegram
    handler + Uvicorn request, both real OS-thread workers) settle before
    close_resources() begins, recorded as an explicit ordered event trace.

Fourth corrective pass additions (independent re-audit — MAJOR: "reachable
storage executor outlives Telegram ownership"): utils.helpers.
save_file_async() and services.image_generation.download_image() used a
bare `await aiofiles.open()/write()` — a cancelled owning Task could
settle while the aiofiles executor thread was still writing. Both now
route their file write through submit_worker()/await_worker() via a new
private `_save_file_sync()`/`_write_image_bytes_sync()` seam:
  - test_telegram_handler_save_file_async_worker_cannot_outlive_
    adapter_settlement — the real save_file_async() (never a fake stand-in
    for the storage call) called from a fake Telegram handler task exactly
    like handlers/voice.py:105, through the real production
    _default_run_telegram()/_settle_telegram_pending_tasks() wiring; the
    real underlying write is wrapped (not replaced) with threading.Event
    control so its executor-thread lifetime is observable independently
    of the handler Task's cancellation.
  - test_save_file_async_worker_cannot_outlive_caller_cancellation —
    the same proof directly against save_file_async() alone (temp path,
    no service_main involvement): worker started, caller cancelled, caller
    stays pending, worker released, cancellation propagates, no
    unretrieved exception, the real write still completed.
  - test_download_image_storage_worker_cannot_outlive_caller_cancellation
    — the same proof against services.image_generation.download_image()'s
    own independent storage-worker boundary (Section 6: its filename
    convention differs from save_file_async()'s, so it keeps its own
    _write_image_bytes_sync() rather than reusing save_file_async()).
"""

import asyncio
import gc
import importlib.metadata
import threading
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI

import db.engine
import rag.index
import service_main
from utils.helpers import await_worker, submit_worker

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _leaked_tasks(baseline: set) -> set:
    """Every task still alive that wasn't already running before this
    test's composition-root call started, and isn't the current test
    coroutine itself — used to prove no background polling/server task is
    ever left alive past run_unified_service()'s own return/raise."""
    current = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t not in baseline and t is not current and not t.done()}


def _event_adapter():
    """A fake adapter that runs forever until its own stop() is called —
    stands in for "Telegram polling"/"Uvicorn serving" without touching
    either real system."""
    event = asyncio.Event()

    async def run() -> None:
        await event.wait()

    return run, event.set, event


class _TelegramSentinelError(Exception):
    pass


class _UvicornSentinelError(Exception):
    pass


class _CleanupSentinelError(Exception):
    pass


class _TelegramHandlerSentinelError(Exception):
    pass


class _FakeBotPendingTasks:
    """Stands in for AsyncTeleBot for _settle_telegram_pending_tasks()'s
    unit tests — only the one private attribute that function touches,
    never a real bot/Telegram network call."""

    def __init__(self) -> None:
        self._pending_tasks: set = set()


# --- normal lifecycle ---------------------------------------------------


async def test_normal_lifecycle_starts_both_adapters_and_closes_shared_resources_once():
    started = {"telegram": False, "uvicorn": False}
    tg_ready = asyncio.Event()
    uv_ready = asyncio.Event()
    tg_run, tg_stop, tg_stop_event = _event_adapter()
    uv_run, uv_stop, uv_stop_event = _event_adapter()

    async def run_telegram() -> None:
        started["telegram"] = True
        tg_ready.set()
        await tg_run()

    async def run_uvicorn() -> None:
        started["uvicorn"] = True
        uv_ready.set()
        await uv_run()

    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        service_main.run_unified_service(
            run_telegram=run_telegram,
            stop_telegram=tg_stop,
            run_uvicorn=run_uvicorn,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )
    )

    await asyncio.wait_for(tg_ready.wait(), timeout=1.0)
    await asyncio.wait_for(uv_ready.wait(), timeout=1.0)
    assert started == {"telegram": True, "uvicorn": True}

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tg_stop_event.is_set()
    assert uv_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


# --- Section 9: Telegram fails during startup ---------------------------


async def test_telegram_startup_failure_stops_uvicorn_and_exits_non_successfully():
    async def failing_telegram() -> None:
        raise _TelegramSentinelError("telegram setup failed")

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(_TelegramSentinelError):
        await service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert uv_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


# --- Section 9: Uvicorn fails during startup -----------------------------


async def test_uvicorn_startup_failure_stops_telegram_and_exits_non_successfully():
    async def failing_uvicorn() -> None:
        raise _UvicornSentinelError("uvicorn startup failed")

    tg_run, tg_stop, tg_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(_UvicornSentinelError):
        await service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=failing_uvicorn,
            stop_uvicorn=lambda: None,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert tg_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


# --- Section 9: Telegram polling raises during runtime -------------------


async def test_telegram_runtime_failure_stops_uvicorn_and_exits_non_successfully():
    started = asyncio.Event()

    async def failing_telegram() -> None:
        started.set()
        await asyncio.sleep(0.02)
        raise _TelegramSentinelError("telegram polling crashed mid-run")

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    with pytest.raises(_TelegramSentinelError):
        await service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert started.is_set()
    assert uv_stop_event.is_set()
    assert close_calls == [1]


# --- MINOR: simultaneous adapter failures --------------------------------


async def test_simultaneous_adapter_failures_deterministic_primary_and_no_unretrieved_exception():
    """
    Independent-audit MINOR finding: the previous `next(iter(done))
    .exception()` was nondeterministic set-iteration order and could leave
    a simultaneous second failure's exception unretrieved (a "Task
    exception was never retrieved" leak). Both adapters are released from
    the same asyncio.Event at once so they complete within the same
    scheduling round — the deterministic stable order (Telegram, then
    Uvicorn) must always pick Telegram as primary, and BOTH exceptions
    must be retrieved (proven via a custom loop exception handler, also
    covering the "no unobserved task exceptions" requirement for this
    path).
    """
    loop = asyncio.get_running_loop()
    unretrieved = []
    original_handler = loop.get_exception_handler()

    def handler(loop, context):
        message = str(context.get("message", ""))
        if "never retrieved" in message:
            unretrieved.append(context)
        elif original_handler is not None:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    try:
        ready = asyncio.Event()

        async def failing_telegram() -> None:
            await ready.wait()
            raise _TelegramSentinelError("telegram failed simultaneously")

        async def failing_uvicorn() -> None:
            await ready.wait()
            raise _UvicornSentinelError("uvicorn failed simultaneously")

        close_calls = []

        async def close_resources() -> None:
            close_calls.append(1)

        baseline = set(asyncio.all_tasks())
        task = asyncio.create_task(
            service_main.run_unified_service(
                run_telegram=failing_telegram,
                stop_telegram=lambda: None,
                run_uvicorn=failing_uvicorn,
                stop_uvicorn=lambda: None,
                close_resources=close_resources,
                shutdown_grace_seconds=1.0,
            )
        )
        await asyncio.sleep(0)  # let both tasks start and reach `ready.wait()`
        ready.set()  # release both "simultaneously" -- both finish in the same round

        with pytest.raises(_TelegramSentinelError):
            await task

        assert close_calls == [1]
        assert _leaked_tasks(baseline) == set()

        del task
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert unretrieved == []


# --- Section 5/9: an adapter finishing on its own (no raise) is itself ---
# --- an unexpected termination, symmetric for both adapters --------------


async def test_telegram_unexpected_normal_exit_stops_uvicorn():
    async def exiting_telegram() -> None:
        return  # returns without ever being asked to stop

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    with pytest.raises(service_main.UnifiedServiceAdapterExited):
        await service_main.run_unified_service(
            run_telegram=exiting_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert uv_stop_event.is_set()
    assert close_calls == [1]


async def test_uvicorn_unexpected_normal_exit_stops_telegram():
    async def exiting_uvicorn() -> None:
        return  # returns without ever being asked to stop

    tg_run, tg_stop, tg_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    with pytest.raises(service_main.UnifiedServiceAdapterExited):
        await service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=exiting_uvicorn,
            stop_uvicorn=lambda: None,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert tg_stop_event.is_set()
    assert close_calls == [1]


# --- Section 9: external cancellation / Ctrl+C ----------------------------


async def test_external_cancellation_stops_both_and_closes_resources_once():
    tg_run, tg_stop, tg_stop_event = _event_adapter()
    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )
    )
    await asyncio.sleep(0)  # let it start and reach the FIRST_COMPLETED wait
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tg_stop_event.is_set()
    assert uv_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


async def test_close_resources_runs_even_when_close_resources_itself_is_slow_but_within_grace():
    """close_resources() is awaited unconditionally by `_supervise()` —
    this is a lightweight sanity check that an async close_resources
    actually gets awaited to completion (not merely scheduled) before
    run_unified_service() returns/raises."""
    tg_run, tg_stop, _ = _event_adapter()
    uv_run, uv_stop, _ = _event_adapter()
    finished = asyncio.Event()

    async def close_resources() -> None:
        await asyncio.sleep(0.01)
        finished.set()

    task = asyncio.create_task(
        service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()


# --- MAJOR 1: cancellation while sibling teardown is already active ------


async def test_cancellation_during_already_in_progress_sibling_teardown_still_settles_and_closes_once():
    """
    Independent-audit MAJOR 1 regression. Telegram fails first (this
    becomes the determined primary failure via `done`); WHILE the
    sibling (Uvicorn) is being settled — specifically, while
    `_settle_tasks()`'s own bounded grace-period wait is already in
    progress — run_unified_service()'s own task is cancelled a SECOND
    time (e.g. a second Ctrl+C arriving mid-shutdown).

    The previous `_graceful_stop()` had no cancellation handling of its
    own: this second cancellation would have escaped mid-settlement,
    reaching close_resources() while the Uvicorn adapter task might still
    be alive/uncancelled. `_run_uncancellable()` must instead keep
    settlement running to completion regardless, and the already-
    determined Telegram failure (not CancelledError) must still be what
    ultimately propagates (Section 8: "the ORIGINAL... failure/
    cancellation semantics" — the adapter failure is what triggered
    teardown in the first place).
    """

    async def failing_telegram() -> None:
        raise _TelegramSentinelError("telegram failed first")

    grace_wait_started = asyncio.Event()
    uvicorn_force_cancelled = asyncio.Event()

    async def unresponsive_uvicorn() -> None:
        grace_wait_started.set()
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            uvicorn_force_cancelled.set()
            raise

    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=unresponsive_uvicorn,
            stop_uvicorn=lambda: None,  # signalled but deliberately ignored -- forces the grace wait
            close_resources=close_resources,
            shutdown_grace_seconds=0.2,
        )
    )

    await asyncio.wait_for(grace_wait_started.wait(), timeout=1.0)
    # Telegram has already failed; settlement of the sibling (Uvicorn) is
    # now inside its bounded grace-period wait (0.2s). Cancel
    # run_unified_service()'s own task WHILE that settlement is active.
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(_TelegramSentinelError):
        await task

    assert uvicorn_force_cancelled.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


async def test_graceful_stop_force_cancels_an_unresponsive_adapter_after_grace_period():
    """Every other test's fakes cooperate with their stop signal
    immediately, so none of them actually exercise `_settle_tasks()`'s
    force-cancel fallback (the sibling still gets a chance to finish its
    own shutdown path, but that chance is bounded)."""

    async def failing_telegram() -> None:
        raise _TelegramSentinelError("telegram failed")

    cancelled = asyncio.Event()

    async def unresponsive_uvicorn() -> None:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(_TelegramSentinelError):
        await service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=unresponsive_uvicorn,
            stop_uvicorn=lambda: None,  # signalled but deliberately ignored
            close_resources=close_resources,
            shutdown_grace_seconds=0.05,
        )

    assert cancelled.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


# --- MAJOR 2: Uvicorn's own lifespan child task must never leak ----------


async def test_uvicorn_forced_cancellation_during_hanging_lifespan_startup_settles_lifespan_task(monkeypatch):
    """
    Independent-audit MAJOR 2 regression, against the REAL installed
    uvicorn 0.52.4 (never a real socket bind — `LifespanOn.startup()`'s
    hang happens BEFORE `Server.startup()` ever reaches
    `loop.create_server()`; see uvicorn/server.py: `await self.lifespan.
    startup()` runs first, socket creation only after).

    The previous implementation's plain `await server.serve()` (no
    `_TrackedLifespanOn`/`_settle_uvicorn_lifespan_task()`) leaves
    uvicorn's own `LifespanOn.main()` task running forever once this
    adapter's own task is force-cancelled: `Task.cancel()` only
    interrupts `server.serve()`'s own coroutine chain, never the separate
    task `LifespanOn.startup()` created for itself via a bare
    `loop.create_task(...)`.

    This test exercises `service_main._build_uvicorn_adapter()` directly
    (only `create_app` is faked, to an ASGI app whose lifespan handler
    never completes startup) — a black-box `asyncio.all_tasks()` check is
    enough to prove the leak either way, without reaching into uvicorn's
    private attributes from the test itself.
    """

    async def hanging_lifespan_app(scope, receive, send) -> None:
        assert scope["type"] == "lifespan"
        await receive()  # consumes "lifespan.startup"
        await asyncio.Event().wait()  # never completes -- simulates a hung startup

    monkeypatch.setenv("WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("WEB_PORT", "0")
    monkeypatch.setattr(
        service_main, "create_app", lambda *, owns_db_lifecycle=True: hanging_lifespan_app
    )

    baseline = set(asyncio.all_tasks())
    run, stop = service_main._build_uvicorn_adapter()

    run_task = asyncio.create_task(run())
    # Let Server.serve() actually reach and get stuck inside
    # LifespanOn.startup()'s `await self.startup_event.wait()`.
    await asyncio.sleep(0.05)

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert _leaked_tasks(baseline) == set()


# --- MINOR: cleanup must not mask a primary failure/cancellation ---------


async def test_cleanup_failure_does_not_mask_a_primary_adapter_exception():
    async def failing_telegram() -> None:
        raise _TelegramSentinelError("telegram failed")

    uv_run, uv_stop, uv_stop_event = _event_adapter()

    async def failing_close_resources() -> None:
        raise _CleanupSentinelError("cleanup failed too")

    baseline = set(asyncio.all_tasks())
    with pytest.raises(_TelegramSentinelError):
        await service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=failing_close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert uv_stop_event.is_set()
    assert _leaked_tasks(baseline) == set()


async def test_cleanup_failure_does_not_mask_external_cancellation():
    tg_run, tg_stop, tg_stop_event = _event_adapter()
    uv_run, uv_stop, uv_stop_event = _event_adapter()

    async def failing_close_resources() -> None:
        raise _CleanupSentinelError("cleanup failed too")

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=failing_close_resources,
            shutdown_grace_seconds=1.0,
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tg_stop_event.is_set()
    assert uv_stop_event.is_set()
    assert _leaked_tasks(baseline) == set()


async def test_cleanup_failure_does_not_mask_an_unexpected_normal_adapter_exit():
    """Section 6's third listed masking case: an adapter that just
    returns without ever being asked to stop (UnifiedServiceAdapterExited)
    is a distinct "primary" trigger from a genuine crash exception or
    external cancellation — confirms it is held to the exact same
    cleanup-masking precedence rule as those two."""

    async def exiting_telegram() -> None:
        return

    uv_run, uv_stop, uv_stop_event = _event_adapter()

    async def failing_close_resources() -> None:
        raise _CleanupSentinelError("cleanup failed too")

    baseline = set(asyncio.all_tasks())
    with pytest.raises(service_main.UnifiedServiceAdapterExited):
        await service_main.run_unified_service(
            run_telegram=exiting_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=failing_close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert uv_stop_event.is_set()
    assert _leaked_tasks(baseline) == set()


# --- SystemExit / BaseException propagation -------------------------------


async def test_uvicorn_system_exit_propagates_stops_telegram_and_closes_resources_once():
    """
    Section 10: installed uvicorn's own `Server.startup()` calls
    `sys.exit(STARTUP_FAILURE)` (a real SystemExit) on a bind OSError
    (uvicorn/server.py) — never invoked for real here (no port bind),
    this only proves run_unified_service()'s own settlement/propagation
    contract for a BaseException that is neither Exception nor
    CancelledError.
    """

    async def system_exiting_uvicorn() -> None:
        raise SystemExit(3)

    tg_run, tg_stop, tg_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(SystemExit) as exc_info:
        await service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=system_exiting_uvicorn,
            stop_uvicorn=lambda: None,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert exc_info.value.code == 3
    assert tg_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


# --- MINOR: Telegram startup-phase shutdown --------------------------------


async def test_shutdown_requested_while_telegram_still_in_its_startup_path_settles_cleanly():
    """
    Independent-audit MINOR finding regression: shutdown requested while
    the Telegram adapter is still inside its OWN startup path (e.g.
    `bot.get_me()` inside `main.setup_bot()`, or `_process_polling()`'s
    own pre-loop `get_me()` call) — BEFORE `infinity_polling()`'s inner
    loop (whose own `except asyncio.CancelledError: return` is what makes
    mid-poll cancellation swallow cleanly, see `_default_stop_telegram()`'s
    docstring) is ever reached. `stop_telegram()` has no cooperative hook
    that could possibly help here (nothing is polling yet) — settlement
    must still complete via forced cancellation alone, never depending on
    `_polling` being observed.
    """
    startup_reached = asyncio.Event()

    async def slow_telegram_startup() -> None:
        startup_reached.set()
        await asyncio.sleep(100)  # stands in for a stuck bot.get_me()/setup_bot()

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        service_main.run_unified_service(
            run_telegram=slow_telegram_startup,
            stop_telegram=lambda: None,  # best-effort only -- no cooperative hook mid-get_me()
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=0.1,
        )
    )
    await asyncio.wait_for(startup_reached.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert uv_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


# --- production default wiring: _default_run_telegram --------------------


async def test_default_run_telegram_calls_setup_bot_then_infinity_polling(monkeypatch):
    calls = []

    async def fake_setup_bot() -> None:
        calls.append("setup")

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        calls.append(("infinity_polling", timeout, skip_pending))

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)
    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    await service_main._default_run_telegram()

    assert calls == ["setup", ("infinity_polling", 10, True)]


def test_default_stop_telegram_sets_the_polling_flag_false():
    service_main.bot._polling = True
    service_main._default_stop_telegram()
    assert service_main.bot._polling is False


# --- production default wiring: _build_uvicorn_adapter --------------------


def test_build_uvicorn_adapter_wires_env_host_port_and_owns_db_lifecycle_false(monkeypatch):
    monkeypatch.setenv("WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("WEB_PORT", "9999")

    create_app_calls = []
    sentinel_app = object()

    def fake_create_app(*, owns_db_lifecycle: bool = True):
        create_app_calls.append(owns_db_lifecycle)
        return sentinel_app

    config_calls = []
    config_instances = []

    class FakeConfig:
        def __init__(self, app, **kwargs):
            config_calls.append((app, kwargs))
            self.loaded = False
            self.lifespan_class = None
            config_instances.append(self)

        def load(self) -> None:
            self.loaded = True

    server_instances = []

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            server_instances.append(self)

        async def serve(self) -> None:
            pass

    monkeypatch.setattr(service_main, "create_app", fake_create_app)
    monkeypatch.setattr(service_main.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(service_main.uvicorn, "Server", FakeServer)

    run, stop = service_main._build_uvicorn_adapter()

    assert create_app_calls == [False]
    assert len(config_calls) == 1
    app_arg, kwargs = config_calls[0]
    assert app_arg is sentinel_app
    assert kwargs == {"host": "127.0.0.1", "port": 9999, "access_log": False}
    assert len(config_instances) == 1
    # Section: config.load() must run, and the lifespan_class override
    # must be applied, BEFORE uvicorn.Server(config) is constructed --
    # otherwise Server._serve()'s own `if not config.loaded: config.load()`
    # would silently recompute lifespan_class back to the plain default.
    assert config_instances[0].loaded is True
    assert config_instances[0].lifespan_class is service_main._TrackedLifespanOn
    assert len(server_instances) == 1
    assert server_instances[0].should_exit is False

    stop()
    assert server_instances[0].should_exit is True


# --- no duplicate Qdrant owner, production defaults end-to-end -----------


async def test_production_default_wiring_has_a_single_qdrant_owner_and_closes_shared_resources_once(monkeypatch):
    """
    Runs run_unified_service() with NO overrides — every seam it falls
    back to (_default_run_telegram/_default_stop_telegram/
    _build_uvicorn_adapter/main.shutdown_bot) is exercised for real, with
    only the actual Qdrant/DB/Telegram/socket I/O boundaries faked. Proves
    a single Qdrant owner ("the local Qdrant store is opened at most once
    per unified service process") and that the web adapter never
    constructs a second Qdrant client (create_app() here is called with
    owns_db_lifecycle=False and never itself touches get_vector_index() at
    all, since no Stage 7A-3 route exists yet).
    """
    monkeypatch.setenv("WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("WEB_PORT", "8000")

    get_vector_index_calls = []
    close_vector_index_calls = []

    class _FakeVectorIndex:
        def index_documents_directory(self, force_reindex: bool = False) -> int:
            return 0

    fake_index = _FakeVectorIndex()

    def fake_get_vector_index():
        get_vector_index_calls.append(1)
        return fake_index

    def fake_close_vector_index() -> None:
        close_vector_index_calls.append(1)

    monkeypatch.setattr(rag.index, "get_vector_index", fake_get_vector_index)
    monkeypatch.setattr(rag.index, "close_vector_index", fake_close_vector_index)

    close_db_calls = []
    monkeypatch.setattr(db.engine, "close_db", lambda: close_db_calls.append(1))

    async def fake_setup_bot() -> None:
        # Simulates exactly what main.py's real setup_bot() does to
        # Qdrant (RAG indexing through the shared singleton) — without
        # handler imports or a real Telegram get_me() network call.
        from rag.index import get_vector_index

        get_vector_index().index_documents_directory(force_reindex=False)

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        service_main.bot._polling = True
        while service_main.bot._polling:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    close_session_calls = []

    async def fake_close_session() -> None:
        close_session_calls.append(1)

    monkeypatch.setattr(service_main.bot, "close_session", fake_close_session)

    create_app_calls = []
    sentinel_app = object()

    def fake_create_app(*, owns_db_lifecycle: bool = True):
        create_app_calls.append(owns_db_lifecycle)
        return sentinel_app

    class FakeConfig:
        def __init__(self, app, **kwargs):
            self.app = app
            self.kwargs = kwargs
            self.loaded = False
            self.lifespan_class = None

        def load(self) -> None:
            self.loaded = True

    server_instances = []

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            server_instances.append(self)

        async def serve(self) -> None:
            while not self.should_exit:
                await asyncio.sleep(0.01)

    monkeypatch.setattr(service_main, "create_app", fake_create_app)
    monkeypatch.setattr(service_main.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(service_main.uvicorn, "Server", FakeServer)

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(service_main.run_unified_service(shutdown_grace_seconds=2.0))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert get_vector_index_calls == [1]
    assert close_vector_index_calls == [1]
    assert close_db_calls == [1]
    assert close_session_calls == [1]
    assert create_app_calls == [False]
    assert len(server_instances) == 1
    assert server_instances[0].should_exit is True
    assert _leaked_tasks(baseline) == set()


# --- Section 11: actual create_app(owns_db_lifecycle=...) lifespan -------


def test_create_app_owns_db_lifecycle_false_lifespan_does_not_close_db(monkeypatch):
    """
    Exercises web.app.create_app(owns_db_lifecycle=False)'s ACTUAL
    lifespan (through Starlette's real ASGI lifespan protocol via
    TestClient, not a fake server) — only app.auth_session.
    apply_startup_posture() (a real DB call in production) and db.engine.
    close_db() are faked, so no real PostgreSQL is touched. Confirms
    service_main.py's unified mode (which passes owns_db_lifecycle=False)
    never has this app's own lifespan racing main.shutdown_bot() to close
    the same shared DB engine.
    """
    from starlette.testclient import TestClient

    import app.auth_session as app_auth_session
    from web.app import create_app

    async def fake_apply_startup_posture(*, requested_secure: bool) -> int:
        return 0

    monkeypatch.setattr(app_auth_session, "apply_startup_posture", fake_apply_startup_posture)

    close_calls = []
    monkeypatch.setattr(db.engine, "close_db", lambda: close_calls.append(1))

    with TestClient(create_app(owns_db_lifecycle=False)) as client:
        response = client.get("/healthz")
        assert response.status_code == 200

    assert close_calls == []


def test_create_app_default_still_owns_db_lifecycle_and_closes_db(monkeypatch):
    """Standalone-mode regression counterpart to the test above:
    create_app()'s own default (owns_db_lifecycle=True, web_main.py's
    standalone behavior) must be completely unaffected by the new
    parameter — same real-lifespan proof, opposite outcome."""
    from starlette.testclient import TestClient

    import app.auth_session as app_auth_session
    from web.app import create_app

    async def fake_apply_startup_posture(*, requested_secure: bool) -> int:
        return 0

    monkeypatch.setattr(app_auth_session, "apply_startup_posture", fake_apply_startup_posture)

    close_calls = []
    monkeypatch.setattr(db.engine, "close_db", lambda: close_calls.append(1))

    with TestClient(create_app()) as client:
        response = client.get("/healthz")
        assert response.status_code == 200

    assert close_calls == [1]


# =========================================================================
# Second corrective pass — MAJOR: Telegram handler child tasks must be
# settled before "Telegram adapter settled" (AsyncTeleBot._pending_tasks)
# =========================================================================


async def test_settle_telegram_pending_tasks_is_a_noop_when_nothing_pending():
    fake_bot = _FakeBotPendingTasks()
    await service_main._settle_telegram_pending_tasks(fake_bot)  # must not raise
    assert fake_bot._pending_tasks == set()


async def test_settle_telegram_pending_tasks_cancels_a_still_live_handler_task():
    """
    Reproduces the previous failure directly: a handler task created the
    same way installed pyTelegramBotAPI 4.36.1's `_process_polling()`
    creates one (`asyncio.create_task(...)` + `add_done_callback(
    self._pending_tasks.discard)`) is still alive when settlement runs.
    Event-based, not sleep-timing-based (Section 5).
    """
    fake_bot = _FakeBotPendingTasks()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler() -> None:
        started.set()
        try:
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(handler())
    fake_bot._pending_tasks.add(task)
    task.add_done_callback(fake_bot._pending_tasks.discard)

    await asyncio.wait_for(started.wait(), timeout=1.0)
    await service_main._settle_telegram_pending_tasks(fake_bot)

    assert cancelled.is_set()
    assert task.done()
    assert task.cancelled()
    assert fake_bot._pending_tasks == set()


async def test_settle_telegram_pending_tasks_retrieves_exception_from_an_already_done_task():
    """
    Installed pyTelegramBotAPI's own `add_done_callback(self._pending_
    tasks.discard)` — like any `Future.add_done_callback()` on an
    already-done future — always defers via `call_soon`, never fires
    synchronously, so a handler task can be `done()` with an exception
    while still present in `_pending_tasks` for one event-loop iteration.
    Must not produce "Task exception was never retrieved" for it, and
    must not attempt to cancel an already-done task.
    """
    fake_bot = _FakeBotPendingTasks()

    async def failing_handler() -> None:
        raise _TelegramHandlerSentinelError("handler failed")

    task = asyncio.create_task(failing_handler())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert task.done()
    fake_bot._pending_tasks.add(task)  # added AFTER completion -- simulates the race window

    loop = asyncio.get_running_loop()
    unretrieved = []
    original_handler = loop.get_exception_handler()

    def handler(loop, context):
        message = str(context.get("message", ""))
        if "never retrieved" in message:
            unretrieved.append(context)
        elif original_handler is not None:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    try:
        await service_main._settle_telegram_pending_tasks(fake_bot)
        del task
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert unretrieved == []


async def test_default_run_telegram_settles_pending_handler_tasks_before_returning(monkeypatch):
    """
    Integration-level proof against the REAL module-level `bot` singleton
    that `_default_run_telegram()` actually wires `_settle_telegram_
    pending_tasks()` in — must fail against the previous implementation
    (a bare `await setup_bot(); await bot.infinity_polling(...)` with no
    such call at all, which would leave `handler_task` alive and never
    cancelled).
    """

    async def fake_setup_bot() -> None:
        pass

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)

    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()

    async def fake_handler() -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    handler_task = asyncio.create_task(fake_handler())
    service_main.bot._pending_tasks.add(handler_task)
    handler_task.add_done_callback(service_main.bot._pending_tasks.discard)

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        await handler_started.wait()
        return  # polling exits normally while the handler task is still alive

    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    try:
        await service_main._default_run_telegram()

        assert handler_cancelled.is_set()
        assert handler_task.done()
        assert handler_task.cancelled()
    finally:
        service_main.bot._pending_tasks.discard(handler_task)
        await asyncio.sleep(0)


async def test_default_run_telegram_settles_pending_tasks_even_when_polling_raises(monkeypatch):
    """The settlement `finally` must run for EVERY way `infinity_polling()`
    can end, not only a normal return — here it raises an ordinary
    exception."""

    async def fake_setup_bot() -> None:
        pass

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)

    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()

    async def fake_handler() -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    handler_task = asyncio.create_task(fake_handler())
    service_main.bot._pending_tasks.add(handler_task)
    handler_task.add_done_callback(service_main.bot._pending_tasks.discard)

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        await handler_started.wait()
        raise _TelegramSentinelError("polling crashed mid-run")

    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    try:
        with pytest.raises(_TelegramSentinelError):
            await service_main._default_run_telegram()

        assert handler_cancelled.is_set()
        assert handler_task.done()
    finally:
        service_main.bot._pending_tasks.discard(handler_task)
        await asyncio.sleep(0)


async def test_production_wiring_settles_telegram_pending_task_before_service_returns(monkeypatch):
    """
    End-to-end proof through run_unified_service() itself (not just
    `_default_run_telegram()` in isolation), using the REAL production
    `_default_run_telegram`/`_default_stop_telegram` wiring — only
    `setup_bot`/`bot.infinity_polling`/`bot.close_session` and the Uvicorn
    adapter are faked, exactly like `test_production_default_wiring_...`
    above. Proves the full ordering: polling settled, then the live
    Telegram handler task settled, THEN (and only then) shared
    close_resources() runs — with zero leaked tasks of any kind by the
    time the call returns.
    """

    async def fake_setup_bot() -> None:
        pass

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)

    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()

    async def fake_handler() -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    handler_task = asyncio.create_task(fake_handler())
    service_main.bot._pending_tasks.add(handler_task)
    handler_task.add_done_callback(service_main.bot._pending_tasks.discard)

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        service_main.bot._polling = True
        await handler_started.wait()
        while service_main.bot._polling:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    close_session_calls = []

    async def fake_close_session() -> None:
        close_session_calls.append(1)

    monkeypatch.setattr(service_main.bot, "close_session", fake_close_session)

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    try:
        baseline = set(asyncio.all_tasks())
        task = asyncio.create_task(
            service_main.run_unified_service(
                run_uvicorn=uv_run,
                stop_uvicorn=uv_stop,
                close_resources=close_resources,
                shutdown_grace_seconds=1.0,
            )
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert handler_cancelled.is_set()
        assert handler_task.done()
        assert uv_stop_event.is_set()
        assert close_calls == [1]
        assert _leaked_tasks(baseline) == set()
    finally:
        service_main.bot._pending_tasks.discard(handler_task)
        await asyncio.sleep(0)


# =========================================================================
# Second corrective pass — MAJOR: _run_uncancellable()'s own protected
# operation must be BaseException-safe (SystemExit/KeyboardInterrupt/a
# CancelledError the protected operation raises on itself)
# =========================================================================


async def test_run_uncancellable_returns_normally_on_protected_operation_success():
    calls = []

    async def protected() -> None:
        calls.append(1)

    result = await service_main._run_uncancellable(protected())

    assert result is None
    assert calls == [1]


async def test_run_uncancellable_propagates_ordinary_exception_from_protected_operation():
    async def protected() -> None:
        raise _CleanupSentinelError("ordinary failure")

    with pytest.raises(_CleanupSentinelError):
        await service_main._run_uncancellable(protected())


async def test_run_uncancellable_propagates_systemexit_from_protected_operation_preserving_code():
    """
    Independent-audit MAJOR (second corrective pass) regression. The
    previous implementation scheduled the raw protected coroutine as a
    bare child task (`asyncio.ensure_future(coro)`) — verified directly
    against installed CPython 3.12's own asyncio/tasks.py
    (`Task.__step_run_and_handle_result`, which re-raises SystemExit/
    KeyboardInterrupt straight out of Task.__step) and asyncio/events.py
    (`Handle._run()`, which explicitly re-raises them too instead of
    routing them through the ordinary callback-exception path) — meaning
    it would never reach this function's own `await asyncio.shield(task)`
    normally, escaping the event loop's own callback dispatch entirely
    instead. Must be preserved as the ORIGINAL SystemExit object (`.code`
    intact), not a synthesized new one.
    """

    async def protected() -> None:
        raise SystemExit(7)

    with pytest.raises(SystemExit) as exc_info:
        await service_main._run_uncancellable(protected())

    assert exc_info.value.code == 7


async def test_run_uncancellable_propagates_keyboardinterrupt_from_protected_operation():
    async def protected() -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        await service_main._run_uncancellable(protected())


async def test_run_uncancellable_propagates_a_cancelled_error_the_protected_operation_raises_on_itself():
    """Section 6: a CancelledError the protected operation raises on
    ITSELF must still be retrieved/propagated, kept structurally distinct
    from `was_cancelled` (external cancellation of the code AWAITING this
    call) — see _run_uncancellable()'s own docstring."""

    async def protected() -> None:
        raise asyncio.CancelledError("protected operation cancelled itself")

    with pytest.raises(asyncio.CancelledError):
        await service_main._run_uncancellable(protected())


async def test_run_uncancellable_lets_protected_operation_finish_before_raising_external_cancellation():
    ready = asyncio.Event()
    finished = []

    async def protected() -> None:
        await ready.wait()
        finished.append(1)

    task = asyncio.create_task(service_main._run_uncancellable(protected()))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert finished == []  # the protected operation must not be interrupted by caller cancellation
    ready.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished == [1]


async def test_run_uncancellable_survives_repeated_external_cancellation_without_restarting_protected_operation():
    ready = asyncio.Event()
    call_count = []

    async def protected() -> None:
        call_count.append(1)
        await ready.wait()

    task = asyncio.create_task(service_main._run_uncancellable(protected()))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    ready.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert call_count == [1]


# --- Section 7: the same cleanup-masking precedence, now for BaseExceptions ---


async def test_cleanup_systemexit_does_not_mask_a_primary_adapter_exception():
    async def failing_telegram() -> None:
        raise _TelegramSentinelError("telegram failed")

    uv_run, uv_stop, uv_stop_event = _event_adapter()

    async def failing_close_resources() -> None:
        raise SystemExit(9)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(_TelegramSentinelError):
        await service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=failing_close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert uv_stop_event.is_set()
    assert _leaked_tasks(baseline) == set()


async def test_cleanup_keyboardinterrupt_does_not_mask_a_primary_adapter_exception():
    async def failing_telegram() -> None:
        raise _TelegramSentinelError("telegram failed")

    uv_run, uv_stop, uv_stop_event = _event_adapter()

    async def failing_close_resources() -> None:
        raise KeyboardInterrupt()

    baseline = set(asyncio.all_tasks())
    with pytest.raises(_TelegramSentinelError):
        await service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=failing_close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert uv_stop_event.is_set()
    assert _leaked_tasks(baseline) == set()


async def test_cleanup_systemexit_does_not_mask_external_cancellation():
    tg_run, tg_stop, tg_stop_event = _event_adapter()
    uv_run, uv_stop, uv_stop_event = _event_adapter()

    async def failing_close_resources() -> None:
        raise SystemExit(9)

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=failing_close_resources,
            shutdown_grace_seconds=1.0,
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tg_stop_event.is_set()
    assert uv_stop_event.is_set()
    assert _leaked_tasks(baseline) == set()


# =========================================================================
# Second corrective pass — MINOR: an adapter cancelled BEFORE the
# composition root ever asked it to stop is an unexpected termination
# =========================================================================


async def test_telegram_self_raised_cancelled_error_stops_uvicorn_and_is_not_successful():
    """
    Independent-audit MINOR (second corrective pass) regression. Under
    the previous implementation, an adapter task found already cancelled
    in `_supervise()`'s initial `done` set was silently skipped by
    `_select_primary_exception()`, and (with no other primary failure and
    no external cancellation of run_unified_service() itself) the whole
    call would return SUCCESSFULLY — violating "any unrequested adapter
    termination must terminate the unified service non-successfully".
    """

    async def self_cancelling_telegram() -> None:
        raise asyncio.CancelledError()

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(service_main.UnifiedServiceAdapterCancelled):
        await service_main.run_unified_service(
            run_telegram=self_cancelling_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=uv_run,
            stop_uvicorn=uv_stop,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert uv_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


async def test_uvicorn_self_raised_cancelled_error_stops_telegram_and_is_not_successful():
    async def self_cancelling_uvicorn() -> None:
        raise asyncio.CancelledError()

    tg_run, tg_stop, tg_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(service_main.UnifiedServiceAdapterCancelled):
        await service_main.run_unified_service(
            run_telegram=tg_run,
            stop_telegram=tg_stop,
            run_uvicorn=self_cancelling_uvicorn,
            stop_uvicorn=lambda: None,
            close_resources=close_resources,
            shutdown_grace_seconds=1.0,
        )

    assert tg_stop_event.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


async def test_requested_sibling_cancellation_during_teardown_is_not_treated_as_unexpected():
    """
    Section 9 — the flip side of the two tests above: a sibling that gets
    cancelled LATER, by `_settle_tasks()`'s own force-cancel fallback
    during ordinary teardown (never present in `_supervise()`'s initial
    `done` set), must never itself become a new `UnifiedServiceAdapter
    Cancelled` primary — the genuine first failure must still propagate
    unchanged. Regression-adjacent to `test_graceful_stop_force_cancels_
    an_unresponsive_adapter_after_grace_period` above; asserted here
    explicitly against the new exception type.
    """

    async def failing_telegram() -> None:
        raise _TelegramSentinelError("telegram failed first")

    cancelled = asyncio.Event()

    async def unresponsive_uvicorn() -> None:
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    baseline = set(asyncio.all_tasks())
    with pytest.raises(_TelegramSentinelError):
        await service_main.run_unified_service(
            run_telegram=failing_telegram,
            stop_telegram=lambda: None,
            run_uvicorn=unresponsive_uvicorn,
            stop_uvicorn=lambda: None,  # signalled but deliberately ignored -- forces force-cancel
            close_resources=close_resources,
            shutdown_grace_seconds=0.05,
        )

    assert cancelled.is_set()
    assert close_calls == [1]
    assert _leaked_tasks(baseline) == set()


# =========================================================================
# Second corrective pass — MINOR: exact dependency pins for the two
# private-implementation integration points above
# =========================================================================


def _requirements_txt_active_lines(prefix: str) -> list:
    lines = (_PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [
        line.strip() for line in lines
        if line.strip().lower().startswith(prefix.lower()) and not line.strip().startswith("#")
    ]


def test_uvicorn_is_pinned_to_the_exact_verified_version_in_requirements():
    """`_TrackedLifespanOn` mirrors installed uvicorn 0.52.4's own private
    `LifespanOn.startup()` line-for-line — a bounded range (previously
    `uvicorn>=0.34.0,<1.0.0`) would silently let a future 0.x uvicorn
    release change that private implementation out from under this
    override. Must be an exact pin, not a range."""
    assert _requirements_txt_active_lines("uvicorn") == ["uvicorn==0.52.4"]


def test_pytelegrambotapi_is_pinned_to_the_exact_verified_version_in_requirements():
    """`_settle_telegram_pending_tasks()` relies on installed
    pyTelegramBotAPI 4.36.1's own private `AsyncTeleBot._pending_tasks`
    attribute — same reasoning as the uvicorn pin above."""
    assert _requirements_txt_active_lines("pytelegrambotapi") == ["pyTelegramBotAPI==4.36.1"]


def test_uvicorn_installed_version_matches_the_pin():
    assert importlib.metadata.version("uvicorn") == "0.52.4"


def test_pytelegrambotapi_installed_version_matches_the_pin():
    assert importlib.metadata.version("pyTelegramBotAPI") == "4.36.1"


# =========================================================================
# Third corrective pass — MAJOR: adapter settlement is not transitive to
# reachable worker/request descendants
# =========================================================================


class _FakeUvicornServerState:
    """Stands in for `uvicorn.server.ServerState` for
    `_settle_uvicorn_request_tasks()`'s own unit tests — only the two
    attributes that function touches (`connections`, `tasks`), never a
    real bound socket."""

    def __init__(self) -> None:
        self.connections: set = set()
        self.tasks: set = set()


class _FakeUvicornConnection:
    """Stands in for one live HTTP connection's protocol instance — only
    the one method (`shutdown()`) `_settle_uvicorn_request_tasks()` calls,
    mirroring installed uvicorn's own per-protocol `shutdown()` (see
    e.g. httptools_impl.py's `HttpToolsProtocol.shutdown()`)."""

    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeUvicornSocketServer:
    """Stands in for one `asyncio.base_events.Server` entry in `Server.
    servers` — only the one method (`close()`) `_settle_uvicorn_request_
    tasks()` calls."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeUvicornServer:
    """Stands in for `uvicorn.Server` for `_settle_uvicorn_request_
    tasks()`'s own unit tests — only the attributes that function reads
    (`server_state`, `servers`)."""

    def __init__(self) -> None:
        self.server_state = _FakeUvicornServerState()
        self.servers: list = []


async def test_settle_uvicorn_request_tasks_is_a_noop_when_nothing_pending():
    server = _FakeUvicornServer()
    await service_main._settle_uvicorn_request_tasks(server)  # must not raise
    assert server.server_state.tasks == set()


async def test_settle_uvicorn_request_tasks_is_a_noop_when_server_state_is_missing():
    """`Server._serve()` sets `self.server_state` in `__init__` (always
    present in the real library), but this defensive guard mirrors
    `_settle_uvicorn_lifespan_task()`'s own `getattr(..., None)` style for
    a server object that never reached that point."""

    class _BareObject:
        pass

    await service_main._settle_uvicorn_request_tasks(_BareObject())  # must not raise


async def test_settle_uvicorn_request_tasks_closes_listeners_and_shuts_down_connections():
    server = _FakeUvicornServer()
    sock_server = _FakeUvicornSocketServer()
    server.servers.append(sock_server)
    connection = _FakeUvicornConnection()
    server.server_state.connections.add(connection)

    await service_main._settle_uvicorn_request_tasks(server)

    assert sock_server.close_calls == 1
    assert connection.shutdown_calls == 1


async def test_settle_uvicorn_request_tasks_cancels_a_still_live_request_task():
    """Reproduces the previous failure directly: a request task created
    the same way installed uvicorn's own `_start_asgi_task()` creates one
    (`loop.create_task(...)` + `add_done_callback(self.tasks.discard)`) is
    still alive when settlement runs. Event-based, not sleep-timing-based."""
    server = _FakeUvicornServer()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def request_handler() -> None:
        started.set()
        try:
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(request_handler())
    server.server_state.tasks.add(task)
    task.add_done_callback(server.server_state.tasks.discard)

    await asyncio.wait_for(started.wait(), timeout=1.0)
    await service_main._settle_uvicorn_request_tasks(server)

    assert cancelled.is_set()
    assert task.done()
    assert task.cancelled()
    assert server.server_state.tasks == set()


async def test_settle_uvicorn_request_tasks_retrieves_exception_from_an_already_done_task():
    """Same race window as `_settle_telegram_pending_tasks()`'s own
    identical test: installed uvicorn's `task.add_done_callback(self.
    tasks.discard)` always defers via `call_soon`, so a request task can
    be `done()` with an exception while still present in `server_state.
    tasks` for one event-loop iteration. Must not produce "Task exception
    was never retrieved" for it, and must not attempt to cancel an
    already-done task."""
    server = _FakeUvicornServer()

    class _RequestHandlerSentinelError(Exception):
        pass

    async def failing_request_handler() -> None:
        raise _RequestHandlerSentinelError("request handler failed")

    task = asyncio.create_task(failing_request_handler())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert task.done()
    server.server_state.tasks.add(task)  # added AFTER completion -- simulates the race window

    loop = asyncio.get_running_loop()
    unretrieved = []
    original_handler = loop.get_exception_handler()

    def handler(loop, context):
        message = str(context.get("message", ""))
        if "never retrieved" in message:
            unretrieved.append(context)
        elif original_handler is not None:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    try:
        await service_main._settle_uvicorn_request_tasks(server)
        del task
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    assert unretrieved == []


# --- Section 12: a REAL uvicorn.Server must never leak a live ASGI request
# task past _build_uvicorn_adapter()'s own run() closure ------------------


def _build_test_uvicorn_server(app) -> "uvicorn.Server":
    """Constructs a real `uvicorn.Server` bound to 127.0.0.1 on an
    OS-assigned ephemeral port (`port=0`) — loopback-only, never a fixed
    or externally-reachable port (Section 9) — wired with the SAME
    `_TrackedLifespanOn` override `_build_uvicorn_adapter()` itself
    installs, so `_settle_uvicorn_lifespan_task()` behaves identically.
    Mirrors `test_uvicorn_forced_cancellation_during_hanging_lifespan_
    startup_leaves_no_dangling_lifespan_task`'s own precedent for binding a
    real loopback socket directly in this test module."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, access_log=False)
    config.load()
    config.lifespan_class = service_main._TrackedLifespanOn
    return uvicorn.Server(config)


async def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition never became true within timeout")
        await asyncio.sleep(interval)


def _bound_port(server: "uvicorn.Server") -> int:
    return server.servers[0].sockets[0].getsockname()[1]


async def test_uvicorn_request_task_cannot_outlive_forced_adapter_cancellation():
    """
    Section 12 regression: a pure-async (no worker thread) in-flight ASGI
    request task must not survive `_build_uvicorn_adapter()`'s own `run()`
    closure ending, even when the OUTER `server.serve()` Task is
    force-cancelled while that request is still in flight — exactly the
    scenario `_settle_tasks()`'s own force-cancel fallback (or nested
    `_run_uncancellable()` cancellation) produces in production. Must fail
    against the previous implementation (`run()`'s `finally` only called
    `_settle_uvicorn_lifespan_task()`, never touching `server_state.tasks`
    at all).
    """
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    app = FastAPI()

    @app.get("/slow")
    async def slow():
        request_started.set()
        await release_request.wait()
        return {"ok": True}

    server = _build_test_uvicorn_server(app)

    async def run() -> None:
        try:
            await server.serve()
        finally:
            await service_main._settle_uvicorn_request_tasks(server)
            await service_main._settle_uvicorn_lifespan_task(server)

    baseline = set(asyncio.all_tasks())
    run_task = asyncio.create_task(run())
    await _wait_until(lambda: server.started)
    port = _bound_port(server)

    import httpx

    async with httpx.AsyncClient() as client:
        request_task = asyncio.create_task(client.get(f"http://127.0.0.1:{port}/slow", timeout=5.0))
        await asyncio.wait_for(request_started.wait(), timeout=2.0)
        assert len(server.server_state.tasks) == 1

        # Forced cancellation while the request is still in flight --
        # never a cooperative should_exit stop.
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert server.server_state.tasks == set()
        assert server.server_state.connections == set()

        # Whatever the CLIENT observed (uvicorn sends a 500 response for
        # an unhandled server-side exception when possible; a connection
        # reset raises instead) is irrelevant to this regression -- what
        # matters (already asserted above) is that the SERVER-side request
        # task and connection genuinely settled before run_task returned.
        # gather(..., return_exceptions=True) retrieves either outcome
        # without raising, so nothing is left unretrieved either way.
        release_request.set()
        await asyncio.wait_for(asyncio.gather(request_task, return_exceptions=True), timeout=5.0)

    assert _leaked_tasks(baseline) == set()


# --- Sections 10/11: a real OS-thread worker submitted from an adapter's
# own task must settle before that task, and before shared cleanup -------


async def test_telegram_handler_resource_worker_cannot_outlive_adapter_settlement(monkeypatch):
    """
    Section 10 regression. Uses a REAL thread worker controlled by
    `threading.Event` (submit_worker()/await_worker() — never merely a
    fake asyncio child): a Telegram handler task submits resource-
    sensitive work, the worker starts and blocks on a real OS thread, the
    handler's own task is cancelled by `_settle_telegram_pending_tasks()`
    — but the worker thread is still running. Must fail under plain
    `asyncio.to_thread()` semantics (the handler task settles the instant
    cancellation is delivered, regardless of the thread) and pass through
    submit_worker()/await_worker().
    """

    async def fake_setup_bot() -> None:
        pass

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)

    async def fake_close_session() -> None:
        pass

    monkeypatch.setattr(service_main.bot, "close_session", fake_close_session)

    handler_started = asyncio.Event()
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    def blocking_resource_call() -> str:
        worker_started.set()
        assert release_worker.wait(timeout=5.0), "release_worker was never set by the test"
        worker_finished.set()
        return "ok"

    async def fake_handler() -> None:
        handler_started.set()
        await await_worker(submit_worker(blocking_resource_call))

    handler_task = asyncio.create_task(fake_handler())
    service_main.bot._pending_tasks.add(handler_task)
    handler_task.add_done_callback(service_main.bot._pending_tasks.discard)

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        service_main.bot._polling = True
        await handler_started.wait()
        while service_main.bot._polling:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    try:
        baseline = set(asyncio.all_tasks())
        task = asyncio.create_task(
            service_main.run_unified_service(
                run_uvicorn=uv_run,
                stop_uvicorn=uv_stop,
                close_resources=close_resources,
                shutdown_grace_seconds=2.0,
            )
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
        await asyncio.get_event_loop().run_in_executor(None, worker_started.wait, 5.0)
        assert worker_started.is_set()

        task.cancel()

        # The worker thread is still blocked -- neither the handler task
        # nor shared cleanup may have settled yet, no matter how many
        # event-loop iterations pass.
        for _ in range(30):
            await asyncio.sleep(0.01)
        assert not handler_task.done()
        assert close_calls == []

        release_worker.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert worker_finished.is_set()
        assert handler_task.done()
        assert handler_task.cancelled()
        assert uv_stop_event.is_set()
        assert close_calls == [1]
        assert _leaked_tasks(baseline) == set()
    finally:
        release_worker.set()
        service_main.bot._pending_tasks.discard(handler_task)
        await asyncio.sleep(0)


async def test_web_request_resource_worker_cannot_outlive_adapter_settlement():
    """
    Section 11 regression — the web-request equivalent of the Telegram
    test above, through a REAL uvicorn.Server/real loopback socket/real
    ASGI request task. A route submits resource-sensitive work via
    submit_worker()/await_worker(); the worker thread is deliberately held
    open by a real threading.Event while `run()`'s own `finally` (forced
    cancellation of `server.serve()`, exactly like `_settle_tasks()`'s own
    fallback) is already settling. Shared cleanup must not be able to
    proceed until the worker genuinely finishes.
    """
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    app = FastAPI()

    @app.get("/slow")
    async def slow():
        def blocking_resource_call() -> str:
            worker_started.set()
            assert release_worker.wait(timeout=5.0), "release_worker was never set by the test"
            worker_finished.set()
            return "ok"

        await await_worker(submit_worker(blocking_resource_call))
        return {"ok": True}

    server = _build_test_uvicorn_server(app)
    settled = asyncio.Event()

    async def run() -> None:
        try:
            await server.serve()
        finally:
            await service_main._settle_uvicorn_request_tasks(server)
            await service_main._settle_uvicorn_lifespan_task(server)
            settled.set()

    baseline = set(asyncio.all_tasks())
    run_task = asyncio.create_task(run())
    await _wait_until(lambda: server.started)
    port = _bound_port(server)

    import httpx

    async with httpx.AsyncClient() as client:
        request_task = asyncio.create_task(client.get(f"http://127.0.0.1:{port}/slow", timeout=5.0))

        await asyncio.get_event_loop().run_in_executor(None, worker_started.wait, 5.0)
        assert worker_started.is_set()

        # Force-cancel while the worker thread is still blocked.
        run_task.cancel()

        for _ in range(30):
            await asyncio.sleep(0.01)
        assert not settled.is_set()
        assert not worker_finished.is_set()
        assert not run_task.done()

        release_worker.set()

        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert worker_finished.is_set()
        assert settled.is_set()
        assert server.server_state.tasks == set()

        # See the analogous comment in
        # test_uvicorn_request_task_cannot_outlive_forced_adapter_
        # cancellation -- the client-observed outcome is irrelevant here;
        # what matters (already asserted above) is that the worker thread
        # and the server-side request task both genuinely settled first.
        await asyncio.wait_for(asyncio.gather(request_task, return_exceptions=True), timeout=5.0)

    assert _leaked_tasks(baseline) == set()


async def test_full_transitive_ordering_telegram_and_uvicorn_workers_settle_before_close_resources(monkeypatch):
    """
    Section 13's central acceptance invariant, end to end through
    run_unified_service(): a REAL Telegram handler worker AND a REAL
    Uvicorn request worker (both real OS threads, controlled by
    threading.Event) must BOTH settle — and their owning adapter tasks
    must both settle — strictly BEFORE close_resources() begins. Recorded
    as an explicit ordered event trace rather than inferred from timing.
    """
    events: list = []
    events_lock = threading.Lock()

    def record(label: str) -> None:
        with events_lock:
            events.append(label)

    async def fake_setup_bot() -> None:
        pass

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)

    async def fake_close_session() -> None:
        pass

    monkeypatch.setattr(service_main.bot, "close_session", fake_close_session)

    # --- Telegram side ---------------------------------------------------
    handler_started = asyncio.Event()
    tg_release_worker = threading.Event()

    def tg_blocking_resource_call() -> None:
        tg_release_worker.wait(timeout=5.0)
        record("telegram worker done")

    async def fake_handler() -> None:
        handler_started.set()
        try:
            await await_worker(submit_worker(tg_blocking_resource_call))
        except asyncio.CancelledError:
            record("telegram handler cancelled")
            raise

    handler_task = asyncio.create_task(fake_handler())
    service_main.bot._pending_tasks.add(handler_task)
    handler_task.add_done_callback(service_main.bot._pending_tasks.discard)

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        service_main.bot._polling = True
        await handler_started.wait()
        while service_main.bot._polling:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    # --- Uvicorn side ------------------------------------------------------
    web_release_worker = threading.Event()
    web_request_started = asyncio.Event()

    app = FastAPI()

    @app.get("/slow")
    async def slow():
        def web_blocking_resource_call() -> None:
            web_release_worker.wait(timeout=5.0)
            record("web worker done")

        web_request_started.set()
        await await_worker(submit_worker(web_blocking_resource_call))
        return {"ok": True}

    server = _build_test_uvicorn_server(app)

    async def run_uvicorn() -> None:
        try:
            await server.serve()
        except asyncio.CancelledError:
            record("web request cancelled")
            raise
        finally:
            await service_main._settle_uvicorn_request_tasks(server)
            await service_main._settle_uvicorn_lifespan_task(server)
            record("uvicorn adapter done")

    def stop_uvicorn() -> None:
        server.should_exit = True

    async def close_resources() -> None:
        record("close_resources begins")
        record("close_resources ends")

    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(
        service_main.run_unified_service(
            run_uvicorn=run_uvicorn,
            stop_uvicorn=stop_uvicorn,
            close_resources=close_resources,
            shutdown_grace_seconds=2.0,
        )
    )

    await asyncio.wait_for(handler_started.wait(), timeout=1.0)
    await _wait_until(lambda: server.started)
    port = _bound_port(server)

    import httpx

    try:
        async with httpx.AsyncClient() as client:
            request_task = asyncio.create_task(client.get(f"http://127.0.0.1:{port}/slow", timeout=5.0))
            await asyncio.wait_for(web_request_started.wait(), timeout=2.0)

            # Force everything down together while BOTH real worker
            # threads are still blocked.
            task.cancel()

            for _ in range(30):
                await asyncio.sleep(0.01)
            assert "close_resources begins" not in events

            # Release the Telegram worker first -- its adapter must
            # settle without needing the web worker to have finished too.
            tg_release_worker.set()
            for _ in range(200):
                if "telegram worker done" in events:
                    break
                await asyncio.sleep(0.01)
            assert "telegram worker done" in events
            assert "close_resources begins" not in events

            web_release_worker.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            await asyncio.wait_for(asyncio.gather(request_task, return_exceptions=True), timeout=5.0)
    finally:
        tg_release_worker.set()
        web_release_worker.set()
        service_main.bot._pending_tasks.discard(handler_task)
        await asyncio.sleep(0)

    assert events.index("telegram worker done") < events.index("close_resources begins")
    assert events.index("web worker done") < events.index("close_resources begins")
    assert events.index("uvicorn adapter done") < events.index("close_resources begins")
    assert events.index("close_resources begins") < events.index("close_resources ends")
    assert _leaked_tasks(baseline) == set()


# =========================================================================
# Fourth corrective pass — MAJOR: "reachable storage executor outlives
# Telegram ownership" (utils.helpers.save_file_async() / services.
# image_generation.download_image() used a bare `aiofiles.open()/write()`
# await instead of submit_worker()/await_worker())
# =========================================================================


async def test_telegram_handler_save_file_async_worker_cannot_outlive_adapter_settlement(monkeypatch):
    """
    Section 8 regression. Exercises the REAL utils.helpers.save_file_async()
    — the exact function handlers/voice.py:105 calls for every real voice
    message — called from a fake Telegram handler task, through the real
    production `_default_run_telegram()`/`_settle_telegram_pending_tasks()`
    wiring (only setup_bot/infinity_polling/close_session are faked,
    exactly like test_telegram_handler_resource_worker_cannot_outlive_
    adapter_settlement above). The storage call itself is never faked:
    save_file_async()'s own worker (`_save_file_sync`) is wrapped (not
    replaced) so it still performs the real file write, with real
    threading.Event control added around it so the executor-thread's
    lifetime can be observed independently of the handler Task's
    cancellation. Must fail against the previous plain-`aiofiles.open()`/
    `.write()` implementation (manually verified — that implementation has
    no `_save_file_sync` seam at all, and the handler Task settles the
    instant cancellation is delivered regardless of the underlying
    executor thread) and pass through submit_worker()/await_worker().
    """
    import utils.helpers as helpers_module

    async def fake_setup_bot() -> None:
        pass

    monkeypatch.setattr(service_main, "setup_bot", fake_setup_bot)

    async def fake_close_session() -> None:
        pass

    monkeypatch.setattr(service_main.bot, "close_session", fake_close_session)

    original_save_file_sync = helpers_module._save_file_sync

    handler_started = asyncio.Event()
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    captured_path: dict = {}

    def blocking_save_file_sync(filepath, content):
        captured_path["path"] = filepath
        worker_started.set()
        assert release_worker.wait(timeout=5.0), "release_worker was never set by the test"
        result = original_save_file_sync(filepath, content)
        worker_finished.set()
        return result

    monkeypatch.setattr(helpers_module, "_save_file_sync", blocking_save_file_sync)

    async def fake_handler() -> None:
        handler_started.set()
        await helpers_module.save_file_async(
            b"fake ogg bytes - stage7a3 fourth corrective pass proof", "ogg"
        )

    handler_task = asyncio.create_task(fake_handler())
    service_main.bot._pending_tasks.add(handler_task)
    handler_task.add_done_callback(service_main.bot._pending_tasks.discard)

    async def fake_infinity_polling(*, timeout=None, skip_pending=None) -> None:
        service_main.bot._polling = True
        await handler_started.wait()
        while service_main.bot._polling:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(service_main.bot, "infinity_polling", fake_infinity_polling)

    uv_run, uv_stop, uv_stop_event = _event_adapter()
    close_calls = []

    async def close_resources() -> None:
        close_calls.append(1)

    try:
        baseline = set(asyncio.all_tasks())
        task = asyncio.create_task(
            service_main.run_unified_service(
                run_uvicorn=uv_run,
                stop_uvicorn=uv_stop,
                close_resources=close_resources,
                shutdown_grace_seconds=2.0,
            )
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
        await asyncio.get_event_loop().run_in_executor(None, worker_started.wait, 5.0)
        assert worker_started.is_set()

        task.cancel()

        # The worker thread is still blocked -- neither the handler task
        # nor shared cleanup may have settled yet, no matter how many
        # event-loop iterations pass.
        for _ in range(30):
            await asyncio.sleep(0.01)
        assert not handler_task.done()
        assert close_calls == []

        release_worker.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert worker_finished.is_set()
        assert handler_task.done()
        assert handler_task.cancelled()
        assert uv_stop_event.is_set()
        assert close_calls == [1]
        assert _leaked_tasks(baseline) == set()

        # The real write genuinely happened -- proves the storage call was
        # never faked, only its timing was put under test control.
        saved_path = captured_path["path"]
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"fake ogg bytes - stage7a3 fourth corrective pass proof"
    finally:
        release_worker.set()
        service_main.bot._pending_tasks.discard(handler_task)
        if captured_path.get("path") is not None:
            helpers_module.cleanup_file(captured_path["path"])
        await asyncio.sleep(0)


async def test_save_file_async_worker_cannot_outlive_caller_cancellation(monkeypatch):
    """
    Section 9 regression: save_file_async() itself, cancelled directly
    (not through the full service_main composition root, and using a
    temporary test path only). Proves the caller's own Task cannot settle
    until the real underlying write has finished, that the original
    CancelledError (not a worker exception) is what the caller observes,
    and that nothing is left as an unretrieved worker exception.
    """
    import utils.helpers as helpers_module

    original_save_file_sync = helpers_module._save_file_sync

    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    captured_path: dict = {}

    def blocking_save_file_sync(filepath, content):
        captured_path["path"] = filepath
        worker_started.set()
        assert release_worker.wait(timeout=5.0), "release_worker was never set by the test"
        result = original_save_file_sync(filepath, content)
        worker_finished.set()
        return result

    monkeypatch.setattr(helpers_module, "_save_file_sync", blocking_save_file_sync)

    async def caller() -> None:
        await helpers_module.save_file_async(b"direct helper regression proof", "tmp")

    loop = asyncio.get_running_loop()
    unretrieved = []
    original_handler = loop.get_exception_handler()

    def handler(loop, context):
        message = str(context.get("message", ""))
        if "never retrieved" in message:
            unretrieved.append(context)
        elif original_handler is not None:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    try:
        task = asyncio.create_task(caller())
        await asyncio.get_event_loop().run_in_executor(None, worker_started.wait, 5.0)
        assert worker_started.is_set()

        task.cancel()

        for _ in range(30):
            await asyncio.sleep(0.01)
        assert not task.done()

        release_worker.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert worker_finished.is_set()
        assert task.cancelled()
        saved_path = captured_path["path"]
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"direct helper regression proof"

        del task
        gc.collect()
        await asyncio.sleep(0)
    finally:
        release_worker.set()
        loop.set_exception_handler(original_handler)
        if captured_path.get("path") is not None:
            helpers_module.cleanup_file(captured_path["path"])

    assert unretrieved == []


async def test_download_image_storage_worker_cannot_outlive_caller_cancellation(monkeypatch):
    """
    Section 10 regression — services.image_generation.download_image()
    (the DALL-E URL-fallback storage path reachable from
    app.tutor.route_image_generation_request() -> handlers/text.py's image
    intent flow, and from generate_image_variations()) keeps its own
    independent storage-worker boundary rather than reusing
    save_file_async(): its filename convention (`generated_<timestamp>.png`
    under GENERATED_IMAGES_DIR) differs from save_file_async()'s
    (`<uuid>.<ext>` under DATA_DIR directly), so reusing it would have
    changed the filename scheme (Section 6 explicitly forbids that). Only
    aiohttp itself is faked here (no real network call); the real
    _write_image_bytes_sync() still performs the real file write, wrapped
    with real threading.Event control.
    """
    import services.image_generation as image_generation

    class _FakeResponse:
        status = 200

        async def read(self) -> bytes:
            return b"fake png bytes - stage7a3 fourth corrective pass proof"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info) -> bool:
            return False

    class _FakeSession:
        def get(self, url):
            return _FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info) -> bool:
            return False

    monkeypatch.setattr(image_generation.aiohttp, "ClientSession", lambda: _FakeSession())

    original_write = image_generation._write_image_bytes_sync

    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    captured_path: dict = {}

    def blocking_write(filepath, data):
        captured_path["path"] = filepath
        worker_started.set()
        assert release_worker.wait(timeout=5.0), "release_worker was never set by the test"
        result = original_write(filepath, data)
        worker_finished.set()
        return result

    monkeypatch.setattr(image_generation, "_write_image_bytes_sync", blocking_write)

    task = asyncio.create_task(image_generation.download_image("https://example.invalid/fake.png"))
    try:
        await asyncio.get_event_loop().run_in_executor(None, worker_started.wait, 5.0)
        assert worker_started.is_set()

        task.cancel()

        for _ in range(30):
            await asyncio.sleep(0.01)
        assert not task.done()

        release_worker.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert worker_finished.is_set()
        assert task.cancelled()
        saved_path = captured_path["path"]
        assert saved_path.exists()
        assert saved_path.read_bytes() == b"fake png bytes - stage7a3 fourth corrective pass proof"
    finally:
        release_worker.set()
        if captured_path.get("path") is not None:
            from utils.helpers import cleanup_file
            cleanup_file(captured_path["path"])
