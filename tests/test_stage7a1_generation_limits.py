"""
Stage 7A-1 corrective-pass regression tests:
app.generation_limits.GenerationAdmissionController — FAIL-FAST admission
semantics (replaces the earlier queuing design: a second concurrent
request for the same user, or a request arriving when the global cap is
exhausted, is rejected IMMEDIATELY with GenerationBusyError, never queued).

Every proof below is either a plain synchronous invariant (acquire/release
are pure, non-awaiting state transitions guarded by a threading.Lock — see
app/generation_limits.py's own docstring for why that is genuinely
cross-event-loop-safe, unlike a persisted asyncio.Lock/Semaphore) or uses
asyncio.Event-based deterministic handoffs for genuinely concurrent
scenarios. A bare `asyncio.sleep(0)` (used sparingly below) is ONLY a pure
scheduling yield — never evidence of ordering or the absence of a race.
Each test uses its own fresh GenerationAdmissionController instance (never
the shared production singleton) so tests can never interfere with each
other's per-user/global state.
"""

import asyncio
import threading
import uuid

import pytest

from app.generation_limits import (
    GenerationAdmissionController,
    GenerationBusyError,
    GenerationPermit,
    GenerationPermitError,
)


def _controller(*, max_per_user=1, max_global=4) -> GenerationAdmissionController:
    return GenerationAdmissionController(max_per_user=max_per_user, max_global=max_global)


# ============================================================================
# A. Basic admission: first request in; a second concurrent request for the
#    SAME user is rejected immediately — never queued, never runs the
#    provider body, never changes the global active count.
# ============================================================================


async def test_first_request_is_admitted():
    controller = _controller()
    user_id = uuid.uuid4()
    async with controller.acquire(user_id):
        assert controller.registry_size() == 1
        assert controller.global_active_count() == 1
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


async def test_second_concurrent_request_same_user_is_rejected_immediately():
    controller = _controller()
    user_id = uuid.uuid4()

    holding = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with controller.acquire(user_id):
            holding.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await holding.wait()

    with pytest.raises(GenerationBusyError):
        async with controller.acquire(user_id):
            pytest.fail("must never enter the body — admission must be rejected before this point")

    release.set()
    await asyncio.wait_for(task, timeout=5)


async def test_second_request_same_user_never_calls_provider_body():
    controller = _controller()
    user_id = uuid.uuid4()

    holding = asyncio.Event()
    release = asyncio.Event()
    second_body_entered = {"flag": False}

    async def holder():
        async with controller.acquire(user_id):
            holding.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await holding.wait()

    try:
        async with controller.acquire(user_id):
            second_body_entered["flag"] = True  # must never run
    except GenerationBusyError:
        pass

    assert second_body_entered["flag"] is False

    release.set()
    await asyncio.wait_for(task, timeout=5)


async def test_second_request_same_user_does_not_change_global_active_count():
    controller = _controller(max_global=4)
    user_id = uuid.uuid4()

    holding = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with controller.acquire(user_id):
            holding.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await holding.wait()
    assert controller.global_active_count() == 1

    with pytest.raises(GenerationBusyError):
        async with controller.acquire(user_id):
            pass

    assert controller.global_active_count() == 1  # unchanged by the rejected attempt

    release.set()
    await asyncio.wait_for(task, timeout=5)
    assert controller.global_active_count() == 0


async def test_nested_acquire_for_same_user_is_rejected_not_deadlocked():
    """Fail-fast semantics mean a caller that mistakenly tries to acquire
    twice for the same generation is rejected immediately with
    GenerationBusyError — never silently re-entered, and never deadlocked
    (unlike the earlier queuing design, where a nested acquire for the same
    user would have blocked forever on the caller's own already-held
    permit)."""
    controller = _controller(max_per_user=1)
    user_id = uuid.uuid4()

    async with controller.acquire(user_id):
        with pytest.raises(GenerationBusyError):
            async with controller.acquire(user_id):
                pytest.fail("must never reach here")


# ============================================================================
# B. Different users run in parallel up to the global cap; the (N+1)th
#    distinct user is rejected while the cap is exhausted.
# ============================================================================


async def test_different_users_run_in_parallel_up_to_global_cap():
    controller = _controller(max_per_user=1, max_global=4)
    user_ids = [uuid.uuid4() for _ in range(4)]

    all_holding = asyncio.Event()
    holding_count = {"n": 0}
    release_all = asyncio.Event()

    async def holder(uid):
        async with controller.acquire(uid):
            holding_count["n"] += 1
            if holding_count["n"] == 4:
                all_holding.set()
            await release_all.wait()

    tasks = [asyncio.create_task(holder(uid)) for uid in user_ids]
    await asyncio.wait_for(all_holding.wait(), timeout=5)
    assert controller.global_active_count() == 4

    release_all.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert controller.global_active_count() == 0


async def test_fifth_distinct_user_is_rejected_when_global_cap_reached():
    controller = _controller(max_per_user=1, max_global=4)
    user_ids = [uuid.uuid4() for _ in range(5)]

    all_holding = asyncio.Event()
    holding_count = {"n": 0}
    release_all = asyncio.Event()

    async def holder(uid):
        async with controller.acquire(uid):
            holding_count["n"] += 1
            if holding_count["n"] == 4:
                all_holding.set()
            await release_all.wait()

    tasks = [asyncio.create_task(holder(uid)) for uid in user_ids[:4]]
    await asyncio.wait_for(all_holding.wait(), timeout=5)
    assert controller.global_active_count() == 4

    with pytest.raises(GenerationBusyError):
        async with controller.acquire(user_ids[4]):
            pytest.fail("the fifth distinct user must be rejected while the global cap is exhausted")

    assert controller.global_active_count() == 4  # the rejected attempt changed nothing

    release_all.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert controller.global_active_count() == 0


# ============================================================================
# C. Release on success / provider exception / (simulated) timeout /
#    caller cancellation — and re-entry afterward.
# ============================================================================


async def test_slot_released_after_success_and_user_can_reenter():
    controller = _controller()
    user_id = uuid.uuid4()

    async with controller.acquire(user_id):
        pass
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0

    async with controller.acquire(user_id):
        pass  # succeeds again — proves the first release was genuine


async def test_slot_released_after_provider_exception_and_user_can_reenter():
    controller = _controller()
    user_id = uuid.uuid4()

    with pytest.raises(RuntimeError):
        async with controller.acquire(user_id):
            raise RuntimeError("simulated provider failure")

    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0

    async with controller.acquire(user_id):
        pass


async def test_slot_released_after_simulated_timeout_and_user_can_reenter():
    """The controller itself has no notion of "timeout" — it is just
    another exception raised inside the acquired block (see
    services.text_llm.TextGenerationTimeoutError / app.text_chat.
    TextChatTimeoutError for the real timeout classification, proven
    end-to-end in tests/test_stage7a1_timeout.py). This proves the
    controller's own release path is identical for that case."""
    controller = _controller()
    user_id = uuid.uuid4()

    class SimulatedTimeout(RuntimeError):
        pass

    with pytest.raises(SimulatedTimeout):
        async with controller.acquire(user_id):
            raise SimulatedTimeout("simulated generation timeout")

    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0

    async with controller.acquire(user_id):
        pass


async def test_slot_released_after_caller_cancellation_and_user_can_reenter():
    controller = _controller()
    user_id = uuid.uuid4()

    holding = asyncio.Event()
    never_release = asyncio.Event()  # deliberately never set

    async def holder():
        async with controller.acquire(user_id):
            holding.set()
            await never_release.wait()

    task = asyncio.create_task(holder())
    await holding.wait()
    assert controller.global_active_count() == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0

    async with controller.acquire(user_id):
        pass  # the cancelled holder's permit was genuinely released


# ============================================================================
# D. Registry cleanup: idle entries are removed; never grows unbounded; no
#    ABA race on release-then-immediate-reacquire.
# ============================================================================


async def test_idle_per_user_entries_are_removed():
    controller = _controller()
    user_id = uuid.uuid4()

    async with controller.acquire(user_id):
        assert controller.registry_size() == 1
    assert controller.registry_size() == 0


async def test_registry_does_not_grow_unbounded_across_many_distinct_users():
    controller = _controller(max_per_user=1, max_global=100)
    for _ in range(50):
        async with controller.acquire(uuid.uuid4()):
            pass
    assert controller.registry_size() == 0


async def test_no_aba_race_release_then_immediate_reacquire_same_user():
    """Admission is a plain counter, not a semaphore instance (see this
    module's own docstring) — there is no per-user *object* whose identity
    could go stale, so a release-then-reacquire sequence, repeated with no
    gap, can never observe a partially-cleaned-up state or spuriously
    report busy."""
    controller = _controller()
    user_id = uuid.uuid4()

    for _ in range(20):
        async with controller.acquire(user_id):
            assert controller.registry_size() == 1
        assert controller.registry_size() == 0


# ============================================================================
# E. Multi-event-loop safety.
# ============================================================================


def test_controller_survives_across_separate_event_loops():
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    user_id = uuid.uuid4()

    async def one_generation():
        async with controller.acquire(user_id):
            pass
        return controller.registry_size()

    # Two SEPARATE event loops, sequentially -- mirrors pytest-asyncio's
    # function-scoped event loop for two different test functions reusing
    # the same module-level singleton controller.
    loop1 = asyncio.new_event_loop()
    try:
        result_1 = loop1.run_until_complete(one_generation())
    finally:
        loop1.close()
    assert result_1 == 0

    loop2 = asyncio.new_event_loop()
    try:
        result_2 = loop2.run_until_complete(one_generation())
    finally:
        loop2.close()
    assert result_2 == 0


# ============================================================================
# F. Exception content: no UUID or user data.
# ============================================================================


async def test_busy_exception_message_contains_no_uuid_or_user_data():
    controller = _controller()
    user_id = uuid.uuid4()

    holding = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with controller.acquire(user_id):
            holding.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await holding.wait()

    with pytest.raises(GenerationBusyError) as excinfo:
        async with controller.acquire(user_id):
            pass

    assert str(user_id) not in str(excinfo.value)
    assert str(excinfo.value) == "Text generation is at capacity right now; please try again shortly"

    release.set()
    await asyncio.wait_for(task, timeout=5)


# ============================================================================
# G. Constructor validation (Stage 7A-1 corrective pass): max_per_user/
#    max_global must be genuine, positive int values -- bool (a subclass
#    of int in Python), zero, negative, float, infinity, string, and None
#    are all rejected.
# ============================================================================


_INVALID_LIMIT_VALUES = [True, False, 0, -1, -100, 1.5, float("inf"), float("nan"), "4", None, [], {}]


@pytest.mark.parametrize("bad_value", _INVALID_LIMIT_VALUES)
def test_constructor_rejects_invalid_max_per_user(bad_value):
    with pytest.raises(ValueError):
        GenerationAdmissionController(max_per_user=bad_value, max_global=4)


@pytest.mark.parametrize("bad_value", _INVALID_LIMIT_VALUES)
def test_constructor_rejects_invalid_max_global(bad_value):
    with pytest.raises(ValueError):
        GenerationAdmissionController(max_per_user=1, max_global=bad_value)


def test_constructor_accepts_genuine_positive_ints():
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


# ============================================================================
# H. Opaque permit ownership (Stage 7A-1 corrective pass): release()/
#    assert_permit_active() only ever mutate/inspect counters after
#    confirming the presented permit is a currently-active entry in THIS
#    controller's own registry -- a fabricated, foreign-controller, or
#    already-released permit never changes counters, in either direction.
# ============================================================================


def test_fabricated_permit_release_raises_and_does_not_change_counters():
    controller = _controller()
    fabricated = GenerationPermit()

    with pytest.raises(GenerationPermitError):
        controller.release(fabricated)

    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


def test_unknown_permit_assert_active_raises_without_changing_counters():
    controller = _controller()
    user_id = uuid.uuid4()
    fabricated = GenerationPermit()

    with pytest.raises(GenerationPermitError):
        controller.assert_permit_active(fabricated, user_id=user_id)

    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


def test_double_release_raises_and_does_not_change_counters_the_second_time():
    controller = _controller()
    user_id = uuid.uuid4()

    permit = controller.acquire_nowait(user_id)
    controller.release(permit)
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0

    with pytest.raises(GenerationPermitError):
        controller.release(permit)

    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


def test_cross_controller_release_raises_and_does_not_change_either_controllers_counters():
    controller_a = _controller()
    controller_b = _controller()
    user_id = uuid.uuid4()

    permit = controller_a.acquire_nowait(user_id)

    with pytest.raises(GenerationPermitError):
        controller_b.release(permit)

    assert controller_a.global_active_count() == 1  # still genuinely held by controller_a
    assert controller_b.global_active_count() == 0
    assert controller_b.registry_size() == 0

    controller_a.release(permit)
    assert controller_a.global_active_count() == 0


def test_cross_controller_assert_permit_active_raises():
    controller_a = _controller()
    controller_b = _controller()
    user_id = uuid.uuid4()

    permit = controller_a.acquire_nowait(user_id)

    with pytest.raises(GenerationPermitError):
        controller_b.assert_permit_active(permit, user_id=user_id)

    controller_a.release(permit)


def test_assert_permit_active_rejects_wrong_user():
    controller = _controller()
    user_id = uuid.uuid4()
    other_user = uuid.uuid4()

    permit = controller.acquire_nowait(user_id)

    with pytest.raises(GenerationPermitError):
        controller.assert_permit_active(permit, user_id=other_user)

    # Misuse never changed counters -- the permit is still genuinely
    # active for its real owner.
    controller.assert_permit_active(permit, user_id=user_id)
    controller.release(permit)


def test_assert_permit_active_rejects_released_permit():
    controller = _controller()
    user_id = uuid.uuid4()

    permit = controller.acquire_nowait(user_id)
    controller.release(permit)

    with pytest.raises(GenerationPermitError):
        controller.assert_permit_active(permit, user_id=user_id)


def test_forged_release_cannot_be_used_to_exceed_capacity():
    """A fabricated permit presented to release() must never free up
    capacity it never genuinely held -- proven by showing the cap is
    still enforced against a distinct user immediately afterward."""
    controller = _controller(max_per_user=1, max_global=1)
    user_id = uuid.uuid4()

    permit = controller.acquire_nowait(user_id)
    assert controller.global_active_count() == 1

    fabricated = GenerationPermit()
    with pytest.raises(GenerationPermitError):
        controller.release(fabricated)

    other_user = uuid.uuid4()
    with pytest.raises(GenerationBusyError):
        controller.acquire_nowait(other_user)  # cap still exhausted

    assert controller.global_active_count() == 1

    controller.release(permit)
    assert controller.global_active_count() == 0
    # Capacity is genuinely available again now.
    permit2 = controller.acquire_nowait(other_user)
    controller.release(permit2)


def test_manual_acquire_nowait_and_release_round_trip():
    controller = _controller()
    user_id = uuid.uuid4()

    permit = controller.acquire_nowait(user_id)
    assert isinstance(permit, GenerationPermit)
    assert controller.registry_size() == 1
    controller.assert_permit_active(permit, user_id=user_id)

    controller.release(permit)
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


# ============================================================================
# I. Genuine cross-OS-thread, cross-event-loop concurrency: proves the
#    plain threading.Lock guarding admission is race-free under REAL
#    parallel contention -- not merely sequential reuse across separate
#    event loops in a single thread (see Section E above for that, weaker,
#    proof).
# ============================================================================


def test_concurrent_multi_thread_multi_event_loop_same_user_admission_is_race_free():
    controller = GenerationAdmissionController(max_per_user=1, max_global=10)
    user_id = uuid.uuid4()
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    # Each worker records its own outcome under synchronization -- never
    # relying on an exception silently escaping a thread (pytest.ini
    # promotes PytestUnhandledThreadExceptionWarning to a hard error, so
    # any genuinely unexpected exception here would fail the suite loudly
    # rather than being swallowed).
    results = [None] * n_threads

    def worker(index):
        loop = asyncio.new_event_loop()
        try:
            async def attempt():
                # Blocking wait is intentional and safe here: this event
                # loop runs exactly one coroutine, so blocking the OS
                # thread on the barrier cannot starve any other task.
                barrier.wait(timeout=10)
                try:
                    permit = controller.acquire_nowait(user_id)
                    return ("ok", permit)
                except GenerationBusyError:
                    return ("busy", None)

            results[index] = loop.run_until_complete(attempt())
        except Exception as e:  # pragma: no cover -- captured, never swallowed
            results[index] = ("error", e)
        finally:
            loop.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    errors = [r for r in results if r is not None and r[0] == "error"]
    assert errors == [], f"worker thread(s) raised unexpectedly: {errors}"

    ok_results = [r for r in results if r[0] == "ok"]
    busy_results = [r for r in results if r[0] == "busy"]
    assert len(ok_results) == 1, f"expected exactly one winner under real cross-thread contention: {results}"
    assert len(busy_results) == n_threads - 1
    assert controller.registry_size() == 1
    assert controller.global_active_count() == 1

    _, winning_permit = ok_results[0]
    controller.release(winning_permit)
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


def test_concurrent_multi_thread_release_from_a_different_thread_than_acquired():
    """Release (not just acquire) must also be safe when invoked from a
    genuinely different OS thread than the one that acquired the permit
    -- the lock protects the shared registries themselves, not "whichever
    thread happens to call in"."""
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    user_id = uuid.uuid4()
    permit = controller.acquire_nowait(user_id)
    assert controller.global_active_count() == 1

    outcome = {}

    def release_from_other_thread():
        try:
            controller.release(permit)
            outcome["ok"] = True
        except Exception as e:  # pragma: no cover -- captured, never swallowed
            outcome["error"] = e

    t = threading.Thread(target=release_from_other_thread)
    t.start()
    t.join(timeout=10)

    assert outcome.get("ok") is True, outcome
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


# ============================================================================
# J. Pinned protected use (Stage 7A-1 SECOND corrective pass): closes the
#    validate-then-use TOCTOU a bare, point-in-time assert_permit_active()
#    left open -- begin_use()/end_use()/use_permit() pin a permit for the
#    duration of its protected provider work, and release() now REFUSES a
#    permit that is currently pinned. Proven with REAL threads forcing the
#    exact interleaving the independent audit reproduced: a permit begins
#    protected use, a concurrent actor tries to release it while pinned,
#    and -- because that release must be refused -- a new admission can
#    never occupy the capacity the still-in-flight permit is relying on.
# ============================================================================


def test_begin_use_validates_like_assert_permit_active():
    controller = _controller()
    user_id = uuid.uuid4()
    other_user = uuid.uuid4()
    fabricated = GenerationPermit()

    permit = controller.acquire_nowait(user_id)

    with pytest.raises(GenerationPermitError):
        controller.begin_use(fabricated, user_id=user_id)
    with pytest.raises(GenerationPermitError):
        controller.begin_use(permit, user_id=other_user)

    # Misuse above never actually pinned anything -- the permit still
    # releases cleanly.
    controller.release(permit)


def test_begin_use_rejects_permit_from_a_different_controller():
    controller_a = _controller()
    controller_b = _controller()
    user_id = uuid.uuid4()

    permit = controller_a.acquire_nowait(user_id)

    with pytest.raises(GenerationPermitError):
        controller_b.begin_use(permit, user_id=user_id)

    controller_a.release(permit)


def test_end_use_without_matching_begin_use_raises():
    controller = _controller()
    user_id = uuid.uuid4()
    permit = controller.acquire_nowait(user_id)

    with pytest.raises(GenerationPermitError):
        controller.end_use(permit)

    # No spurious pin was created by the failed end_use() -- release()
    # still succeeds normally.
    controller.release(permit)


def test_release_refuses_a_permit_that_is_currently_pinned():
    controller = _controller()
    user_id = uuid.uuid4()
    permit = controller.acquire_nowait(user_id)

    controller.begin_use(permit, user_id=user_id)
    with pytest.raises(GenerationPermitError):
        controller.release(permit)
    # The refused release changed nothing -- the permit is still genuinely
    # active and still counted.
    assert controller.global_active_count() == 1
    assert controller.registry_size() == 1

    controller.end_use(permit)
    controller.release(permit)  # succeeds now that the pin is gone
    assert controller.global_active_count() == 0


def test_use_permit_context_manager_pins_and_unpins_on_success():
    controller = _controller()
    user_id = uuid.uuid4()
    permit = controller.acquire_nowait(user_id)

    with controller.use_permit(permit, user_id=user_id):
        with pytest.raises(GenerationPermitError):
            controller.release(permit)

    # Pin is gone once the `with` block exits normally -- release() now
    # succeeds.
    controller.release(permit)
    assert controller.global_active_count() == 0


def test_use_permit_context_manager_unpins_on_exception():
    controller = _controller()
    user_id = uuid.uuid4()
    permit = controller.acquire_nowait(user_id)

    with pytest.raises(RuntimeError):
        with controller.use_permit(permit, user_id=user_id):
            raise RuntimeError("simulated protected-work failure")

    # Pin is gone even though the block exited via an exception.
    controller.release(permit)
    assert controller.global_active_count() == 0


async def test_use_permit_context_manager_unpins_on_cancellation():
    """use_permit() is a plain sync context manager, but it must still
    unpin correctly when wrapped around an `await` that is cancelled --
    proving it never holds threading.Lock across that await (a held lock
    would make no difference to cancellation here, but a broken
    try/finally inside the generator-based context manager would leak the
    pin)."""
    controller = _controller()
    user_id = uuid.uuid4()
    permit = controller.acquire_nowait(user_id)

    started = asyncio.Event()

    async def protected_work():
        with controller.use_permit(permit, user_id=user_id):
            started.set()
            never = asyncio.Event()
            await never.wait()

    task = asyncio.create_task(protected_work())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Pin is gone -- release() now succeeds.
    controller.release(permit)
    assert controller.global_active_count() == 0


def test_stale_permit_release_while_pinned_cannot_free_capacity_for_a_new_admission():
    """Deterministic single-threaded reproduction of the interleaving the
    independent audit described: a permit begins protected use (validated,
    then relied on); an attempt to release that SAME permit while it is
    still pinned must be refused; because it is refused, a distinct new
    user can never occupy the only global slot while the old permit's
    protected work is still in flight."""
    controller = _controller(max_per_user=1, max_global=1)
    old_user = uuid.uuid4()
    new_user = uuid.uuid4()

    old_permit = controller.acquire_nowait(old_user)
    controller.begin_use(old_permit, user_id=old_user)

    # "old permit is released" -- must be refused while pinned.
    with pytest.raises(GenerationPermitError):
        controller.release(old_permit)

    # "another request acquires the only global slot" -- must still be
    # impossible: capacity was never actually freed.
    with pytest.raises(GenerationBusyError):
        controller.acquire_nowait(new_user)

    # The old request's protected work finishes; only now can it be
    # legitimately released, and only then does capacity become available.
    controller.end_use(old_permit)
    controller.release(old_permit)

    new_permit = controller.acquire_nowait(new_user)
    controller.release(new_permit)


def test_concurrent_release_during_protected_use_cannot_free_capacity_for_a_new_admission():
    """The full two-REAL-thread race reproduction: while one thread holds
    a permit pinned for protected use, a SECOND, genuinely concurrent
    thread races to release that same permit. The race must always
    resolve to "refused" -- proven not by timing, but by the deterministic
    invariant checked immediately after: with max_global=1, a distinct new
    user's acquire_nowait() must still be rejected as busy, because the
    racing release() never actually freed the slot."""
    controller = GenerationAdmissionController(max_per_user=1, max_global=1)
    old_user = uuid.uuid4()
    new_user = uuid.uuid4()

    old_permit = controller.acquire_nowait(old_user)

    entered_use = threading.Event()
    finish_protected_work = threading.Event()
    protected_work_done = threading.Event()

    def old_worker():
        with controller.use_permit(old_permit, user_id=old_user):
            entered_use.set()
            finish_protected_work.wait(timeout=10)
        protected_work_done.set()

    release_attempted = threading.Event()
    release_outcome = {}

    def racing_release_worker():
        entered_use.wait(timeout=10)
        try:
            controller.release(old_permit)
            release_outcome["result"] = "released"
        except GenerationPermitError:
            release_outcome["result"] = "refused"
        finally:
            release_attempted.set()

    t_old = threading.Thread(target=old_worker)
    t_release = threading.Thread(target=racing_release_worker)
    t_old.start()
    t_release.start()

    assert release_attempted.wait(timeout=10)
    assert release_outcome.get("result") == "refused", (
        "a release racing against an in-progress protected use must always be refused -- "
        "never allowed to free capacity out from under still-active protected work"
    )

    # The defeated race changed nothing: capacity is still genuinely held,
    # so a distinct new user must still be rejected as busy.
    with pytest.raises(GenerationBusyError):
        controller.acquire_nowait(new_user)

    finish_protected_work.set()
    t_old.join(timeout=10)
    t_release.join(timeout=10)
    assert protected_work_done.is_set()

    # Only now, after protected use genuinely ended, can release succeed
    # and capacity become available again -- proving this was a real,
    # working lifecycle, not a permanently stuck permit.
    controller.release(old_permit)
    new_permit = controller.acquire_nowait(new_user)
    controller.release(new_permit)
    assert controller.global_active_count() == 0
