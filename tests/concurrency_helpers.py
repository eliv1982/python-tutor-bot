"""
Shared helper for Stage 6C real-thread, real-PostgreSQL concurrency
regression tests (Stage 6C corrective pass, independent-audit MAJOR 1/2) —
every tests/test_stage6c_*.py module that starts real threads against a
real database routes its worker bookkeeping through this module, rather
than each test re-inventing its own unsynchronized dict/list.

Two shapes are covered:

  - `run_workers()`: N truly-concurrent workers racing the SAME operation
    (e.g. six near-simultaneous POST /api/link/telegram/start calls, or
    twelve first-ever create_attempt_sync() callers) — released together
    via a shared threading.Barrier so they begin at genuinely the same
    moment, with each worker's own outcome (return value OR raised
    exception, never both) recorded exactly once under a lock, keyed by
    worker id.
  - `capture()`: a single staged actor inside a deliberate
    pause/release Event handoff (the "pause after acquiring lock X,
    confirm the other side is genuinely blocked on it, then release"
    pattern already used throughout tests/test_stage6c_lock_ordering.py
    and friends) — wraps that one thread's body so a raised exception
    becomes part of its own WorkerRecord instead of only being caught by
    pytest's PytestUnhandledThreadExceptionWarning-as-error promotion
    (pytest.ini), which fires but gives no clean, worker-keyed assertion
    surface of its own.

Every caller is expected to follow up with an explicit liveness check
(`assert not thread.is_alive()`) after `join(timeout=...)` — this module
starts/joins threads for `run_workers()` but leaves that assertion, and
every domain-specific assertion (exact outcome multiset, final database
state), to the caller, matching this suite's existing style.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from sqlalchemy import text

from db.engine import get_sync_engine


@dataclass
class WorkerRecord:
    """Exactly one of `result`/`exception` is meaningful for a COMPLETED
    worker — `exception is None` is this module's own uniform "did this
    worker succeed" check, never a bare truthiness check on `result`
    (which may legitimately be None/False/an empty value for some
    callers)."""

    result: Any = None
    exception: Optional[BaseException] = None


def capture(fn: Callable[[], Any]) -> WorkerRecord:
    """Runs `fn()` and returns a WorkerRecord carrying either its return
    value or the exception it raised — never lets an exception propagate
    out of a thread target uncaught. Intended for the single-actor staged
    (pause/release Event) tests: `outcome["record"] = capture(lambda: ...)`
    inside the thread target, then `outcome["record"].exception`/`.result`
    asserted on after the thread is joined."""
    record = WorkerRecord()
    try:
        record.result = fn()
    except BaseException as e:  # pragma: no cover - surfaced via the record
        record.exception = e
    return record


def run_workers(
    worker_ids: Sequence[Hashable],
    body: Callable[[Hashable], Any],
    *,
    timeout: float = 15.0,
    use_barrier: bool = True,
) -> Tuple[Dict[Hashable, WorkerRecord], Dict[Hashable, threading.Thread]]:
    """
    Starts one thread per id in `worker_ids`, each running `body(worker_id)`.
    When `use_barrier` (the default), every thread waits on a shared
    threading.Barrier sized to `len(worker_ids)` immediately before calling
    `body` — every worker begins the operation under test at genuinely the
    same moment, rather than in launch order (a `Barrier` timeout, or one
    worker's thread failing to even start, breaks the barrier for every
    other waiter, which becomes that worker's own captured
    BrokenBarrierError — never a silent hang).

    Each worker's own outcome — its return value OR its raised exception,
    NEVER both — is recorded exactly once, under a shared lock, keyed by
    worker id, as a WorkerRecord (see that class's own docstring). Every
    thread is then joined with `timeout`.

    Returns `(records, threads)`, both keyed by worker id — callers are
    expected to assert, explicitly, in this order: every thread's
    `.is_alive()` is False; `set(records) == set(worker_ids)`; every
    record's `.exception is None`; the exact expected result multiset;
    finally the real database's post-condition state.
    """
    records: Dict[Hashable, WorkerRecord] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(len(worker_ids)) if use_barrier and len(worker_ids) > 1 else None

    def _run(worker_id: Hashable) -> None:
        def _body_after_barrier():
            if barrier is not None:
                barrier.wait(timeout=timeout)
            return body(worker_id)

        record = capture(_body_after_barrier)
        with lock:
            records[worker_id] = record

    threads: Dict[Hashable, threading.Thread] = {
        worker_id: threading.Thread(target=_run, args=(worker_id,)) for worker_id in worker_ids
    }
    for t in threads.values():
        t.start()
    for t in threads.values():
        t.join(timeout=timeout)

    return records, threads


def assert_all_terminated(threads: Dict[Hashable, threading.Thread]) -> None:
    """Fixed, worker-keyed liveness assertion — fails with the specific
    ids still alive (never secret-bearing, always just worker ids) rather
    than a bare `assert not t.is_alive()` repeated per-thread."""
    still_alive = [worker_id for worker_id, t in threads.items() if t.is_alive()]
    assert still_alive == [], f"worker(s) did not complete within the bound: {still_alive!r}"


def assert_no_exceptions(records: Dict[Hashable, WorkerRecord]) -> None:
    failed = {worker_id: record.exception for worker_id, record in records.items() if record.exception is not None}
    assert failed == {}, f"unexpected exception(s) escaped worker(s): {failed!r}"


def wait_until_blocked_on(*, table_substring: str, timeout: float = 5.0) -> bool:
    """Polls pg_stat_activity for a REAL, genuinely-waiting backend whose
    current query mentions `table_substring`, via a short-lived connection
    separate from every contending side — the same pattern
    tests/test_stage6a_corrective3_policy_race.py's own
    `_wait_until_blocked_on_policy_lock()` established, generalized here so
    every Stage 6C module that needs it shares one implementation instead
    of re-declaring it per file."""
    engine = get_sync_engine()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND query ILIKE :pattern"
                ),
                {"pattern": f"%{table_substring}%"},
            ).scalar_one()
        if count > 0:
            return True
        time.sleep(0.02)
    return False


def wait_until_blocked_on_advisory_lock(*, timeout: float = 5.0) -> bool:
    """Same idea as `wait_until_blocked_on()`, for the
    `pg_advisory_xact_lock` case specifically (an advisory lock wait's
    `query` text names the function call itself, never a table)."""
    engine = get_sync_engine()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND query ILIKE '%pg_advisory_xact_lock%'"
                )
            ).scalar_one()
        if count > 0:
            return True
        time.sleep(0.02)
    return False
