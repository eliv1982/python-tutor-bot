"""
Stage 6B independent-audit corrective pass #1, MAJOR 2 — unbounded public
persistent OAuth transaction storage. Every unauthenticated
`GET /api/auth/github/login` used to commit a new `github_oauth_transactions`
row with no cleanup, no cap, and no rate limit: an attacker could grow
PostgreSQL storage without bound merely by repeating the request.

db.oauth_transactions.create_sync() is now a single atomic transaction,
serialized across every process via `SELECT ... FOR UPDATE` on the
`github_oauth_admission` singleton row (db/models.py's GithubOAuthAdmission
— the same singleton-row-lock idiom db/auth_sessions.py already
established for `web_session_policy`), that deletes expired rows, resets/
advances a fixed GLOBAL rate window, and refuses to insert a new row at
all once either the rate window or the outstanding-row hard cap is
exhausted.

Every concurrency test below uses REAL threads against REAL PostgreSQL,
coordinated with `threading.Event` (never `time.sleep()` as the actual
synchronization mechanism) plus create_sync()'s own test-only
`_test_hook_after_lock` seam — the exact same idiom
tests/test_stage6a_corrective3_policy_race.py already established for
`web_session_policy`'s identical lock, adapted here to the OAuth admission
row.
"""

import hashlib
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select, text

import db.oauth_transactions as db_oauth_transactions
from db.engine import get_sync_engine
from db.models import GITHUB_OAUTH_ADMISSION_ID, GithubOAuthAdmission, GithubOAuthTransaction


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — every test here
    needs the REAL `github_oauth_transactions`/`github_oauth_admission`
    tables."""
    yield


def _hash(raw: str) -> bytes:
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _future_expiry(seconds: float = 600) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _past_expiry(seconds: float = 1) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def _create(
    *, expires_at=None, max_starts_per_window=10_000, max_outstanding=10_000, window_seconds=60,
    _test_hook_after_lock=None,
) -> bool:
    raw_state = secrets.token_urlsafe(32)
    return db_oauth_transactions.create_sync(
        state_hash=_hash(raw_state),
        code_verifier=secrets.token_urlsafe(48),
        expires_at=expires_at or _future_expiry(),
        max_starts_per_window=max_starts_per_window,
        max_outstanding=max_outstanding,
        window_seconds=window_seconds,
        _test_hook_after_lock=_test_hook_after_lock,
    )


def _outstanding_count() -> int:
    engine = get_sync_engine()
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(GithubOAuthTransaction)).scalar_one()


def _wait_until_blocked_on_admission_lock(*, timeout: float = 5.0) -> bool:
    """Mirrors tests/test_stage6a_corrective3_policy_race.py's
    `_wait_until_blocked_on_policy_lock()` exactly, scoped to the OAuth
    admission row's own table name instead."""
    engine = get_sync_engine()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND query ILIKE '%github_oauth_admission%'"
                )
            ).scalar_one()
        if count > 0:
            return True
        time.sleep(0.02)
    return False


# --- outstanding-row hard cap -------------------------------------------------


def test_rows_below_cap_are_admitted(postgres_db):
    for _ in range(5):
        assert _create(max_outstanding=5) is True
    assert _outstanding_count() == 5


def test_reaching_cap_rejects_the_next_create(postgres_db):
    for _ in range(3):
        assert _create(max_outstanding=3) is True
    assert _create(max_outstanding=3) is False
    assert _outstanding_count() == 3


def test_rejected_create_inserts_nothing(postgres_db):
    assert _create(max_outstanding=1) is True
    assert _outstanding_count() == 1
    assert _create(max_outstanding=1) is False
    # Still exactly one row — the rejected attempt inserted nothing.
    assert _outstanding_count() == 1


def test_expired_rows_are_deleted_and_free_capacity(postgres_db):
    assert _create(expires_at=_past_expiry(), max_outstanding=1) is True
    assert _outstanding_count() == 1

    # A second create at the same cap: cleanup (part of THIS call) removes
    # the expired row first, freeing capacity for the new one.
    assert _create(max_outstanding=1) is True
    assert _outstanding_count() == 1


# --- global rate window -------------------------------------------------------


def test_rate_threshold_rejects_without_inserting(postgres_db):
    for _ in range(4):
        assert _create(max_starts_per_window=4, window_seconds=60) is True
    assert _outstanding_count() == 4

    assert _create(max_starts_per_window=4, window_seconds=60) is False
    # No new row from the rejected attempt.
    assert _outstanding_count() == 4


def test_rate_window_reset_admits_again_after_elapsing(postgres_db):
    assert _create(max_starts_per_window=1, window_seconds=1) is True
    assert _create(max_starts_per_window=1, window_seconds=1) is False

    time.sleep(1.2)

    assert _create(max_starts_per_window=1, window_seconds=1) is True


def test_rate_limited_rejection_does_not_consume_outstanding_cap(postgres_db):
    """A rate-limited rejection must not insert a row — proven directly by
    confirming the outstanding count never grows past what was actually
    admitted, even across several rejected attempts."""
    assert _create(max_starts_per_window=1, window_seconds=60) is True
    for _ in range(5):
        assert _create(max_starts_per_window=1, window_seconds=60) is False
    assert _outstanding_count() == 1


# --- storage-growth regression proof (Section 11) ----------------------------


def test_repeated_creation_attempts_never_exceed_the_configured_hard_cap(postgres_db):
    """Issues more than 2x the configured hard cap worth of create
    attempts and proves the persisted row count never exceeds it, while
    every rejection inserted nothing."""
    cap = 10
    admitted = 0
    rejected = 0
    for _ in range(cap * 3):
        if _create(max_starts_per_window=10_000, max_outstanding=cap):
            admitted += 1
        else:
            rejected += 1
        assert _outstanding_count() <= cap

    assert admitted == cap
    assert rejected == cap * 3 - cap
    assert _outstanding_count() == cap

    # Expire everything, then confirm a fresh admission is possible again
    # once cleanup reclaims the space — storage is bounded, not "stuck".
    engine = get_sync_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE github_oauth_transactions SET expires_at = now() - interval '1 hour'"))
    assert _create(max_starts_per_window=10_000, max_outstanding=cap) is True
    assert _outstanding_count() == 1


# --- concurrency: hard cap can never be overshot ------------------------------


def test_concurrent_creates_near_cap_cannot_overshoot_it(postgres_db):
    """Real-thread proof (Section 10): many concurrent create_sync() calls
    racing against a small hard cap must never leave more rows than the
    cap allows, regardless of how many threads "won" the race to try."""
    cap = 5
    thread_count = 25
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(thread_count)

    def _attempt():
        barrier.wait(timeout=5)
        outcome = _create(max_starts_per_window=10_000, max_outstanding=cap)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=_attempt) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert all(not t.is_alive() for t in threads)
    admitted = sum(1 for r in results if r is True)
    assert admitted == cap
    assert _outstanding_count() == cap


def test_concurrent_creates_near_rate_limit_cannot_overshoot_it(postgres_db):
    limit = 5
    thread_count = 25
    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(thread_count)

    def _attempt():
        barrier.wait(timeout=5)
        outcome = _create(max_starts_per_window=limit, window_seconds=60, max_outstanding=10_000)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=_attempt) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert all(not t.is_alive() for t in threads)
    admitted = sum(1 for r in results if r is True)
    assert admitted == limit
    assert _outstanding_count() == limit


# --- deterministic real-PostgreSQL row-lock interleaving proof --------------


def test_admission_row_lock_genuinely_serializes_two_concurrent_creators(postgres_db):
    """Deterministic (no sleep-based race) proof that the admission
    singleton row lock actually serializes two concurrent create_sync()
    calls, mirroring tests/test_stage6a_corrective3_policy_race.py's own
    `_test_hook_after_lock` + pg_stat_activity polling technique exactly.
    The first caller is paused holding the lock; the second is confirmed
    GENUINELY BLOCKED on it (via pg_stat_activity) before the first is
    released — proving real row-lock contention, not merely "happened to
    run after"."""
    first_holds_lock = threading.Event()
    release_first = threading.Event()

    def _pause_first():
        first_holds_lock.set()
        assert release_first.wait(timeout=5), "test never released the first caller"

    first_outcome = {}

    def _run_first():
        first_outcome["admitted"] = _create(max_outstanding=10, _test_hook_after_lock=_pause_first)

    first_thread = threading.Thread(target=_run_first)
    first_thread.start()
    assert first_holds_lock.wait(timeout=5), "first caller never reached the admission row lock"

    second_outcome = {}

    def _run_second():
        second_outcome["admitted"] = _create(max_outstanding=10)

    second_thread = threading.Thread(target=_run_second)
    second_thread.start()

    assert _wait_until_blocked_on_admission_lock(), (
        "second caller never showed up as genuinely blocked on the admission row lock — "
        "test setup failed to actually force the contested ordering"
    )

    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive() and not second_thread.is_alive()

    assert first_outcome["admitted"] is True
    assert second_outcome["admitted"] is True
    assert _outstanding_count() == 2


# --- rollback consistency on rejection ---------------------------------------


def test_rollback_on_rate_rejection_leaves_admission_row_and_lock_usable(postgres_db):
    """A rejection must roll back cleanly: the admission row's own state
    stays coherent and — critically — the row lock is actually released
    (proven by a subsequent lock-requiring call succeeding immediately,
    no hang)."""
    assert _create(max_starts_per_window=1, window_seconds=60) is True
    assert _create(max_starts_per_window=1, window_seconds=60) is False

    # The lock was genuinely released — this would hang/timeout otherwise.
    assert _create(max_starts_per_window=10_000, window_seconds=60) is True


def test_rollback_on_cap_rejection_leaves_no_orphan_row(postgres_db):
    assert _create(max_outstanding=1) is True
    assert _create(max_outstanding=1) is False
    assert _outstanding_count() == 1


# --- window bookkeeping is actually persisted ---------------------------------


def test_starts_in_window_is_persisted_and_increments(postgres_db):
    engine = get_sync_engine()

    def _read_admission():
        with engine.connect() as conn:
            return conn.execute(
                select(GithubOAuthAdmission.starts_in_window).where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
            ).scalar_one()

    before = _read_admission()
    assert _create(max_starts_per_window=10_000, window_seconds=60) is True
    after = _read_admission()
    assert after == before + 1


def test_singleton_admission_row_exists_exactly_once(postgres_db):
    engine = get_sync_engine()
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(GithubOAuthAdmission)).scalar_one()
    assert count == 1
