"""
Stage 6A independent-audit corrective pass #3 — the MAJOR finding
remaining after pass #2: an opposite-posture OLD PROCESS could still
create a web session after a NEWER process's one-time startup revocation
had already run, and that late session would never be swept up by
anyone. Real reproduction (independent auditor):

  1. insecure application B completes startup (its revocation commits);
  2. old secure application A is still running;
  3. A creates a NEW secure session — nothing stops it, because B's
     revocation already ran and nothing re-checks it;
  4. B rejects the bearer only because ITS OWN posture differs (a filter,
     not a kill);
  5. posture later returns to secure;
  6. the late-issued secure bearer authenticates successfully — revived.

Fix (see db/models.py's WebSessionPolicy docstring for the full
narrative): `web_session_policy`, a singleton, PostgreSQL-authoritative
row, is now the ONE shared serialization point between session creation
(db.auth_sessions.create_sync()) and posture transition
(db.auth_sessions.apply_startup_posture_sync()) — both acquire it via
`SELECT ... FOR UPDATE` as the FIRST statement of their transaction and
hold that lock through their own commit. Real PostgreSQL row-level mutual
exclusion, not a process-local lock, not an advisory comment, not timing.

Every concurrency test below uses REAL threads against REAL PostgreSQL,
coordinated with `threading.Event` (never `time.sleep()` as the actual
synchronization mechanism) plus `db.auth_sessions.create_sync()`'s/
`apply_startup_posture_sync()`'s test-only `_test_hook_after_lock` seam
(mirrors the exact threading.Event pattern
tests/test_stage1e1_cancellation_safety.py already established for
worker-thread synchronization elsewhere in this codebase, adapted to a DB
transaction instead of a worker thread) to force one side to hold its
transaction open on command. Genuine row-lock CONTENTION (not merely
"happened to run after") is confirmed by polling PostgreSQL's own
`pg_locks` system view for a real, ungranted lock — a real, pollable fact,
not a timing guess (the same "poll a real condition" idiom
tests/conftest.py's own `_wait_for_real_postgres_connection()` already
uses for container readiness).
"""

import hashlib
import random
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, insert, select, text
from sqlalchemy.exc import IntegrityError

import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
from db.engine import get_sync_engine
from db.models import WEB_SESSION_POLICY_ID, WebSession, WebSessionPolicy


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — every test here
    needs REAL `users` rows and a REAL `web_session_policy` singleton."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _hash(raw_token: str) -> bytes:
    return hashlib.sha256(raw_token.encode("utf-8")).digest()


def _future_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _wait_until_blocked_on_policy_lock(*, timeout: float = 5.0) -> bool:
    """Polls PostgreSQL's own `pg_stat_activity` until at least one other
    backend is shown genuinely waiting on a lock (`wait_event_type =
    'Lock'`) while executing a query against `web_session_policy` —
    deterministic, real confirmation of row-lock contention. (A `SELECT
    ... FOR UPDATE` row-lock wait shows up as the SECOND transaction
    blocking on the FIRST transaction's `transactionid`, not as an
    ungranted `pg_locks` row scoped to the table's OID — `pg_stat_activity`
    plus `wait_event_type` is the reliable, documented way to observe
    this, regardless of which underlying lock TYPE the wait resolves to.)
    Never the actual synchronization primitive itself (that is always a
    threading.Event) — only used to PROVE contention genuinely happened
    before a test proceeds to release the side holding the lock. Uses its
    own short-lived connection each poll, separate from either side's own
    session, so it never itself contends for the lock."""
    engine = get_sync_engine()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND query ILIKE '%web_session_policy%'"
                )
            ).scalar_one()
        if count > 0:
            return True
        time.sleep(0.02)
    return False


# --- Test 1: creation wins the lock race ------------------------------------


def test_ordering_a_creation_wins_the_lock_race_transition_still_revokes_it(postgres_db):
    """Required Test 1. The creator (create_sync, issued_secure=True,
    matching the fixture's default secure DB posture) is deterministically
    made to acquire the policy row lock FIRST and hold its transaction
    open. Only once a concurrently-started transition
    (apply_startup_posture_sync(requested_secure=False)) is confirmed
    GENUINELY BLOCKED on that same lock (via pg_locks) does the test
    release the creator. Final assertion: the transition still revokes the
    session that committed just before it, and the bearer stays dead even
    after posture is later restored to secure."""
    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash(raw_token)
    expires_at = _future_expiry()

    creator_holds_lock = threading.Event()
    release_creator = threading.Event()

    def _pause_creator():
        creator_holds_lock.set()
        assert release_creator.wait(timeout=5), "test never released the creator"

    creator_outcome = {}

    def _run_creator():
        try:
            db_auth_sessions.create_sync(
                token_hash=token_hash, user_id=user_id, issued_secure=True,
                expires_at=expires_at, _test_hook_after_lock=_pause_creator,
            )
        except Exception as e:  # pragma: no cover - failure path asserted below
            creator_outcome["error"] = e

    creator_thread = threading.Thread(target=_run_creator)
    creator_thread.start()
    assert creator_holds_lock.wait(timeout=5), "creator never reached the policy row lock"

    transition_outcome = {}

    def _run_transition():
        transition_outcome["revoked"] = db_auth_sessions.apply_startup_posture_sync(requested_secure=False)

    transition_thread = threading.Thread(target=_run_transition)
    transition_thread.start()

    assert _wait_until_blocked_on_policy_lock(), (
        "transition never showed up as genuinely blocked on the policy row lock — "
        "test setup failed to actually force the contested ordering"
    )

    release_creator.set()
    creator_thread.join(timeout=5)
    transition_thread.join(timeout=5)
    assert not creator_thread.is_alive() and not transition_thread.is_alive()

    assert "error" not in creator_outcome, creator_outcome.get("error")
    assert transition_outcome["revoked"] >= 1

    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None

    # Posture restored to secure — the late-created bearer must STILL be
    # dead (the actual regression under test).
    db_auth_sessions.apply_startup_posture_sync(requested_secure=True)
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None


# --- Test 2: transition wins the lock race ----------------------------------


def test_ordering_b_transition_wins_the_lock_race_stale_creation_fails_closed(postgres_db):
    """Required Test 2. The transition
    (apply_startup_posture_sync(requested_secure=False)) is deterministically
    made to acquire the policy row lock FIRST and hold its transaction
    open. A concurrently-started, stale-posture create_sync()
    (issued_secure=True — still believing the old, secure posture) is
    confirmed GENUINELY BLOCKED on that same lock before the test releases
    the transition. Final assertion: creation fails closed with
    StalePostureError and no row is ever inserted."""
    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash(raw_token)
    expires_at = _future_expiry()

    transition_holds_lock = threading.Event()
    release_transition = threading.Event()

    def _pause_transition():
        transition_holds_lock.set()
        assert release_transition.wait(timeout=5), "test never released the transition"

    def _run_transition():
        db_auth_sessions.apply_startup_posture_sync(
            requested_secure=False, _test_hook_after_lock=_pause_transition
        )

    transition_thread = threading.Thread(target=_run_transition)
    transition_thread.start()
    assert transition_holds_lock.wait(timeout=5), "transition never reached the policy row lock"

    creator_outcome = {}

    def _run_creator():
        try:
            db_auth_sessions.create_sync(
                token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at
            )
        except Exception as e:
            creator_outcome["error"] = e

    creator_thread = threading.Thread(target=_run_creator)
    creator_thread.start()

    assert _wait_until_blocked_on_policy_lock(), (
        "creation never showed up as genuinely blocked on the policy row lock — "
        "test setup failed to actually force the contested ordering"
    )

    release_transition.set()
    transition_thread.join(timeout=5)
    creator_thread.join(timeout=5)
    assert not creator_thread.is_alive() and not transition_thread.is_alive()

    assert isinstance(creator_outcome.get("error"), db_auth_sessions.StalePostureError), creator_outcome

    # No row was ever inserted for this token.
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False) is None


# --- Test 3: same-posture concurrency ---------------------------------------


def test_same_posture_concurrent_creations_both_succeed(postgres_db):
    """Required Test 3. Two ordinary, non-contrived concurrent creations
    under the SAME (already-authoritative) posture must both succeed —
    the policy lock briefly serializes them but never rejects either."""
    user_id = _real_user()
    expires_at = _future_expiry()
    token_a, token_b = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    hash_a, hash_b = _hash(token_a), _hash(token_b)

    errors = []

    def _create(token_hash):
        try:
            db_auth_sessions.create_sync(
                token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at
            )
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=_create, args=(hash_a,))
    t2 = threading.Thread(target=_create, args=(hash_b,))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert errors == []
    assert db_auth_sessions.get_active_sync(token_hash=hash_a, expected_secure=True) is not None
    assert db_auth_sessions.get_active_sync(token_hash=hash_b, expected_secure=True) is not None


# --- Test 4: same-posture multi-instance startup ----------------------------


def test_same_posture_concurrent_startups_do_not_revoke_compatible_sessions(postgres_db):
    """Required Test 4. Two concurrent "startups" (apply_startup_posture_sync)
    both requesting the SAME, already-current posture must not revoke a
    genuinely compatible active session — a same-posture rolling deploy
    (two new instances starting up together) must be a safe no-op."""
    user_id = _real_user()
    expires_at = _future_expiry()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash(raw_token)
    db_auth_sessions.create_sync(
        token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at
    )

    results = []

    def _startup():
        results.append(db_auth_sessions.apply_startup_posture_sync(requested_secure=True))

    t1 = threading.Thread(target=_startup)
    t2 = threading.Thread(target=_startup)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert results == [0, 0]
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is not None


# --- Tests 5 & 6: sequential regression, direct sync-layer proof -----------


def test_sequential_regression_secure_insecure_secure_via_sync_layer(postgres_db):
    """Required Test 5. Complements
    tests/test_stage6a_corrective2_posture_transition.py's Scenario A
    (full HTTP/FastAPI path) with a fast, direct proof through the exact
    sync functions this pass rewrote."""
    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash(raw_token)
    expires_at = _future_expiry()

    db_auth_sessions.create_sync(token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at)
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is not None

    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None

    db_auth_sessions.apply_startup_posture_sync(requested_secure=True)
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None, (
        "the original secure session must not revive merely because posture returned"
    )


def test_sequential_regression_insecure_secure_insecure_via_sync_layer(postgres_db):
    """Required Test 6 (symmetric)."""
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash(raw_token)
    expires_at = _future_expiry()

    db_auth_sessions.create_sync(token_hash=token_hash, user_id=user_id, issued_secure=False, expires_at=expires_at)
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False) is not None

    db_auth_sessions.apply_startup_posture_sync(requested_secure=True)
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False) is None

    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False) is None, (
        "the original insecure session must not revive merely because posture returned"
    )


# --- Database policy tests --------------------------------------------------


def test_singleton_policy_row_exists_exactly_once(postgres_db):
    engine = get_sync_engine()
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(WebSessionPolicy)).scalar_one()
    assert count == 1


def test_singleton_check_constraint_rejects_a_second_row(postgres_db):
    """If a constraint protects singleton identity, verify it."""
    engine = get_sync_engine()
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(insert(WebSessionPolicy).values(id=2, current_secure=True))


def test_singleton_primary_key_rejects_a_duplicate_id_row(postgres_db):
    engine = get_sync_engine()
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(insert(WebSessionPolicy).values(id=WEB_SESSION_POLICY_ID, current_secure=False))


def test_current_posture_is_persisted_across_reads(postgres_db):
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    engine = get_sync_engine()
    with engine.connect() as conn:
        value = conn.execute(
            select(WebSessionPolicy.current_secure).where(WebSessionPolicy.id == WEB_SESSION_POLICY_ID)
        ).scalar_one()
    assert value is False


def test_transition_persists_posture_and_revokes_incompatible_but_not_compatible(postgres_db):
    user_id = _real_user()
    expires_at = _future_expiry()
    compatible_hash = _hash(secrets.token_urlsafe(32))
    incompatible_hash = _hash(secrets.token_urlsafe(32))

    # DB starts secure (fixture default) — create a session that WILL be
    # compatible with the posture we transition TO below.
    db_auth_sessions.create_sync(
        token_hash=compatible_hash, user_id=user_id, issued_secure=True, expires_at=expires_at
    )
    # Stand in for an already-incompatible row (insecure) — a raw insert,
    # since create_sync() itself would refuse to create this under the
    # current (secure) policy; see
    # tests/test_stage6a_corrective2_posture_transition.py's own
    # apply_startup_posture tests for the same technique.
    engine = get_sync_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(WebSession).values(
                session_token_hash=incompatible_hash, user_id=user_id, issued_secure=False, expires_at=expires_at
            )
        )

    revoked = db_auth_sessions.apply_startup_posture_sync(requested_secure=True)

    assert revoked == 1
    with engine.connect() as conn:
        persisted = conn.execute(
            select(WebSessionPolicy.current_secure).where(WebSessionPolicy.id == WEB_SESSION_POLICY_ID)
        ).scalar_one()
    assert persisted is True
    assert db_auth_sessions.get_active_sync(token_hash=compatible_hash, expected_secure=True) is not None
    assert db_auth_sessions.get_active_sync(token_hash=incompatible_hash, expected_secure=False) is None


def test_create_sync_rejects_stale_posture(postgres_db):
    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash(raw_token)

    # DB policy defaults to secure=True (fixture) — requesting
    # issued_secure=False is therefore stale.
    with pytest.raises(db_auth_sessions.StalePostureError):
        db_auth_sessions.create_sync(
            token_hash=token_hash, user_id=user_id, issued_secure=False, expires_at=_future_expiry()
        )
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False) is None
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None


def test_create_sync_succeeds_when_posture_matches(postgres_db):
    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash(raw_token)

    db_auth_sessions.create_sync(
        token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=_future_expiry()
    )
    record = db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True)
    assert record is not None
    assert record.user_id == user_id


def test_rollback_on_insert_failure_leaves_policy_unchanged_and_no_orphan_session(postgres_db):
    """Rollback consistency: an INSERT that fails AFTER the policy lock/
    check already passed (here, a digest-length CHECK-constraint
    violation on purpose) must leave no orphan session row, must not
    disturb the policy row, and — critically — must actually release the
    row lock (proven by successfully performing another lock-requiring
    operation immediately afterward, with no hang)."""
    user_id = _real_user()

    with pytest.raises(IntegrityError):
        db_auth_sessions.create_sync(
            token_hash=b"too-short",  # violates ck_web_sessions_session_token_hash_length
            user_id=user_id, issued_secure=True, expires_at=_future_expiry(),
        )

    engine = get_sync_engine()
    with engine.connect() as conn:
        policy_value = conn.execute(
            select(WebSessionPolicy.current_secure).where(WebSessionPolicy.id == WEB_SESSION_POLICY_ID)
        ).scalar_one()
        session_count = conn.execute(
            select(func.count()).select_from(WebSession).where(WebSession.user_id == user_id)
        ).scalar_one()
    assert policy_value is True  # unchanged fixture default
    assert session_count == 0

    # Lock genuinely released — this would hang/timeout otherwise (no
    # timeout is applied here on purpose: a stuck lock must fail the test
    # run visibly rather than silently pass via a generous timeout).
    assert db_auth_sessions.apply_startup_posture_sync(requested_secure=True) == 0
