"""
Server-side web-session persistence (Stage 6A) — SYNC, deliberately (see
db/engine.py's module docstring for why: psycopg async mode is
incompatible with Windows' default ProactorEventLoop). app/auth_session.py
wraps every function here in asyncio.to_thread(), the same idiom
db/identity.py and db/preferences.py already use for this codebase's other
blocking-I/O boundaries.

Only a SHA-256 digest of the browser's bearer session token
(`session_token_hash`) is ever persisted here — never the raw token
itself; see db/models.py's WebSession docstring. A plain, fast digest is
enough (no bcrypt/argon2/salt): the value being hashed is a 256-bit
secrets.token_urlsafe() output, not a user-chosen low-entropy password —
brute-forcing it back from its hash is already computationally infeasible,
so a slow/salted KDF would add cost without adding real protection here,
while a plain digest keeps every lookup a single indexed equality query
(get_active_sync() below).

Stage 6A independent-audit corrective pass #3: create_sync() and
apply_startup_posture_sync() are the ONE shared transactional protocol
guaranteeing the database (never any one process's memory) is
authoritative for the current web-session cookie posture — see
db/models.py's WebSessionPolicy docstring for the full rationale of the
race this closes and why a `SELECT ... FOR UPDATE` on that singleton row
is the serialization point both functions share. Read both docstrings
before touching either function: correctness here depends entirely on the
row lock being acquired FIRST and held through the final commit, in both.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import WEB_SESSION_POLICY_ID, WebSession, WebSessionPolicy


class StalePostureError(Exception):
    """Raised by create_sync() when the caller's requested `issued_secure`
    no longer matches the authoritative posture persisted in
    `web_session_policy` (Stage 6A independent-audit corrective pass #3) —
    e.g. an old process still running under a cookie posture a newer
    process has since transitioned away from (see
    apply_startup_posture_sync()'s own docstring for the exact race this
    closes). A plain, clean domain signal — never a raw SQLAlchemy/DB
    exception — safe to surface all the way up through
    app/auth_session.py (which re-exports this exact class) to a future
    Stage 6B caller without leaking persistence internals. No session row
    is inserted and the transaction is rolled back before this is raised."""


@dataclass(frozen=True)
class SessionRecord:
    """Minimal, concrete snapshot of one currently-active `web_sessions`
    row — see get_active_sync()'s own "fail-closed lookup" contract."""
    user_id: uuid.UUID
    created_at: datetime
    expires_at: datetime


def create_sync(
    *,
    token_hash: bytes,
    user_id: uuid.UUID,
    issued_secure: bool,
    expires_at: datetime,
    _test_hook_after_lock: Optional[Callable[[], None]] = None,
) -> None:
    """
    Insert a brand-new session row — but ONLY inside the SAME transaction
    as a fresh, LOCKED read of the authoritative posture in
    `web_session_policy` (Stage 6A independent-audit corrective pass #3;
    see db/models.py's WebSessionPolicy docstring for the full protocol
    and the exact cross-process race this closes).

    `SELECT ... FOR UPDATE` on the singleton policy row is the FIRST
    statement this function executes. The row lock it takes is held for
    the rest of THIS function's transaction — through the posture
    comparison, the INSERT (or the early rollback), and the final commit
    — because everything below runs on the same `Session`/connection, and
    nothing here ever commits or opens a second transaction before the
    real work is done. This is what makes "check posture, then insert" a
    single atomic unit against a concurrent apply_startup_posture_sync()
    call, which takes the exact same lock, the exact same way, before
    doing anything else: whichever of the two reaches the lock first runs
    to completion (commit) before the other can even read the row.

    `expires_at` is computed by the caller (app/auth_session.py, from
    session_config.SESSION_TTL_SECONDS) — this function performs no
    expiry-policy decisions of its own. `issued_secure` (Stage 6A
    independent-audit corrective pass #2, Major 1) is what gets persisted
    onto the new row (see db/models.py's WebSession docstring) — the
    SAME value is also what gets compared against the locked policy row
    here in pass #3.

    Raises StalePostureError (no row inserted, transaction rolled back) if
    `issued_secure` no longer matches the authoritative posture — this is
    not a bug in the caller; it means the calling process's own
    web_config.COOKIE_SECURE is stale relative to the database.

    `_test_hook_after_lock`: test-only synchronization seam, never passed
    by any real (non-test) caller — see tests/test_stage6a_corrective3_
    policy_race.py's own docstring for why this exists (deterministic,
    non-sleep-based proof of the real PostgreSQL row-lock interleaving,
    mirroring the threading.Event pattern tests/test_stage1e1_
    cancellation_safety.py already established for worker-thread
    synchronization elsewhere in this codebase). If provided, it is
    called with no arguments immediately after the policy row lock is
    acquired, before the posture comparison — a test can use it to pause
    this transaction open (via a blocking threading.Event.wait()) for as
    long as needed to deterministically force a concurrent caller to
    contend on the same lock.
    """
    with Session(get_sync_engine()) as session:
        current_secure = session.execute(
            select(WebSessionPolicy.current_secure)
            .where(WebSessionPolicy.id == WEB_SESSION_POLICY_ID)
            .with_for_update()
        ).scalar_one()

        if _test_hook_after_lock is not None:
            _test_hook_after_lock()

        if current_secure != issued_secure:
            session.rollback()
            raise StalePostureError(
                f"requested issued_secure={issued_secure} does not match the authoritative "
                f"database posture (current_secure={current_secure}) — refusing to create a "
                f"session under a stale process posture"
            )

        session.add(
            WebSession(
                session_token_hash=token_hash,
                user_id=user_id,
                issued_secure=issued_secure,
                expires_at=expires_at,
            )
        )
        session.commit()


def get_active_sync(*, token_hash: bytes, expected_secure: bool) -> Optional[SessionRecord]:
    """
    Fail-closed resolution: returns None for an UNKNOWN token hash, an
    EXPIRED session, a REVOKED session, and a session bound to the WRONG
    cookie posture, all alike — every one of those collapses to exactly
    the same "not authenticated" outcome for a caller (see
    app/auth_session.py's resolve_session_user_id()), by design: a caller
    probing with a garbage token must not be able to distinguish "no such
    session" from "session exists but is expired/revoked/wrong-posture"
    from the return value alone.

    `expected_secure` (Stage 6A independent-audit corrective pass #2,
    Major 1) is the CURRENT cookie posture the caller expects — a session
    created under one posture (`issued_secure`) must never resolve while
    the application is running under the other, so a secure-production
    deployment can never be authenticated by a bare/insecure-posture
    session even if one somehow reached this call. This filter alone is
    NOT the authoritative revival guarantee — that is create_sync()'s
    transactional posture check plus apply_startup_posture_sync()'s
    transactional revocation sweep (Stage 6A independent-audit corrective
    pass #3; see db/models.py's WebSessionPolicy docstring). No ordinary
    read needs its own lock: a transition may race with an in-flight
    already-authenticated request, and normal in-flight semantics (it
    completes using whatever snapshot it read) are acceptable — the
    critical invariant is only that AFTER a transition commits, an
    incompatible session can never begin a NEW authenticated request, and
    a stale process can never mint a new incompatible one; see
    tests/test_stage6a_corrective2_posture_transition.py and
    tests/test_stage6a_corrective3_policy_race.py.

    `expires_at`/"now" comparison is evaluated by PostgreSQL's own now()
    (func.now()), never Python's local clock — immune to app-server/DB
    clock skew. Both sides are `TIMESTAMP WITH TIME ZONE` (see
    db/models.py's WebSession docstring, Stage 6A independent-audit
    corrective pass #1 Blocker 1), so this is an instant-vs-instant
    comparison, correct regardless of the PostgreSQL session's TimeZone
    GUC — a naive-timestamp column compared this way is NOT
    timezone-independent (proven by the auditor; see
    tests/test_stage6a_corrective1_timezone_expiry.py), which is exactly
    why this table deliberately does NOT follow every other table's plain
    naive-timestamp convention.
    """
    with Session(get_sync_engine()) as session:
        row = session.execute(
            select(WebSession.user_id, WebSession.created_at, WebSession.expires_at).where(
                WebSession.session_token_hash == token_hash,
                WebSession.issued_secure == expected_secure,
                WebSession.revoked_at.is_(None),
                WebSession.expires_at > func.now(),
            )
        ).first()
        if row is None:
            return None
        return SessionRecord(user_id=row.user_id, created_at=row.created_at, expires_at=row.expires_at)


def revoke_sync(*, token_hash: bytes) -> None:
    """Logout: marks the session permanently invalid regardless of its
    `expires_at`. Idempotent — revoking an already-revoked or nonexistent
    token hash is a safe no-op (mirrors db.documents.delete_sync()'s same
    "safe to call unconditionally" contract), so a caller never needs to
    check existence first. Deliberately posture-agnostic (no
    `expected_secure`/policy-lock involvement, unlike create_sync()/
    get_active_sync() above) — revoking a specific, already-known token
    hash by its exact digest needs no posture check of its own; the
    posture invariant only matters for CREATING a new session or
    resolving an arbitrary bearer to a user, not for explicitly revoking
    one you already have the digest of."""
    with Session(get_sync_engine()) as session:
        session.execute(
            update(WebSession)
            .where(WebSession.session_token_hash == token_hash, WebSession.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )
        session.commit()


def apply_startup_posture_sync(
    *, requested_secure: bool, _test_hook_after_lock: Optional[Callable[[], None]] = None
) -> int:
    """
    Transactional posture transition (Stage 6A independent-audit
    corrective pass #3) — run once at every FastAPI startup (web/app.py's
    lifespan), BEFORE the app begins serving requests. Replaces pass #2's
    `revoke_sessions_with_incompatible_posture_sync()`, a bare,
    unsynchronized UPDATE that raced against create_sync() ACROSS PROCESS
    BOUNDARIES: an old, still-running opposite-posture process could
    commit a brand-new, already-incompatible session AFTER a newer
    process's one-time revocation had already run — that late row was
    never swept up by anyone (see db/models.py's WebSessionPolicy
    docstring for the full narrative).

    Within ONE transaction, in this exact order:
      1. `SELECT ... FOR UPDATE` the singleton `web_session_policy` row —
         the EXACT SAME lock create_sync() takes, first thing, before
         doing anything else. Whichever of the two calls (a concurrent
         create_sync() or this function) reaches the lock first runs to
         completion (commit) before the other can even read the row.
      2. Set `current_secure = requested_secure` — unconditionally, even
         if it already matches. Simpler than gating on "did it actually
         change", and still correct/cheap (a single-row UPDATE either
         way): this table has exactly one row.
      3. Revoke EVERY still-active session whose `issued_secure` doesn't
         match `requested_secure`, in the SAME transaction, before the
         lock is released. Running this UNCONDITIONALLY (never gated on
         "posture actually changed", and never skipped merely because
         step 2 found nothing to change) is what makes the "creation won
         the lock race" ordering safe: by the time this statement runs,
         PostgreSQL's READ COMMITTED isolation gives it a fresh
         statement-level snapshot that already includes any row a
         concurrent create_sync() call committed while THIS function was
         blocked waiting for the very same lock.
      4. Commit — releasing the lock, making the new posture and the
         revocation visible together, atomically, to every other
         transaction.

    The two required interleavings this makes safe (see
    tests/test_stage6a_corrective3_policy_race.py for the real-Postgres,
    real-thread proof of both):
      - Creation wins the lock race: it inserts under the OLD posture and
        commits first; this function (unblocked next) still revokes that
        row in step 3, because step 3 always re-observes current state.
      - This function wins the lock race: it changes the posture and
        revokes first; a concurrent create_sync() (unblocked next) then
        sees its own `issued_secure` no longer match and fails closed
        (StalePostureError) without inserting anything.

    Calling this at every startup (not merely "when a change is
    detected") is deliberate: it needs no persisted memory of "the
    previous posture" to compare against, and step 3 is a safe no-op
    (zero rows touched) on any startup where the posture didn't actually
    change since the last one.

    Deliberately narrow: `web_sessions` and `web_session_policy` only —
    never touches `users`, `telegram_accounts`, `user_preferences`, or
    `documents`. Returns the number of `web_sessions` rows revoked
    (informational only — startup logging / regression-test assertions).

    `_test_hook_after_lock`: see create_sync()'s own docstring — the
    identical test-only synchronization seam, mirrored here.
    """
    with Session(get_sync_engine()) as session:
        session.execute(
            select(WebSessionPolicy.id)
            .where(WebSessionPolicy.id == WEB_SESSION_POLICY_ID)
            .with_for_update()
        ).scalar_one()

        if _test_hook_after_lock is not None:
            _test_hook_after_lock()

        session.execute(
            update(WebSessionPolicy)
            .where(WebSessionPolicy.id == WEB_SESSION_POLICY_ID)
            .values(current_secure=requested_secure, updated_at=func.now())
        )
        result = session.execute(
            update(WebSession)
            .where(WebSession.issued_secure != requested_secure, WebSession.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )
        session.commit()
        return result.rowcount
