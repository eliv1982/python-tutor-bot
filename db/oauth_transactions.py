"""
Server-side GitHub OAuth transaction persistence (Stage 6B) — replay-
resistant, short-lived state for the Authorization Code + PKCE flow
(web/github_oauth.py). SYNC, deliberately (see db/engine.py's module
docstring for why), wrapped in asyncio.to_thread() by
app/oauth_transaction.py, the same idiom db/identity.py/db/auth_sessions.py
already use for this codebase's other blocking-I/O boundaries.

Only a SHA-256 digest of the random `state` value is ever persisted
(`state_hash`, the PK) — never the raw state itself — mirroring
db/models.py's WebSession `session_token_hash` design exactly (see that
model's own docstring, and db/models.py's GithubOAuthTransaction docstring
above it): a stolen database dump can never be replayed as a live OAuth
callback without also knowing the actual `state` value that hashes to a
given row.

The PKCE `code_verifier` IS stored in cleartext — unlike `state`, it is
never observable by a network attacker who merely watches the browser's
GitHub redirect (it never leaves this server: see app/oauth_transaction.py
and services/github_oauth_client.py), and the token-exchange step needs
the exact original value handed back. Its confidentiality rests on
ordinary database access control, the same trust boundary every other
plaintext column in this schema already relies on. claim_sync() DELETEs
the row it claims (see its own docstring), so that cleartext value never
outlives its one legitimate use.

Stage 6B independent-audit corrective pass #1, MAJOR 2 — unbounded public
storage growth: EVERY unauthenticated `GET /api/auth/github/login` used to
unconditionally commit a new row here, with no cleanup, no cap, and no
rate limit — an attacker could grow this table (and PostgreSQL's disk
usage) without bound merely by repeating the request. create_sync() below
is now a single atomic transaction, serialized across every FastAPI
worker/process via `SELECT ... FOR UPDATE` on the
db.models.GithubOAuthAdmission singleton row (the exact row-lock idiom
db/auth_sessions.py's create_sync()/apply_startup_posture_sync() already
established for web_session_policy — see that module's own docstring),
that (1) deletes every already-expired transaction row, (2) resets or
advances a fixed-length GLOBAL rate window, and (3) refuses to insert a
new transaction at all if either the rate window or the outstanding-row
hard cap is already exhausted — all before a row is ever inserted. A
rejection is reported back to the caller (return value `False`) rather
than raised as a bare exception: it is an ordinary, expected outcome under
load/abuse, not a persistence-layer error. See db/models.py's
GithubOAuthAdmission docstring for the full rationale, and
tests/test_stage6b_corrective1_oauth_admission.py for the real-PostgreSQL,
real-thread proof that concurrent callers can never overshoot either
bound.

This is deliberately GLOBAL, never per-client-IP — see db/models.py's
GithubOAuthAdmission docstring for why a per-IP scheme would be premature
(no trustworthy reverse-proxy IP contract yet) and unnecessary (a global
bound already makes storage growth impossible). A future reverse-proxy/
edge rate limit in front of `/api/auth/github/login` remains valid
defense-in-depth on top of this, documented as a production requirement in
README.md — never a substitute for this application/database-layer bound.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import GITHUB_OAUTH_ADMISSION_ID, GithubOAuthAdmission, GithubOAuthTransaction


@dataclass(frozen=True)
class ClaimedTransaction:
    """Returned ONLY by claim_sync() (Stage 6C corrective pass,
    independent-audit MAJOR 1) — carries the transaction's captured
    `auth_generation` alongside its `code_verifier` so the callback can
    perform the generation-aware resolution
    (db.github_identity.resolve_or_create_user_by_github_id_for_oauth_sync())
    without a second round trip. Replaces claim_sync()'s previous bare
    `Optional[str]` return."""

    code_verifier: str
    auth_generation: int


def create_sync(
    *,
    state_hash: bytes,
    code_verifier: str,
    expires_at: datetime,
    max_starts_per_window: int,
    max_outstanding: int,
    window_seconds: int,
    _test_hook_after_lock: Optional[Callable[[], None]] = None,
) -> bool:
    """
    Admission-controlled transaction creation (Stage 6B independent-audit
    corrective pass #1, MAJOR 2). `expires_at` is computed by the caller
    (app/oauth_transaction.py, from
    github_oauth_config.OAUTH_TRANSACTION_TTL_SECONDS); `max_starts_per_window`/
    `max_outstanding`/`window_seconds` are likewise resolved by the caller
    from github_oauth_config — this function performs no configuration
    parsing of its own, mirroring db.auth_sessions.create_sync()'s own
    split of concerns.

    Returns True (row inserted) or False (admission rejected — the rate
    window or the outstanding-row hard cap was already exhausted; NO row
    was inserted and the transaction was rolled back). Never raises for an
    ordinary rejection — a caller under load/abuse is an expected
    condition, not a persistence error.

    The row lock on the GithubOAuthAdmission singleton (`SELECT ... FOR
    UPDATE`, acquired FIRST and held through the final commit/rollback) is
    the ONE serialization point shared by every concurrent call to this
    function, across every process sharing this database — see
    db/models.py's GithubOAuthAdmission docstring for the full protocol.
    Everything below runs on that same locked transaction:

      1. Delete every `github_oauth_transactions` row whose `expires_at`
         is already past PostgreSQL's own now() — bounded, indexed
         (ix_github_oauth_transactions_expires_at) cleanup that runs on
         every single call, so abandoned transactions never require a
         separate background worker to be swept up (Section 9).
      2. If the current rate window has elapsed, reset it (fresh
         `window_start`/`starts_in_window = 0`) — a plain fixed-window
         counter, deliberately not a sliding-log: simple, O(1) storage,
         and sufficient for this application's low interactive volume.
      3. Reject (rollback, return False) if `starts_in_window` already
         reached `max_starts_per_window` — the GLOBAL rate bound.
      4. Reject (rollback, return False) if the outstanding row count
         (COUNT(*) of `github_oauth_transactions`, taken AFTER step 1's
         cleanup — see db/models.py's GithubOAuthTransaction docstring for
         why every remaining row is, by construction, still genuinely
         live) already reached `max_outstanding` — the hard physical-
         storage bound, independent of the rate window (a slow trickle of
         starts that are never completed/expired can still fill this).
      5. Otherwise: increment `starts_in_window`, insert the new
         transaction row, commit.

    `_test_hook_after_lock`: test-only synchronization seam, never passed
    by any real (non-test) caller — identical idiom to
    db.auth_sessions.create_sync()'s own parameter of the same name (see
    that function's docstring). If provided, it is called with no
    arguments immediately after the admission row lock is acquired, before
    any of steps 1-5 — a test can use it to pause this transaction open
    (via a blocking threading.Event.wait()) for as long as needed to
    deterministically force a concurrent caller to contend on the same
    lock, without any time.sleep()-based race.

    Stage 6C corrective pass (independent-audit MAJOR 1): while this same
    locked transaction holds the GithubOAuthAdmission row, it also reads
    `unlink_generation` (the ONE global, monotonically-increasing OAuth-
    generation counter — see that model's own docstring) and stores it
    verbatim as the new transaction's `auth_generation`. This is the ONLY
    place `auth_generation` is ever captured — never derived from an
    application timestamp or process-local state — and reusing the
    already-locked admission row for this read (rather than a second lock)
    is what guarantees the captured value is exactly the generation in
    effect at the instant this transaction is admitted, with no window for
    a concurrent unlink to advance it in between.
    """
    with Session(get_sync_engine()) as session:
        window_start, starts_in_window, unlink_generation = session.execute(
            select(
                GithubOAuthAdmission.window_start,
                GithubOAuthAdmission.starts_in_window,
                GithubOAuthAdmission.unlink_generation,
            )
            .where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
            .with_for_update()
        ).one()

        if _test_hook_after_lock is not None:
            _test_hook_after_lock()

        session.execute(delete(GithubOAuthTransaction).where(GithubOAuthTransaction.expires_at <= func.now()))

        now = session.execute(select(func.now())).scalar_one()
        if now - window_start >= timedelta(seconds=window_seconds):
            window_start = now
            starts_in_window = 0

        if starts_in_window >= max_starts_per_window:
            session.rollback()
            return False

        outstanding = session.execute(select(func.count()).select_from(GithubOAuthTransaction)).scalar_one()
        if outstanding >= max_outstanding:
            session.rollback()
            return False

        session.execute(
            update(GithubOAuthAdmission)
            .where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
            .values(window_start=window_start, starts_in_window=starts_in_window + 1)
        )
        session.add(
            GithubOAuthTransaction(
                state_hash=state_hash,
                code_verifier=code_verifier,
                expires_at=expires_at,
                auth_generation=unlink_generation,
            )
        )
        session.commit()
        return True


def claim_sync(*, state_hash: bytes) -> Optional[ClaimedTransaction]:
    """
    Atomic, single-use claim: DELETEs the transaction and returns its
    `code_verifier` (plus, Stage 6C corrective pass MAJOR 1, its captured
    `auth_generation` — see ClaimedTransaction above) in ONE statement
    (`DELETE ... RETURNING`), or None if
    no row exists / it has already been claimed / it has expired — all
    three collapse to the exact same fail-closed outcome, mirroring
    db.auth_sessions.get_active_sync()'s own "unknown vs. expired vs.
    revoked are indistinguishable to the caller" design: a caller probing
    with a garbage/replayed state must not be able to tell those cases
    apart from the return value alone.

    Stage 6B independent-audit corrective pass #1, Section 8: this used to
    be an `UPDATE ... SET consumed_at = now() ... RETURNING`, leaving a
    consumed row (and its cleartext PKCE verifier) in the table
    indefinitely. DELETE ... RETURNING gives the identical single-use
    guarantee — a second claim's WHERE predicate matches zero rows because
    the row is simply gone, exactly as final as "already consumed" was —
    while also removing the cleartext verifier immediately and keeping
    every row remaining in the table a genuinely live, unclaimed
    transaction (see db/models.py's GithubOAuthTransaction docstring).

    Needs no separate advisory lock (unlike
    db.identity.resolve_or_create_user_by_*_id_sync()'s first-creation
    race, or this module's own create_sync() admission control): a single
    `DELETE ... WHERE state_hash = ... AND expires_at > now() ...
    RETURNING` is already an atomic, row-locking statement in PostgreSQL.
    Two concurrent callbacks racing for the same state_hash: whichever
    transaction's DELETE reaches the row first takes its row lock,
    deletes it, and commits; the second, once unblocked by that commit,
    re-evaluates the SAME WHERE predicate under READ COMMITTED and finds
    no matching row at all — it matches zero rows and gets None, never the
    verifier a concurrent caller already claimed. This is the real-
    PostgreSQL proof required for "two concurrent callbacks must not both
    succeed" (see tests/test_stage6b_oauth_transactions.py).

    An EXPIRED-but-never-claimed row is deliberately left untouched by
    this function (the WHERE predicate excludes it, matching zero rows —
    it is neither claimed nor deleted here); it is swept up by
    create_sync()'s own expired-row cleanup instead (Section 9), the same
    "claim never deletes an expired row itself" behavior the previous
    UPDATE-based version had for `consumed_at`.

    `expires_at > func.now()` is evaluated by PostgreSQL's own now(),
    never Python's local clock — immune to app-server/DB clock skew, and
    (both sides being `TIMESTAMP WITH TIME ZONE`) correct regardless of
    the PostgreSQL session's TimeZone GUC, same rationale as
    db.auth_sessions.get_active_sync()'s own expiry comparison.
    """
    with Session(get_sync_engine()) as session:
        result = session.execute(
            delete(GithubOAuthTransaction)
            .where(
                GithubOAuthTransaction.state_hash == state_hash,
                GithubOAuthTransaction.expires_at > func.now(),
            )
            .returning(GithubOAuthTransaction.code_verifier, GithubOAuthTransaction.auth_generation)
        )
        row = result.first()
        session.commit()
        if row is None:
            return None
        return ClaimedTransaction(code_verifier=row[0], auth_generation=row[1])
