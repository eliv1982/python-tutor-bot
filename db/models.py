"""
Concrete ORM models (Stage 5C) — plain SQLAlchemy 2.x declarative classes,
not a generic repository/interface hierarchy. One class per persisted
entity; db/identity.py, db/preferences.py, db/documents.py own the actual
operations against these.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class User(Base):
    """Canonical internal user identity. Every other table hangs off this
    UUID — Telegram (and, later, any other adapter) is external identity,
    never itself the primary key of anything else in this schema."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TelegramAccount(Base):
    """Telegram numeric id -> internal user UUID mapping. The Telegram id
    IS the primary key (it is already a natural unique identity — no
    redundant surrogate key), and user_id is UNIQUE so one internal user
    can never accumulate two Telegram accounts pointing at it either
    (linking/merging multiple Telegram accounts to one user is explicitly
    out of Stage 5C's scope)."""

    __tablename__ = "telegram_accounts"

    telegram_user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True
    )
    linked_at: Mapped[datetime] = mapped_column(server_default=func.now())


class UserPreference(Base):
    """Durable mode/voice preferences. One row per user, created lazily on
    first write (see db/preferences.py's upsert helpers) — no separate
    "create preferences row" step. Deliberately does NOT store chat
    history (Stage 5C explicitly keeps that ephemeral/in-memory)."""

    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    mode: Mapped[str | None] = mapped_column(String, nullable=True)
    voice: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Document(Base):
    """Durable ownership/catalog record for a managed Telegram upload.

    `id` deliberately reuses the upload's own storage UUID (the physical
    filename's stem, uuid4().hex — see app/documents.py's
    _store_document_exclusively()) rather than generating a separate id:
    this keeps the DB row, the physical file, the sidecar, and the RAG
    document_id string ("upload:<hex>") in a clean 1:1 correspondence with
    no separate id-mapping table needed.

    `status`: 'pending' (physical file + sidecar durable, indexing not yet
    confirmed) -> 'active' (indexing confirmed) -> 'deleting' (Stage 7A-3:
    an authenticated owner-initiated delete has begun; physical/index/
    catalog cleanup may still be in flight or may have failed partway and
    be awaiting a retry). A row stuck at 'pending' indicates an interrupted
    ingest (e.g. a process crash) — the same category of orphan a crash can
    already leave in the physical file/sidecar/Qdrant layers today, with no
    automatic recovery daemon (documented as a known follow-up, not
    implemented here); a row stuck at 'deleting' is the identical category
    of orphan for an interrupted delete. This table is NOT the document
    content store — the physical file + sidecar remain the durable content
    source (see rag/sidecar.py); this is ownership/catalog metadata only.

    `ACTIVE_STATUSES` (db/documents.py) is exactly `{'active'}` — both
    'pending' and 'deleting' are therefore never retrieval/list/detail
    visible, with no separate visibility flag needed."""

    __tablename__ = "documents"
    __table_args__ = (CheckConstraint("status IN ('pending', 'active', 'deleting')", name="status_valid"),)

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    stored_name: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class WebSession(Base):
    """Server-side authentication session for the FastAPI web adapter
    (Stage 6A) — the browser's Set-Cookie/Cookie value is a high-entropy
    CSPRNG bearer token minted by app/auth_session.py; only its SHA-256
    digest is ever written here (`session_token_hash`, the PK — same
    natural-key-as-PK choice as TelegramAccount.telegram_user_id above, no
    redundant surrogate id needed). A stolen database dump therefore can
    never be replayed as a live browser session; only the actual
    Cookie/Set-Cookie header value would work.

    `session_token_hash` additionally carries an explicit
    `octet_length(session_token_hash) = 32` CHECK constraint
    (`ck_web_sessions_session_token_hash_length`) — Stage 6A independent-
    audit corrective pass #1, Major 2: `LargeBinary(32)` alone compiles to
    an UNCONSTRAINED `BYTEA` on PostgreSQL (the length argument has no
    effect on that dialect, unlike a fixed-width character type), so
    without this CHECK, PostgreSQL itself would silently accept a digest of
    any length — proven exploitable by the auditor inserting a 1-byte
    digest directly. Mirrored in alembic/versions/0002_web_sessions.py so
    `alembic check` stays clean.

    `created_at`/`expires_at`/`revoked_at` are `TIMESTAMP WITH TIME ZONE`
    (`DateTime(timezone=True)`), deliberately DIFFERENT from every other
    table's plain naive `DateTime` in this schema — Stage 6A independent-
    audit corrective pass #1, Blocker 1: `get_active_sync()` below is the
    ONLY place in this codebase that compares a stored timestamp against
    PostgreSQL's own `now()` to make a live authorization decision, and the
    auditor proved that a naive `TIMESTAMP WITHOUT TIME ZONE` column
    compared against `now()` only ever gives correct results when the
    PostgreSQL session's `TimeZone` GUC happens to be UTC — `now()` (a
    `timestamptz`, an absolute instant) gets implicitly cast to the
    session's local wall-clock time before the naive comparison, silently
    extending or shortening effective session lifetime by exactly the
    session's UTC offset when it isn't. `timestamptz` columns compared
    against `timestamptz` `now()` are an instant-vs-instant comparison,
    correct regardless of session `TimeZone` — see
    tests/test_stage6a_corrective1_timezone_expiry.py for the real-
    PostgreSQL proof (session TimeZone explicitly set to
    'America/New_York'). No other table needs this: none of them ever
    compares a stored timestamp against "now" to gate anything.

    `user_id` is deliberately NOT unique (unlike TelegramAccount.user_id):
    one canonical user may hold several concurrent sessions across
    browsers/devices. `revoked_at` NULL = still potentially valid (subject
    to `expires_at`); non-NULL (logout) = permanently invalid regardless of
    `expires_at`. No sliding expiration — `expires_at` is fixed once, at
    creation time (see app/auth_session.py's create_session()).

    `issued_secure` (Stage 6A independent-audit corrective pass #2,
    Major 1) records the cookie posture (`web_config.COOKIE_SECURE`) in
    effect at the moment this session was CREATED — WEB-session metadata,
    never a second user identity. A plain, non-nullable boolean: exactly
    two postures exist today, so there is no ambiguous/ NULL "unknown
    posture" state to guard against. Ordinary authentication
    (db.auth_sessions.get_active_sync()) only ever resolves a session
    whose `issued_secure` matches the CURRENTLY EXPECTED posture — a
    session minted under one posture must never authenticate under the
    other, even transiently.

    The authoritative closure of "a session must never silently become
    valid again after the deployment's posture changes and later
    reverts" is db.auth_sessions.apply_startup_posture_sync() below, run
    once at every FastAPI startup (web/app.py's lifespan) — it
    permanently revokes (not merely filters) every still-active session
    bound to whichever posture is NOT the one the application is starting
    under. Stage 6A independent-audit corrective pass #3 hardened this
    further: apply_startup_posture_sync() alone (pass #2's design) raced
    against db.auth_sessions.create_sync() across PROCESS boundaries — an
    old, still-running opposite-posture process could commit a brand-new,
    now-incompatible session AFTER a newer process's one-time revocation
    had already run, and that late row would never be swept up by
    anyone. Closed by making WebSessionPolicy (below) the single
    PostgreSQL-authoritative source of truth for the current posture, and
    making BOTH apply_startup_posture_sync() and create_sync() acquire
    that singleton row's lock (`SELECT ... FOR UPDATE`) as the FIRST
    statement of their transaction — see WebSessionPolicy's own docstring
    for the full transactional protocol. See
    tests/test_stage6a_corrective2_posture_transition.py for the
    sequential regression proof (the independent auditor's original
    secure->insecure->secure reproduction) and
    tests/test_stage6a_corrective3_policy_race.py for the real-Postgres,
    real-thread proof that BOTH required interleavings between a
    concurrent creation and a concurrent transition resolve safely."""

    __tablename__ = "web_sessions"
    __table_args__ = (
        CheckConstraint("octet_length(session_token_hash) = 32", name="session_token_hash_length"),
    )

    session_token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    issued_secure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# The one and only allowed primary key value for WebSessionPolicy's
# singleton row — see that class's own docstring.
WEB_SESSION_POLICY_ID = 1


class WebSessionPolicy(Base):
    """Authoritative, PostgreSQL-persisted web-session cookie-posture
    policy (Stage 6A independent-audit corrective pass #3) — a singleton
    row (`id` fixed to WEB_SESSION_POLICY_ID, enforced by a CHECK
    constraint so a second row can never exist) recording the ONE
    `current_secure` posture every web process must agree with before it
    may create a new `web_sessions` row.

    This is web-auth RUNTIME POLICY metadata — deliberately not a user
    model, not an identity source, not ownership state, not Telegram
    state, and not read/written by anything outside db/auth_sessions.py.

    Why this table exists (closing corrective pass #2's remaining race):
    pass #2 made `web_config.COOKIE_SECURE` — a PROCESS-LOCAL flag, never
    persisted — the only notion of "current posture", and revoked
    incompatible sessions via a single, unsynchronized UPDATE run once at
    startup. That UPDATE and db.auth_sessions.create_sync()'s INSERT ran
    in independent, unsynchronized transactions, so an old process that
    was still running under the OLD posture (unaware a newer process had
    already transitioned and revoked) could commit a brand-new,
    already-incompatible session AFTER the revocation had already run —
    a session nobody would ever sweep up again.

    The fix makes the database itself, not any one process's memory, the
    single source of truth, and gives session creation and posture
    transition ONE shared serialization point:

    - db.auth_sessions.apply_startup_posture_sync() (posture transition,
      run once at every FastAPI startup) and db.auth_sessions.create_sync()
      (session creation) BOTH acquire THIS row via `SELECT ... FOR UPDATE`
      as the FIRST statement of their transaction, and hold that lock for
      the rest of their transaction (through the UPDATE/INSERT and the
      final commit).
    - Whichever of the two reaches the lock first runs to completion
      (commit) before the other can even read the row — real PostgreSQL
      row-level mutual exclusion, not a process-local lock, an advisory
      comment, or timing-based retries.
    - If creation wins the race: it inserts under the still-old posture,
      commits, releases the lock — then the transition (now unblocked)
      re-reads a FRESH snapshot (READ COMMITTED) that includes the
      just-committed row, and its own unconditional "revoke everything
      incompatible with the posture I'm setting" sweep catches it.
    - If transition wins the race: it changes `current_secure` and
      revokes first, commits, releases the lock — then creation (now
      unblocked) re-reads the row, sees its own `issued_secure` no longer
      matches, and fails closed (StalePostureError, db/auth_sessions.py)
      WITHOUT inserting anything.

    Either ordering leaves the database with no active session under an
    unauthoritative posture. See db/auth_sessions.py's
    apply_startup_posture_sync()/create_sync() for the concrete
    implementation, and tests/test_stage6a_corrective3_policy_race.py for
    the real-PostgreSQL, real-thread proof of both orderings.

    Seeded to `current_secure = true` (the fail-safe default, matching
    web_config.py's own COOKIE_SECURE default) directly by
    alembic/versions/0002_web_sessions.py at migration time — never
    lazily created by application code. This sidesteps the "two first web
    processes racing to create the singleton row on a fresh database"
    bootstrap question entirely: by the time any application code can
    possibly run, the schema (and this seeded row) already exists,
    because Alembic migrations are the only thing that ever creates
    tables in this project (see README.md's "Схема создаётся ТОЛЬКО через
    Alembic" section)."""

    __tablename__ = "web_session_policy"
    __table_args__ = (CheckConstraint(f"id = {WEB_SESSION_POLICY_ID}", name="singleton_id"),)

    # autoincrement=False: `id` is a fixed, always-1 natural key (the CHECK
    # constraint enforces it), never a generated surrogate — same pattern
    # TelegramAccount.telegram_user_id above already uses for the same
    # reason. Without this, SQLAlchemy's implicit-autoincrement rule for a
    # lone integer PK column creates an actual, entirely unused PostgreSQL
    # SEQUENCE for a table that only ever has exactly one row.
    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    current_secure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class GithubAccount(Base):
    """GitHub numeric user id -> internal user UUID mapping (Stage 6B) —
    mirrors TelegramAccount above exactly, and is subject to the identical
    Stage 6C non-merge boundary: `user_id` is UNIQUE, so one internal user
    can never accumulate two GitHub accounts, and linking an existing
    Telegram-backed user to a GitHub identity (or vice versa) is
    explicitly out of Stage 6B's scope — a human with both a Telegram-
    backed canonical user and a GitHub-backed one intentionally ends up
    with two separate canonical users until Stage 6C's explicit linking
    ships. See db/github_identity.py's own module docstring.

    `github_user_id` is GitHub's own stable NUMERIC `id` from
    `GET https://api.github.com/user` — NEVER the login/username (which
    can change) and NEVER email (which may be absent/private/change). It
    IS the primary key — same natural-key-as-PK choice as
    TelegramAccount.telegram_user_id above, no redundant surrogate key
    needed."""

    __tablename__ = "github_accounts"

    github_user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True
    )
    linked_at: Mapped[datetime] = mapped_column(server_default=func.now())


class GithubOAuthTransaction(Base):
    """Short-lived, replay-resistant server-side OAuth transaction state
    (Stage 6B) for the GitHub Authorization Code + PKCE flow — see
    db/oauth_transactions.py for the create/claim operations and
    web/github_oauth.py for how the login flow actually uses them.

    `state_hash` (SHA-256 digest of the random `state` value handed to
    GitHub) is the PRIMARY KEY — the exact same "digest only, never the
    raw bearer/secret value itself" design as WebSession.
    session_token_hash above, and the same `octet_length(...) = 32` CHECK
    constraint for the identical reason documented on that model:
    `LargeBinary(32)` alone compiles to an unconstrained BYTEA on
    PostgreSQL and enforces nothing server-side without it. A stolen
    database dump can therefore never be replayed as a live OAuth
    callback without also knowing the actual `state` value that hashes to
    a given row — and even then, db.oauth_transactions.claim_sync()'s
    atomic single-use DELETE ... RETURNING means it can be redeemed at
    most once.

    `code_verifier` IS stored in cleartext, deliberately unlike
    `state_hash` — see db/oauth_transactions.py's own module docstring for
    why that is an acceptable trust boundary here (it never crosses the
    browser, unlike `state`, which a network observer watching the
    GitHub redirect can always see regardless of how it's stored here).

    `auth_generation` (Stage 6C corrective pass, independent-audit MAJOR 1)
    is the `github_oauth_admission.unlink_generation` value in effect at
    the moment `db.oauth_transactions.create_sync()` created this row —
    captured under that singleton's own lock, in the same transaction (see
    GithubOAuthAdmission's own docstring below). Carried through
    claim_sync() unchanged and handed to
    db.github_identity.resolve_or_create_user_by_github_id_for_oauth_sync()
    at callback time: that resolver rejects the transaction outright if a
    LATER unlink has since tombstoned this GitHub identity at a higher
    generation, closing a real race where a callback that already
    authenticated with GitHub — but had not yet resolved/created its
    canonical mapping — could otherwise recreate (or re-attach to) a
    mapping a concurrent unlink just tore down. NOT NULL with a `>= 0`
    CHECK and a `0` default: a transaction created before this column
    existed (impossible in practice — transactions are short-lived — but
    also a transaction created without an explicit value) reads as
    generation 0, the earliest possible generation, so it is rejected by
    any tombstone at all, never treated as inherently trustworthy. See
    db/telegram_link.py's module docstring for the complete protocol.

    `created_at`/`expires_at` are `TIMESTAMP WITH TIME ZONE` — same
    rationale as WebSession above (Stage 6A independent-audit corrective
    pass #1, Blocker 1): db.oauth_transactions.claim_sync() and
    create_sync()'s own expired-row cleanup both compare `expires_at`
    against PostgreSQL's own now(), so this must be an instant-vs-instant
    comparison, correct regardless of the PostgreSQL session's TimeZone
    GUC. `expires_at` also carries a plain b-tree index — create_sync()'s
    admission-path cleanup (independent-audit corrective pass #1, MAJOR 2)
    deletes every row past its expiry on every call, and this index is
    what keeps that a cheap, indexed operation rather than a growing
    sequential scan as the table churns.

    independent-audit corrective pass #1 (MAJOR 2, Section 8) REMOVED the
    previous `consumed_at` column entirely: claim_sync() now atomically
    DELETEs the row it claims (`DELETE ... WHERE state_hash = :h AND
    expires_at > now() ... RETURNING code_verifier`) instead of merely
    marking it consumed with an UPDATE. Replay-safety is unchanged (a
    second claim attempt matches zero rows — the row is simply gone,
    exactly as final as "already consumed" was) and this closes two
    findings at once: the cleartext PKCE verifier of a completed
    transaction no longer lingers in the table indefinitely (Section 8 —
    "do not retain cleartext PKCE verifiers indefinitely"), and every row
    remaining in this table at any moment is, by construction, still a
    genuinely live, unclaimed, unexpired transaction — which is exactly
    the "outstanding transaction count" create_sync()'s own admission
    control (db/models.py's GithubOAuthAdmission, below) counts against
    its hard cap. No row is ever left behind after either a successful
    claim (deleted here) or an expiry sweep (deleted by create_sync()'s
    own cleanup step) — there is no more "dead but not yet cleaned up"
    row category left for this table at all."""

    __tablename__ = "github_oauth_transactions"
    __table_args__ = (
        CheckConstraint("octet_length(state_hash) = 32", name="state_hash_length"),
        CheckConstraint("auth_generation >= 0", name="auth_generation_non_negative"),
    )

    state_hash: Mapped[bytes] = mapped_column(LargeBinary(32), primary_key=True)
    code_verifier: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    auth_generation: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")


# The one and only allowed primary key value for GithubOAuthAdmission's
# singleton row — same pattern as WEB_SESSION_POLICY_ID above.
GITHUB_OAUTH_ADMISSION_ID = 1


class GithubOAuthAdmission(Base):
    """Singleton, PostgreSQL-authoritative GLOBAL admission-control state
    for GitHub OAuth login starts (Stage 6B independent-audit corrective
    pass #1, MAJOR 2) — closes "an unauthenticated attacker can grow
    `github_oauth_transactions` without bound by hammering `GET
    /api/auth/github/login`" by making db.oauth_transactions.create_sync()
    a single atomic transaction that locks THIS row first
    (`SELECT ... FOR UPDATE`, the exact same singleton-row-lock idiom
    WebSessionPolicy above already established for db.auth_sessions.
    create_sync()/apply_startup_posture_sync()), then — all inside that
    one lock — deletes expired transaction rows, resets/advances a fixed
    GLOBAL rate window, and refuses to insert a new transaction row at all
    if either the rate window or the outstanding-row hard cap
    (`COUNT(*)` of `github_oauth_transactions` after cleanup) is already
    exhausted. Real PostgreSQL row-level mutual exclusion serializes this
    across every FastAPI worker/process sharing the same database — never
    a process-local counter, which would only bound one process at a time.

    Deliberately GLOBAL, not per-IP: this application has no trustworthy
    reverse-proxy client-IP contract yet (Stage 6B is not the deployment
    stage — see README.md's production deployment invariants), so a
    per-IP scheme here would either trust a spoofable header or persist
    IP addresses for no real benefit. A global bound is sufficient to make
    storage growth impossible and keeps this table entirely free of any
    new PII. Future reverse-proxy/edge rate limiting remains valid
    defense-in-depth on top of this, never a substitute for it (see
    db/oauth_transactions.py's own module docstring).

    `window_start`/`starts_in_window` implement one fixed-size sliding
    window (never persisted per caller): `starts_in_window` counts
    admitted `/login` starts since `window_start`; once
    `now() - window_start` exceeds the window length, create_sync() resets
    both back to a fresh window as part of the same locked transaction.
    Rejected attempts (rate-limited OR over the outstanding-row hard cap)
    never increment this counter and never insert a transaction row — a
    rejection costs the database nothing but the (already-necessary)
    singleton-row lock and the cleanup DELETE.

    `id` is CHECK-constrained to exactly GITHUB_OAUTH_ADMISSION_ID — same
    "a second row can never exist" enforcement as WebSessionPolicy.id
    above. Seeded with its one row directly by
    alembic/versions/0003_github_oauth.py at migration time, never lazily
    created by application code, for the identical "sidesteps a first-
    process bootstrap race entirely" reason WebSessionPolicy's own
    docstring documents.

    `unlink_generation` (Stage 6C corrective pass, independent-audit
    MAJOR 1) is the ONE global, monotonically-increasing OAuth-generation
    counter this application maintains — reusing THIS row (rather than a
    new singleton) so "capture the current generation at login" (db.
    oauth_transactions.create_sync(), which already locks this row first)
    and "advance the generation at unlink" (db.telegram_link.
    unlink_github_sync(), which acquires this same row's lock as the LAST
    step of a successful unlink) share one existing lock idiom instead of
    inventing a second. It is deliberately a single GLOBAL counter, not one
    per GitHub identity: every successful unlink — regardless of which
    GitHub id it affects — advances it, and db.github_identity.
    resolve_or_create_user_by_github_id_for_oauth_sync() compares an OAuth
    transaction's captured `auth_generation` against the PER-IDENTITY
    tombstone this same unlink writes (GithubUnlinkTombstone below) to
    decide staleness; the global counter only needs to be monotonic, never
    partitioned, for that comparison to be correct. NOT NULL with a `>= 0`
    CHECK and a `0` default (added by this same migration, alongside
    `github_oauth_transactions.auth_generation`)."""

    __tablename__ = "github_oauth_admission"
    __table_args__ = (
        CheckConstraint(f"id = {GITHUB_OAUTH_ADMISSION_ID}", name="singleton_id"),
        CheckConstraint("unlink_generation >= 0", name="unlink_generation_non_negative"),
    )

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    starts_in_window: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    unlink_generation: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")


class GithubUnlinkTombstone(Base):
    """Durable, per-GitHub-identity record of the LATEST successful unlink's
    generation (Stage 6C corrective pass, independent-audit MAJOR 1) — one
    bounded row per GitHub identity that has EVER been unlinked, written
    atomically by db.telegram_link.unlink_github_sync() in the same
    transaction as the generation bump on GithubOAuthAdmission above and
    the unlink's own mutations (mapping removal, session revocation/
    deletion, user deletion where applicable).

    `github_user_id` is the PRIMARY KEY (mirrors GithubAccount.
    github_user_id's own natural-key choice) — at most one tombstone row
    per GitHub identity; a SECOND unlink of the same identity (after a
    fresh re-link) simply advances this row's `unlink_generation` via
    `INSERT ... ON CONFLICT DO UPDATE` rather than accumulating history,
    since only the LATEST generation ever matters for the staleness
    comparison db.github_identity.
    resolve_or_create_user_by_github_id_for_oauth_sync() performs.

    Deliberately carries NO foreign key to `github_accounts`: the entire
    point of this table is to remember a GitHub identity's unlink history
    AFTER its `github_accounts` mapping has been removed (that removal is
    exactly what triggers writing/advancing this row) — an FK to a
    provider mapping that no longer exists by design would be incoherent,
    unlike every other cross-table reference in this schema.

    `unlink_generation` carries a `> 0` CHECK (never `>= 0`, unlike the
    admission singleton's own generation column): a tombstone is only ever
    written as the result of a successful unlink, which always advances
    the global counter from its current value to a strictly higher one
    (starting from 1) — a tombstone at generation 0 could never mean
    anything meaningful (every OAuth transaction's own default
    `auth_generation` is already 0, so a generation-0 tombstone could never
    reject anything) and would only ever indicate a persistence bug.

    `unlinked_at` is `TIMESTAMP WITH TIME ZONE`, diagnostic/observability
    only — no code path in this application compares it against `now()` or
    otherwise makes an authorization decision from it; `unlink_generation`
    alone is the authoritative signal."""

    __tablename__ = "github_unlink_tombstones"
    __table_args__ = (CheckConstraint("unlink_generation > 0", name="unlink_generation_positive"),)

    github_user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    unlink_generation: Mapped[int] = mapped_column(BigInteger, nullable=False)
    unlinked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class TelegramLinkAttempt(Base):
    """Short-lived, replay-resistant server-side state for one authenticated
    web (GitHub) user's outstanding request to link their canonical
    identity to a Telegram account (Stage 6C) — see db/telegram_link.py for
    the create/redeem operations and web/routes.py's
    POST /api/link/telegram/start for how the flow starts it, and
    handlers/start.py's `/start link_<secret>` payload for how it is
    redeemed.

    `web_user_id` (not a surrogate id) is the PRIMARY KEY — this table holds
    AT MOST ONE outstanding attempt per canonical user by construction (a
    second POST /api/link/telegram/start atomically SUPERSEDES the first;
    see db/telegram_link.py's create_attempt_sync()), so there is no
    redundant surrogate key to add. `ON DELETE RESTRICT` (explicit, not the
    unspecified-FK default used elsewhere in this schema) is deliberate: a
    `users` row must never be deleted while it still holds an outstanding
    link attempt — every code path that deletes a `users` row (Stage 6C's
    merge-on-redemption and GitHub-only unlink) is REQUIRED to delete this
    row itself, in the same transaction, BEFORE the `users` row — as part of
    the corrected lock order (Stage 6C corrective pass, independent-audit
    MAJOR 2 — see db/telegram_link.py's own module docstring for the full
    protocol): advisory lock, where one applies -> `github_accounts` rows ->
    `users` rows -> (success path only) `web_session_policy`/
    `github_oauth_admission` -> this table's row, ALWAYS LAST, mutated via
    one atomic statement (never a separate prior lock/probe step). RESTRICT
    is a defense-in-depth backstop making "never leave a `users` row FK-
    orphaned by this table" a hard database invariant, not merely an
    application-level convention — it says nothing about lock ORDER, which
    is the opposite of this table's actual position in it (LAST, not
    first).

    `link_secret_hash` (SHA-256 digest of the raw bearer secret handed to
    the browser as part of the `https://t.me/<bot>?start=link_<secret>`
    deep link) is UNIQUE and NOT the primary key — mirrors WebSession.
    session_token_hash's/GithubOAuthTransaction.state_hash's own "digest
    only, never the raw secret" design (see those models' docstrings): a
    stolen database dump can never be replayed as a valid redemption
    without also knowing the raw secret that hashes to a given row. It is
    UNIQUE (not the PK) because the natural lookup key for CREATING/
    SUPERSEDING an attempt is `web_user_id` (one outstanding attempt per
    user), while the natural lookup key for REDEEMING one is the secret's
    digest — both must be fast, indexed lookups, hence a unique index on
    each. The same `octet_length(link_secret_hash) = 32` CHECK constraint
    every other digest column in this schema carries, for the identical
    reason (`sa.LargeBinary(length=32)` alone compiles to an unconstrained
    BYTEA on PostgreSQL).

    `created_at`/`expires_at` are `TIMESTAMP WITH TIME ZONE` — same
    rationale as every other short-lived bearer-secret table in this schema
    (WebSession, GithubOAuthTransaction): db.telegram_link.py's redemption
    path compares `expires_at` against PostgreSQL's own now(), which must be
    an instant-vs-instant comparison, correct regardless of the PostgreSQL
    session's TimeZone GUC. `expires_at` carries a plain b-tree index for
    the same reason GithubOAuthTransaction.expires_at does: bounded,
    indexed expired-row cleanup (see db/telegram_link.py's
    cleanup_expired_attempts_sync()) must never degrade to a sequential
    scan as this table churns."""

    __tablename__ = "telegram_link_attempts"
    __table_args__ = (
        CheckConstraint("octet_length(link_secret_hash) = 32", name="link_secret_hash_length"),
    )

    web_user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), primary_key=True
    )
    link_secret_hash: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
