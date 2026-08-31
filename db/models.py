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
    confirmed) -> 'active' (indexing confirmed). A row stuck at 'pending'
    indicates an interrupted ingest (e.g. a process crash) — the same
    category of orphan a crash can already leave in the physical
    file/sidecar/Qdrant layers today, with no automatic recovery daemon
    (documented as a known follow-up, not implemented here). This table is
    NOT the document content store — the physical file + sidecar remain
    the durable content source (see rag/sidecar.py); this is ownership/
    catalog metadata only."""

    __tablename__ = "documents"
    __table_args__ = (CheckConstraint("status IN ('pending', 'active')", name="status_valid"),)

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
