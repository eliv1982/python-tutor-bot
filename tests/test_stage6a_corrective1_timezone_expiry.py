"""
Stage 6A independent-audit corrective pass #1 — Blocker 1 real-PostgreSQL
regression proof: session expiry must be correct regardless of the
PostgreSQL session's `TimeZone` GUC.

Pre-fix behavior: `web_sessions.expires_at` was a naive `TIMESTAMP WITHOUT
TIME ZONE`, compared against PostgreSQL's own `now()` in
db.auth_sessions.get_active_sync(). The auditor set the PostgreSQL session
TimeZone to America/New_York and reproduced
`expired_by_utc=true|accepted_by_current_predicate=true`: `now()` (an
absolute instant, a `timestamptz`) gets implicitly CAST to the session's
local wall-clock time before a naive comparison, so a session that had
already expired by a UTC clock could still read as valid whenever the
session TimeZone was behind UTC — silently extending effective session
lifetime by the session's UTC offset.

Fix: `expires_at`/`created_at`/`revoked_at` are now `TIMESTAMP WITH TIME
ZONE` (see db/models.py's WebSession docstring) — an instant-vs-instant
comparison against `now()`, correct no matter what the session's TimeZone
GUC is.

Every test here deliberately runs against a connection whose PostgreSQL
session TimeZone has been forced to a specific non-UTC zone via a
SQLAlchemy `engine.connect` event (the only reliable way to guarantee EVERY
connection the shared pooled Engine hands out for the rest of the test
actually ran `SET TIME ZONE`, since `ALTER ROLE ... SET` only affects
connections established afterward, and a bare `SET TIME ZONE` on one
borrowed connection wouldn't reliably apply to a different one the pool
might hand back later) — never mocked datetime/database comparisons.
"""

import random
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
from db.engine import get_sync_engine


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module needs
    a REAL `users` row, never the offline in-memory fake."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _token_hash(raw: str) -> bytes:
    import hashlib
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _force_session_timezone_on_every_future_connection(engine, tz_name: str) -> None:
    """Registers a `connect` listener that runs `SET TIME ZONE <tz_name>`
    (then commits — `SET` without LOCAL is otherwise transactional and
    would be rolled back with the connection's first implicit transaction,
    undoing it before any later statement observes it) on every NEW
    physical DBAPI connection this Engine creates from here on, then
    disposes the pool's currently-open connections so the very next
    checkout is guaranteed to be a freshly-connected one that actually ran
    it — a connection already sitting in the pool from before this call
    would not have."""
    def _on_connect(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute(f"SET TIME ZONE '{tz_name}'")
        cursor.close()
        dbapi_connection.commit()

    event.listen(engine, "connect", _on_connect)
    engine.dispose()


def _reported_session_timezone(engine) -> str:
    with Session(engine) as session:
        return session.execute(text("SHOW TIME ZONE")).scalar_one()


@pytest.mark.asyncio
async def test_utc_expired_session_is_rejected_under_a_non_utc_session_timezone(postgres_db):
    """The exact scenario the auditor reproduced: America/New_York is
    behind UTC, which is the dangerous direction (it's the one that used to
    silently EXTEND apparent validity, not shorten it)."""
    engine = get_sync_engine()
    _force_session_timezone_on_every_future_connection(engine, "America/New_York")

    reported_tz = _reported_session_timezone(engine)
    assert reported_tz not in ("UTC", "Etc/UTC"), (
        f"test setup failed to actually shift the PostgreSQL session TimeZone "
        f"(reported {reported_tz!r}) — this test would be vacuous"
    )

    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _token_hash(raw_token)
    # Already expired by a UTC clock, one hour ago — under the pre-fix
    # naive-timestamp column, America/New_York's ~4-5 hour negative offset
    # would have made this still read as "not yet expired" to PostgreSQL's
    # own now()-based comparison.
    expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

    db_auth_sessions.create_sync(token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at)

    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None
    assert await auth_session.resolve_session_user_id(raw_token, expected_secure=True) is None


@pytest.mark.asyncio
async def test_utc_valid_session_still_authenticates_under_a_non_utc_session_timezone(postgres_db):
    """Companion positive proof: the fix must not overcorrect into
    rejecting a genuinely still-valid session merely because the
    PostgreSQL session TimeZone is non-UTC."""
    engine = get_sync_engine()
    _force_session_timezone_on_every_future_connection(engine, "America/New_York")
    assert _reported_session_timezone(engine) not in ("UTC", "Etc/UTC")

    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _token_hash(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    db_auth_sessions.create_sync(token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at)

    record = db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True)
    assert record is not None
    assert record.user_id == user_id
    assert await auth_session.resolve_session_user_id(raw_token, expected_secure=True) == user_id


@pytest.mark.asyncio
async def test_utc_expired_session_is_rejected_under_a_positive_offset_timezone(postgres_db):
    """Independence proof in the OTHER direction too (Asia/Tokyo, UTC+9) —
    the fix must be a genuine instant-vs-instant comparison, not merely
    "happens to work for America/New_York"."""
    engine = get_sync_engine()
    _force_session_timezone_on_every_future_connection(engine, "Asia/Tokyo")
    assert _reported_session_timezone(engine) not in ("UTC", "Etc/UTC")

    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _token_hash(raw_token)
    expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

    db_auth_sessions.create_sync(token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at)

    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None
    assert await auth_session.resolve_session_user_id(raw_token, expected_secure=True) is None
