"""
Stage 6A regression tests: server-side web-session persistence
(db.auth_sessions) against a REAL disposable PostgreSQL container —
proving genuine database behavior (the FK constraint, PostgreSQL's own
now() driving expiry), never pretend/mocked behavior. See
tests/conftest.py's postgres_container()/postgres_db() fixtures.

Skips cleanly (not a failure) if Docker/the postgres:16-alpine image is
unavailable. Every test truncates its tables fresh via postgres_db.
"""

import hashlib
import random
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
from db.engine import get_sync_engine
from db.models import WebSession


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture (same mechanism
    tests/test_stage5c_preferences.py already uses) — this module needs a
    REAL `users` row to satisfy web_sessions' FK constraint, never the
    offline in-memory fake."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _token_hash(raw: str) -> bytes:
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _aware_utcnow() -> datetime:
    """web_sessions.expires_at is a `TIMESTAMP WITH TIME ZONE` column
    (independent-audit corrective pass #1, Blocker 1 — see db/models.py's
    WebSession docstring) — these tests call db.auth_sessions directly, so
    (unlike app/auth_session.py's create_session()) they must supply an
    aware datetime themselves. Renamed from this file's previous
    `_naive_utcnow()`, which stripped tzinfo before this pass: that
    stripping is exactly the naive-timestamp behavior the audit proved
    unsafe (session validity became dependent on the PostgreSQL session's
    TimeZone GUC) — see
    tests/test_stage6a_corrective1_timezone_expiry.py for the real-
    PostgreSQL regression proof."""
    return datetime.now(timezone.utc)


def test_create_then_get_active_resolves_the_correct_user(postgres_db):
    user_id = _real_user()
    token_hash = _token_hash(secrets.token_urlsafe(32))
    expires_at = _aware_utcnow() + timedelta(hours=1)

    db_auth_sessions.create_sync(token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at)
    record = db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True)

    assert record is not None
    assert record.user_id == user_id


def test_unknown_token_hash_resolves_to_none(postgres_db):
    assert db_auth_sessions.get_active_sync(token_hash=_token_hash("never-created"), expected_secure=True) is None


def test_expired_session_resolves_to_none(postgres_db):
    user_id = _real_user()
    token_hash = _token_hash(secrets.token_urlsafe(32))
    db_auth_sessions.create_sync(
        token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=_aware_utcnow() - timedelta(seconds=1)
    )

    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None


def test_revoked_session_resolves_to_none(postgres_db):
    user_id = _real_user()
    token_hash = _token_hash(secrets.token_urlsafe(32))
    db_auth_sessions.create_sync(
        token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=_aware_utcnow() + timedelta(hours=1)
    )

    db_auth_sessions.revoke_sync(token_hash=token_hash)

    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None


def test_revoke_is_idempotent_for_an_already_revoked_session(postgres_db):
    user_id = _real_user()
    token_hash = _token_hash(secrets.token_urlsafe(32))
    db_auth_sessions.create_sync(
        token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=_aware_utcnow() + timedelta(hours=1)
    )

    db_auth_sessions.revoke_sync(token_hash=token_hash)
    db_auth_sessions.revoke_sync(token_hash=token_hash)  # must not raise

    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is None


def test_revoke_of_unknown_token_is_a_safe_no_op(postgres_db):
    db_auth_sessions.revoke_sync(token_hash=_token_hash("never-existed"))  # must not raise


def test_only_the_digest_is_persisted_never_the_raw_token(postgres_db):
    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = _token_hash(raw_token)
    db_auth_sessions.create_sync(
        token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=_aware_utcnow() + timedelta(hours=1)
    )

    with Session(get_sync_engine()) as session:
        row = session.execute(
            select(WebSession.session_token_hash).where(WebSession.user_id == user_id)
        ).scalar_one()

    assert row == token_hash
    assert row != raw_token.encode("utf-8")
    assert raw_token.encode("utf-8") not in row


def test_web_sessions_user_id_has_a_foreign_key_to_users(postgres_db):
    """Regression proof for "canonical user FK/integrity": a session row
    can never point at a nonexistent user."""
    token_hash = _token_hash(secrets.token_urlsafe(32))
    with pytest.raises(IntegrityError):
        db_auth_sessions.create_sync(
            token_hash=token_hash, user_id=uuid.uuid4(), issued_secure=True,
            expires_at=_aware_utcnow() + timedelta(hours=1),
        )


def test_two_users_sessions_never_cross_resolve(postgres_db):
    user_a, user_b = _real_user(), _real_user()
    token_a, token_b = _token_hash(secrets.token_urlsafe(32)), _token_hash(secrets.token_urlsafe(32))
    expires = _aware_utcnow() + timedelta(hours=1)

    db_auth_sessions.create_sync(token_hash=token_a, user_id=user_a, issued_secure=True, expires_at=expires)
    db_auth_sessions.create_sync(token_hash=token_b, user_id=user_b, issued_secure=True, expires_at=expires)

    assert db_auth_sessions.get_active_sync(token_hash=token_a, expected_secure=True).user_id == user_a
    assert db_auth_sessions.get_active_sync(token_hash=token_b, expected_secure=True).user_id == user_b
