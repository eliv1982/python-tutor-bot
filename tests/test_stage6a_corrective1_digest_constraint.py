"""
Stage 6A independent-audit corrective pass #1 — Major 2 real-PostgreSQL
regression proof: `web_sessions.session_token_hash` must be enforced to be
EXACTLY 32 bytes by the database itself, not merely by SQLAlchemy's
`LargeBinary(32)` Python-side type (which compiles to an unconstrained
`BYTEA` on PostgreSQL — the length argument has no server-side effect on
that dialect). The auditor proved this by inserting a 1-byte digest
directly and having PostgreSQL accept it.

Fix: an explicit `octet_length(session_token_hash) = 32` CHECK constraint
(`ck_web_sessions_session_token_hash_length`), added to both db/models.py
and alembic/versions/0002_web_sessions.py.

Every INSERT here goes through a raw SQLAlchemy Core `insert()` — NOT
db.auth_sessions.create_sync() — deliberately, since that function's own
Python-level type hint (`token_hash: bytes`) offers no way to construct a
call with a wrong-length digest that would even reach the database; this
module needs to prove the DATABASE's own enforcement, independent of any
Python-side care taken by the calling code above it.
"""

import random
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

import db.identity as db_identity
from db.engine import get_sync_engine
from db.models import WebSession


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — needs a REAL
    `users` row to satisfy web_sessions' FK constraint."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _future_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _try_insert(*, digest: bytes, user_id: uuid.UUID) -> None:
    engine = get_sync_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(WebSession).values(
                session_token_hash=digest,
                user_id=user_id,
                issued_secure=True,
                expires_at=_future_expiry(),
            )
        )


def test_exactly_32_byte_digest_is_accepted(postgres_db):
    user_id = _real_user()
    digest = b"\x00" * 32
    _try_insert(digest=digest, user_id=user_id)

    engine = get_sync_engine()
    with engine.begin() as conn:
        row = conn.execute(
            select(WebSession.session_token_hash).where(WebSession.user_id == user_id)
        ).scalar_one()
    assert row == digest


def test_31_byte_digest_is_rejected_by_postgresql(postgres_db):
    user_id = _real_user()
    with pytest.raises(IntegrityError):
        _try_insert(digest=b"\x00" * 31, user_id=user_id)


def test_33_byte_digest_is_rejected_by_postgresql(postgres_db):
    user_id = _real_user()
    with pytest.raises(IntegrityError):
        _try_insert(digest=b"\x00" * 33, user_id=user_id)


def test_1_byte_digest_is_rejected_by_postgresql(postgres_db):
    """The auditor's exact reproduction."""
    user_id = _real_user()
    with pytest.raises(IntegrityError):
        _try_insert(digest=b"\x00", user_id=user_id)


def test_empty_digest_is_rejected_by_postgresql(postgres_db):
    user_id = _real_user()
    with pytest.raises(IntegrityError):
        _try_insert(digest=b"", user_id=user_id)


def test_64_byte_digest_is_rejected_by_postgresql(postgres_db):
    """A plausible-looking-but-wrong length (e.g. a SHA-512 digest by
    mistake) must be rejected just as surely as an obviously-wrong one."""
    user_id = _real_user()
    with pytest.raises(IntegrityError):
        _try_insert(digest=b"\x00" * 64, user_id=user_id)
