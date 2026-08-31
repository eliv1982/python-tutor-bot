"""
Stage 6A regression tests: the application-layer session lifecycle
(app.auth_session — create/resolve/revoke, user profile lookup) against a
REAL disposable PostgreSQL container. See tests/conftest.py's
postgres_container()/postgres_db() fixtures.
"""

import random
import uuid

import pytest

import app.auth_session as auth_session
import db.identity as db_identity


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    needs a REAL `users` row, never the offline in-memory fake."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


@pytest.mark.asyncio
async def test_create_session_issues_a_high_entropy_opaque_token(postgres_db):
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=True)

    assert isinstance(issued.raw_token, str)
    # secrets.token_urlsafe(32) -> 43 base64url characters (256 bits).
    assert len(issued.raw_token) >= 40
    assert str(user_id) not in issued.raw_token


@pytest.mark.asyncio
async def test_created_session_resolves_to_the_correct_user(postgres_db):
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=True)

    resolved = await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=True)

    assert resolved == user_id


@pytest.mark.asyncio
async def test_resolve_rejects_none_and_empty_token(postgres_db):
    assert await auth_session.resolve_session_user_id(None, expected_secure=True) is None
    assert await auth_session.resolve_session_user_id("", expected_secure=True) is None


@pytest.mark.asyncio
async def test_resolve_rejects_an_unknown_garbage_token(postgres_db):
    assert await auth_session.resolve_session_user_id("not-a-real-session-token", expected_secure=True) is None


@pytest.mark.asyncio
async def test_resolve_rejects_a_canonical_user_uuid_used_as_a_session_token(postgres_db):
    """Regression proof: canonical user UUIDs must never work as session
    identifiers, even if an attacker (or a bug) supplies a real, currently
    active user's own UUID directly as if it were a bearer token."""
    real_user_id = _real_user()
    await auth_session.create_session(real_user_id, issued_secure=True)

    assert await auth_session.resolve_session_user_id(str(real_user_id), expected_secure=True) is None


@pytest.mark.asyncio
async def test_revoke_session_invalidates_it(postgres_db):
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=True)

    await auth_session.revoke_session(issued.raw_token)

    assert await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=True) is None


@pytest.mark.asyncio
async def test_revoke_session_of_none_or_unknown_token_does_not_raise(postgres_db):
    await auth_session.revoke_session(None)
    await auth_session.revoke_session("some-token-that-was-never-issued")


@pytest.mark.asyncio
async def test_two_users_sessions_are_fully_isolated(postgres_db):
    user_a, user_b = _real_user(), _real_user()
    session_a = await auth_session.create_session(user_a, issued_secure=True)
    session_b = await auth_session.create_session(user_b, issued_secure=True)

    assert await auth_session.resolve_session_user_id(session_a.raw_token, expected_secure=True) == user_a
    assert await auth_session.resolve_session_user_id(session_b.raw_token, expected_secure=True) == user_b

    # Revoking A's session must never affect B's.
    await auth_session.revoke_session(session_a.raw_token)
    assert await auth_session.resolve_session_user_id(session_a.raw_token, expected_secure=True) is None
    assert await auth_session.resolve_session_user_id(session_b.raw_token, expected_secure=True) == user_b


@pytest.mark.asyncio
async def test_get_user_profile_returns_only_safe_fields(postgres_db):
    user_id = _real_user()
    profile = await auth_session.get_user_profile(user_id)

    assert profile is not None
    assert profile.id == user_id
    assert profile.created_at is not None
    assert set(vars(profile).keys()) == {"id", "created_at"}


@pytest.mark.asyncio
async def test_get_user_profile_of_unknown_user_returns_none(postgres_db):
    assert await auth_session.get_user_profile(uuid.uuid4()) is None
