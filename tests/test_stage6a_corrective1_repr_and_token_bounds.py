"""
Stage 6A independent-audit corrective pass #1 — Major 1 (raw bearer token
must never appear in repr/debug output) and the "required hardening"
session-token input-bounds regression tests.

The token-shape tests deliberately do NOT need postgres_db for the
"malformed input never reaches the database" half — db.auth_sessions.
get_active_sync/revoke_sync are monkeypatched to a sentinel that fails the
test if called at all, proving app.auth_session's shape check short-
circuits BEFORE any DB access, independent of whether a database is even
reachable. The "a legitimately generated token still works" tests use
postgres_db for a genuine end-to-end proof.
"""

import random
import secrets
import uuid

import pytest

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.identity as db_identity


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — some tests below
    need a REAL `users` row (postgres_db), and none of them want the
    offline in-memory identity fake shadowing db.identity in a way that
    would mask what's actually under test."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _fail_if_called(*args, **kwargs):
    raise AssertionError("db.auth_sessions was reached for a malformed/impossible token shape")


# --- Major 1: raw bearer token must never appear in repr() -----------------


def test_issued_session_repr_never_contains_the_raw_token():
    issued = auth_session.IssuedSession(raw_token="super-secret-raw-bearer-token-value", expires_at=None)
    rendered = repr(issued)
    assert "super-secret-raw-bearer-token-value" not in rendered
    assert "raw_token" not in rendered  # field(repr=False) omits the field entirely


def test_issued_session_repr_still_shows_expires_at():
    """The fix must hide ONLY the secret field, not turn the whole repr
    useless for debugging non-sensitive fields."""
    import datetime
    expires_at = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    issued = auth_session.IssuedSession(raw_token="irrelevant", expires_at=expires_at)
    assert "expires_at" in repr(issued)
    assert "2026" in repr(issued)


def test_issued_session_str_also_never_contains_the_raw_token():
    """str() on a plain dataclass falls back to __repr__ — confirm the
    same guarantee holds through that path too."""
    issued = auth_session.IssuedSession(raw_token="another-secret-value", expires_at=None)
    assert "another-secret-value" not in str(issued)


def test_no_other_new_dataclass_leaks_a_secret_in_repr():
    """Sweep of every OTHER Stage 6A dataclass that could plausibly carry
    sensitive material — none of them do (UserProfile/SessionRecord only
    ever carry a canonical user id and plain timestamps), confirmed here so
    this stays true if a field is ever added carelessly."""
    profile = auth_session.UserProfile(id=uuid.uuid4(), created_at=None)
    assert "raw_token" not in repr(profile) and "token_hash" not in repr(profile)

    record = db_auth_sessions.SessionRecord(user_id=uuid.uuid4(), created_at=None, expires_at=None)
    rendered = repr(record)
    assert "raw_token" not in rendered and "token_hash" not in rendered


# --- required hardening: session-token input bounds -------------------------


@pytest.mark.asyncio
async def test_a_genuinely_generated_token_is_accepted(postgres_db):
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=True)
    assert await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=True) == user_id


@pytest.mark.asyncio
async def test_empty_token_is_rejected_without_reaching_the_database(monkeypatch):
    monkeypatch.setattr(db_auth_sessions, "get_active_sync", _fail_if_called)
    assert await auth_session.resolve_session_user_id("", expected_secure=True) is None
    assert await auth_session.resolve_session_user_id(None, expected_secure=True) is None


@pytest.mark.asyncio
async def test_truncated_token_is_rejected_without_reaching_the_database(monkeypatch):
    monkeypatch.setattr(db_auth_sessions, "get_active_sync", _fail_if_called)
    truncated = secrets.token_urlsafe(32)[:20]
    assert await auth_session.resolve_session_user_id(truncated, expected_secure=True) is None


@pytest.mark.asyncio
async def test_oversized_token_is_rejected_without_reaching_the_database(monkeypatch):
    monkeypatch.setattr(db_auth_sessions, "get_active_sync", _fail_if_called)
    # Far larger than any legitimate secrets.token_urlsafe(32) output could
    # ever be (43 chars) — simulates an attacker sending an unbounded
    # cookie value.
    oversized = "a" * 1_000_000
    assert await auth_session.resolve_session_user_id(oversized, expected_secure=True) is None


@pytest.mark.asyncio
async def test_token_with_malformed_characters_is_rejected_without_reaching_the_database(monkeypatch):
    monkeypatch.setattr(db_auth_sessions, "get_active_sync", _fail_if_called)
    # Same length as a real token (43 chars) but with characters outside
    # the base64url alphabet — proves the shape check inspects content, not
    # merely length.
    malformed = "!" * 43
    assert await auth_session.resolve_session_user_id(malformed, expected_secure=True) is None

    padded = secrets.token_urlsafe(32)[:-1] + "="  # base64 padding char, never in token_urlsafe output
    assert await auth_session.resolve_session_user_id(padded, expected_secure=True) is None


@pytest.mark.asyncio
async def test_a_slightly_off_length_real_looking_token_is_rejected_without_reaching_the_database(monkeypatch):
    monkeypatch.setattr(db_auth_sessions, "get_active_sync", _fail_if_called)
    one_char_short = secrets.token_urlsafe(32)[:-1]
    one_char_long = secrets.token_urlsafe(32) + "a"
    assert await auth_session.resolve_session_user_id(one_char_short, expected_secure=True) is None
    assert await auth_session.resolve_session_user_id(one_char_long, expected_secure=True) is None


@pytest.mark.asyncio
async def test_malformed_token_never_becomes_an_authenticated_session(postgres_db):
    """End-to-end (real DB reachable this time, proving the rejection isn't
    merely an artifact of the DB being unreachable): a well-formed-length
    but never-issued/garbage token must resolve to no user, and neither
    must a structurally-impossible one."""
    never_issued = secrets.token_urlsafe(32)
    assert await auth_session.resolve_session_user_id(never_issued, expected_secure=True) is None
    assert await auth_session.resolve_session_user_id("too-short", expected_secure=True) is None
    assert await auth_session.resolve_session_user_id("x" * 500, expected_secure=True) is None


@pytest.mark.asyncio
async def test_revoke_of_a_malformed_token_does_not_reach_the_database(monkeypatch):
    monkeypatch.setattr(db_auth_sessions, "revoke_sync", _fail_if_called)
    await auth_session.revoke_session("!" * 43)
    await auth_session.revoke_session("too-short")
    await auth_session.revoke_session("x" * 1_000_000)
