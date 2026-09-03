"""
Stage 6A regression tests: the FastAPI web adapter end to end (web.app,
web.routes, web.dependencies) via Starlette's TestClient, against a REAL
disposable PostgreSQL container for every session-backed assertion. See
tests/conftest.py's postgres_container()/postgres_db() fixtures.

Never contacts GitHub, OpenAI, Anthropic, Telegram, or Qdrant Cloud —
pytest.ini's --disable-socket already enforces this at the socket layer;
these tests never construct any provider client at all.
"""

import random
import uuid

import pytest
from starlette.testclient import TestClient

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
import session_config
import web_config
from web.app import create_app
from web.dependencies import CSRF_HEADER_NAME


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    needs REAL `users` rows and REAL sessions, never the offline in-memory
    fake."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def test_app_can_be_constructed_offline():
    """No database/network access happens merely by building the app —
    the deliberately-unreachable poisoned DATABASE_URL (tests/conftest.py)
    is still in effect here and this must not touch it."""
    app = create_app()
    assert app is not None


def test_healthz_returns_ok_without_authentication():
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_me_without_any_cookie_is_unauthenticated():
    client = TestClient(create_app())
    response = client.get("/api/me")
    assert response.status_code == 401


def test_me_with_a_malformed_cookie_is_unauthenticated(monkeypatch, postgres_db):
    """A non-empty but never-issued token still reaches the real database
    (there is no shortcut that pattern-matches "plausible" tokens — see
    app/auth_session.resolve_session_user_id()), so this needs a real,
    reachable PostgreSQL like every other non-empty-token scenario."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), "not-a-real-session-token")
    response = client.get("/api/me")
    assert response.status_code == 401


def test_me_with_a_canonical_user_uuid_as_the_cookie_is_unauthenticated(monkeypatch, postgres_db):
    """A canonical user UUID must never work as a session id, even when
    supplied directly as the cookie value."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    real_user_id = _real_user()
    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), str(real_user_id))
    response = client.get("/api/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_with_a_valid_session_returns_only_safe_fields(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    response = client.get("/api/me")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user_id)
    # Stage 6C, Section M added a safe `telegram_linked: bool` field — this
    # module's _real_user() helper resolves via Telegram, so it must be
    # True here, and never the Telegram numeric id or any other new field.
    assert set(body.keys()) == {"id", "created_at", "telegram_linked"}
    assert body["telegram_linked"] is True


@pytest.mark.asyncio
async def test_me_with_an_expired_session_is_unauthenticated(monkeypatch, postgres_db):
    """SESSION_TTL_SECONDS now lives in session_config.py, not web_config.py
    (independent-audit corrective pass #1, minor finding #1 — see
    app/auth_session.py's own docstring on the import-isolation
    rationale)."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    monkeypatch.setattr(session_config, "SESSION_TTL_SECONDS", -10)  # already expired at creation
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    response = client.get("/api/me")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_with_a_revoked_session_is_unauthenticated(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)
    await auth_session.revoke_session(issued.raw_token)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    response = client.get("/api/me")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_session_belonging_to_user_a_never_authenticates_as_user_b(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_a, user_b = _real_user(), _real_user()
    issued_a = await auth_session.create_session(user_a, issued_secure=False)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued_a.raw_token)
    response = client.get("/api/me")

    assert response.status_code == 200
    assert response.json()["id"] == str(user_a)
    assert response.json()["id"] != str(user_b)


@pytest.mark.asyncio
async def test_logout_without_csrf_header_is_rejected(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    response = client.post("/api/logout")

    assert response.status_code == 403
    # Rejected CSRF must not have revoked the session.
    assert await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=False) == user_id


def test_logout_without_any_session_is_unauthenticated():
    client = TestClient(create_app())
    response = client.post("/api/logout", headers={CSRF_HEADER_NAME: "irrelevant"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_logout_with_a_valid_csrf_proof_revokes_the_server_side_session(monkeypatch, postgres_db):
    from web.csrf import derive_csrf_token

    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    response = client.post(
        "/api/logout", headers={CSRF_HEADER_NAME: derive_csrf_token(issued.raw_token)}
    )

    assert response.status_code == 204
    # Server-side state was actually revoked, not merely the browser
    # cookie — the same raw token must no longer resolve at all.
    assert await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=False) is None

    # And the now-revoked session can no longer reach the protected route.
    followup = client.get("/api/me")
    assert followup.status_code == 401


@pytest.mark.asyncio
async def test_csrf_token_from_one_session_does_not_authorize_another(monkeypatch, postgres_db):
    from web.csrf import derive_csrf_token

    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_a, user_b = _real_user(), _real_user()
    issued_a = await auth_session.create_session(user_a, issued_secure=False)
    issued_b = await auth_session.create_session(user_b, issued_secure=False)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued_b.raw_token)
    # Forged: valid session cookie for B, but a CSRF token derived from A's
    # (different) raw token.
    response = client.post(
        "/api/logout", headers={CSRF_HEADER_NAME: derive_csrf_token(issued_a.raw_token)}
    )

    assert response.status_code == 403
    assert await auth_session.resolve_session_user_id(issued_b.raw_token, expected_secure=False) == user_b


@pytest.mark.asyncio
async def test_unhandled_exception_never_leaks_internals_to_the_client(monkeypatch, postgres_db):
    """create_app() never sets debug=True — Starlette's default
    ServerErrorMiddleware then returns a fixed generic body for any
    unhandled exception, never the exception's own message/traceback. This
    proves it end to end against our actual app, not just bare FastAPI()."""
    sensitive = "sensitive internal detail: " + web_config.SESSION_SECRET_KEY

    async def _boom(user_id):
        raise RuntimeError(sensitive)

    monkeypatch.setattr(auth_session, "get_user_profile", _boom)

    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)

    client = TestClient(create_app(), raise_server_exceptions=False)
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    response = client.get("/api/me")

    assert response.status_code == 500
    assert sensitive not in response.text
    assert web_config.SESSION_SECRET_KEY not in response.text


@pytest.mark.asyncio
async def test_get_requests_never_mutate_session_state(monkeypatch, postgres_db):
    """No sliding expiration: repeated GETs against a protected route must
    not extend/alter the session's expires_at."""
    import hashlib

    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)
    token_hash = hashlib.sha256(issued.raw_token.encode("utf-8")).digest()
    before = db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False)

    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    client.get("/api/me")
    client.get("/api/me")
    client.get("/healthz")

    after = db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False)
    assert before.expires_at == after.expires_at
