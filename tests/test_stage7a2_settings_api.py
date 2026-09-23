"""
Stage 7A-2 regression tests: authenticated GET/PATCH /api/settings, end to
end through the real FastAPI app, real session authentication, real CSRF,
and real db.preferences against a disposable PostgreSQL container (see
tests/conftest.py's postgres_container()/postgres_db()).

Never contacts any provider, Qdrant, or Telegram.
"""

import random
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
import db.preferences as db_preferences
import session_config
import web_config
from config import BotMode
from db.engine import get_sync_engine
from db.models import UserPreference
from secrecy_helpers import assert_no_secret_leak
from web.app import create_app
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME

INVALID = {"detail": "Invalid request"}
SENTINEL = "SENTINEL-7a2-settings-input-5e0c"


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's autouse fake — real rows only."""
    yield


@pytest.fixture(autouse=True)
def _real_db(postgres_db, monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)


def _real_user() -> uuid.UUID:
    return db_identity.resolve_or_create_user_by_telegram_id_sync(random.randint(10 ** 11, 10 ** 12 - 1))


async def _session_for(user_id: uuid.UUID) -> str:
    return (await auth_session.create_session(user_id, issued_secure=False)).raw_token


def _client(raw_token: str) -> TestClient:
    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), raw_token)
    return client


def _csrf(raw_token: str) -> dict:
    return {CSRF_HEADER_NAME: derive_csrf_token(raw_token)}


def _preference_row_count() -> int:
    with Session(get_sync_engine()) as session:
        return session.execute(select(func.count()).select_from(UserPreference)).scalar_one()


# ============================================================================
# A. Authentication.
# ============================================================================


def test_unauthenticated_get_is_401():
    response = TestClient(create_app()).get("/api/settings")
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_unauthenticated_patch_is_401_and_writes_nothing():
    response = TestClient(create_app()).patch(
        "/api/settings", json={"mode": BotMode.RAG}, headers={CSRF_HEADER_NAME: "irrelevant"}
    )
    assert response.status_code == 401
    assert _preference_row_count() == 0


async def test_expired_revoked_and_malformed_sessions_are_401(monkeypatch):
    user_id = _real_user()
    revoked = await _session_for(user_id)
    await auth_session.revoke_session(revoked)
    monkeypatch.setattr(session_config, "SESSION_TTL_SECONDS", -10)
    expired = await _session_for(user_id)

    for token in (revoked, expired, "not-a-real-session-token", str(user_id)):
        client = _client(token)
        assert client.get("/api/settings").status_code == 401
        assert client.patch("/api/settings", json={"mode": BotMode.RAG}, headers=_csrf(token)).status_code == 401
    assert _preference_row_count() == 0


# ============================================================================
# B. GET.
# ============================================================================


async def test_get_without_row_returns_default_and_creates_no_row():
    token = await _session_for(_real_user())
    response = _client(token).get("/api/settings")
    assert response.status_code == 200
    assert response.json() == {"mode": BotMode.TEXT}
    assert _preference_row_count() == 0


async def test_get_does_not_require_csrf():
    user_id = _real_user()
    db_preferences.set_mode_sync(user_id, BotMode.RAG)
    response = _client(await _session_for(user_id)).get("/api/settings")
    assert response.status_code == 200
    assert response.json() == {"mode": BotMode.RAG}


async def test_get_with_legacy_stored_mode_returns_default_not_500():
    user_id = _real_user()
    db_preferences.set_mode_sync(user_id, "legacy-chat")
    response = _client(await _session_for(user_id)).get("/api/settings")
    assert response.status_code == 200
    assert response.json() == {"mode": BotMode.TEXT}
    assert db_preferences.get_preferences_sync(user_id) == ("legacy-chat", None)


# ============================================================================
# C. PATCH.
# ============================================================================


@pytest.mark.parametrize("mode", BotMode.ALL)
async def test_patch_valid_mode_persists_and_is_read_back(mode):
    user_id = _real_user()
    token = await _session_for(user_id)
    client = _client(token)

    response = client.patch("/api/settings", json={"mode": mode}, headers=_csrf(token))
    assert response.status_code == 200
    assert response.json() == {"mode": mode}
    assert db_preferences.get_preferences_sync(user_id) == (mode, None)
    assert client.get("/api/settings").json() == {"mode": mode}


async def test_patch_preserves_existing_voice():
    user_id = _real_user()
    db_preferences.set_voice_sync(user_id, "nova")
    token = await _session_for(user_id)
    assert _client(token).patch("/api/settings", json={"mode": BotMode.RAG}, headers=_csrf(token)).status_code == 200
    assert db_preferences.get_preferences_sync(user_id) == (BotMode.RAG, "nova")


_INVALID_BODIES = {
    "unknown-mode": {"mode": SENTINEL},
    "empty-mode": {"mode": ""},
    "wrong-case": {"mode": "TEXT"},
    "padded": {"mode": " rag"},
    "empty-object": {},
    "explicit-null": {"mode": None},
    "non-string": {"mode": 1},
    "list": {"mode": ["text"]},
    "unknown-field": {"mode": BotMode.RAG, "voice": "nova"},
    "client-user-id": {"mode": BotMode.RAG, "user_id": str(uuid.uuid4())},
    "top-level-list": [{"mode": BotMode.RAG}],
}


@pytest.mark.parametrize("body", list(_INVALID_BODIES.values()), ids=list(_INVALID_BODIES))
async def test_invalid_patch_is_sanitized_422_and_writes_nothing(body, caplog):
    user_id = _real_user()
    token = await _session_for(user_id)
    response = _client(token).patch("/api/settings", json=body, headers=_csrf(token))
    assert response.status_code == 422
    assert response.json() == INVALID
    assert_no_secret_leak(SENTINEL, response.text, caplog=caplog)
    assert _preference_row_count() == 0


async def test_malformed_json_patch_is_422_and_writes_nothing():
    token = await _session_for(_real_user())
    response = _client(token).patch(
        "/api/settings", content=b'{"mode": ', headers={**_csrf(token), "content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json() == INVALID
    assert _preference_row_count() == 0


async def test_invalid_patch_does_not_change_existing_row():
    user_id = _real_user()
    db_preferences.set_mode_sync(user_id, BotMode.VISION)
    token = await _session_for(user_id)
    response = _client(token).patch("/api/settings", json={"mode": "bogus"}, headers=_csrf(token))
    assert response.status_code == 422
    assert db_preferences.get_preferences_sync(user_id) == (BotMode.VISION, None)


# ============================================================================
# D. CSRF.
# ============================================================================


async def test_patch_without_valid_csrf_is_403_and_writes_nothing():
    user_id = _real_user()
    token = await _session_for(user_id)
    other_token = await _session_for(_real_user())
    client = _client(token)

    for headers in ({}, {CSRF_HEADER_NAME: "wrong"}, _csrf(other_token)):
        response = client.patch("/api/settings", json={"mode": BotMode.RAG}, headers=headers)
        assert response.status_code == 403
        assert response.json() == {"detail": "CSRF validation failed"}
    assert _preference_row_count() == 0


# ============================================================================
# E. Ownership.
# ============================================================================


async def test_users_only_ever_read_and_write_their_own_preference():
    user_a, user_b = _real_user(), _real_user()
    token_a, token_b = await _session_for(user_a), await _session_for(user_b)
    db_preferences.set_mode_sync(user_b, BotMode.VISION)

    client_a = _client(token_a)
    # A cannot target B by supplying B's identity — no such field exists.
    rejected = client_a.patch(
        "/api/settings", json={"mode": BotMode.RAG, "user_id": str(user_b)}, headers=_csrf(token_a)
    )
    assert rejected.status_code == 422

    assert client_a.patch("/api/settings", json={"mode": BotMode.RAG}, headers=_csrf(token_a)).status_code == 200
    # A's GET reflects only A; B's row is untouched and B still reads its own.
    assert client_a.get("/api/settings").json() == {"mode": BotMode.RAG}
    assert db_preferences.get_preferences_sync(user_b) == (BotMode.VISION, None)
    assert _client(token_b).get("/api/settings").json() == {"mode": BotMode.VISION}
    assert db_preferences.get_preferences_sync(user_a) == (BotMode.RAG, None)
