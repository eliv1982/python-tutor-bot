"""
Stage 6C regression tests: the durable OAuth generation/tombstone protocol
(Stage 6C corrective pass, independent-audit MAJOR 1, Section F) — every
required race proven through the REAL FastAPI callback path
(web.app.create_app() + Starlette TestClient), with ONLY the outbound
GitHub provider HTTP mocked (services/github_oauth_client.py's `_client()`
seam, the same boundary tests/test_stage6b_github_oauth_routes.py already
mocks at) — never by mocking app.github_identity.resolve_user_uuid_for_oauth()
itself to fabricate an outcome. Real disposable PostgreSQL via
tests/conftest.py's postgres_db; database generation/tombstone values are
read directly and asserted on throughout, not merely inferred from HTTP
status codes.

Section F's five required races, in order:
  A. OAuth started BEFORE unlink — paused after verified provider auth,
     before identity resolution; unlink completes; callback resumes; stale
     generation rejected; no user/mapping/session recreated.
  B. OAuth started AFTER unlink — captures the new generation; allowed; a
     fresh GitHub-only mapping/session is created normally.
  C. An OLD callback after a NEWER flow already recreated a mapping — the
     old generation remains rejected even though a mapping now exists.
  D. Callback resolved before unlink but has not yet issued a session —
     the existing (pre-Stage-6C) race-safe create_for_github_sync()
     behavior remains effective.
  E. Two successive unlink/relink cycles — generation increases
     monotonically; a callback from either earlier generation cannot
     restore access.
"""

import random
import threading
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

import app.auth_session as auth_session
import app.github_identity as github_identity
import db.auth_sessions as db_auth_sessions
import db.github_identity as db_github_identity
import db.telegram_link as db_telegram_link
import services.github_oauth_client as github_oauth_client
import web_config
from concurrency_helpers import capture
from db.engine import get_sync_engine
from db.models import GITHUB_OAUTH_ADMISSION_ID, GithubOAuthAdmission, GithubUnlinkTombstone
from web.app import create_app

LOGIN_PATH = "/api/auth/github/login"
CALLBACK_PATH = "/api/auth/github/callback"


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


@pytest.fixture(autouse=True)
def _insecure_posture_for_testing(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    yield


def _install_mock_github(monkeypatch, *, github_id: int) -> None:
    """Mocks ONLY the outbound GitHub HTTP calls (token exchange + /user) —
    identity resolution/generation checking below always runs for real."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gho_faketoken123", "token_type": "bearer"})
        if request.url.path == "/user":
            return httpx.Response(200, json={"id": github_id, "login": "octocat"})
        raise AssertionError(f"unexpected outbound GitHub request: {request.url}")

    def _client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(github_oauth_client, "_client", _client)


def _do_login(client: TestClient) -> str:
    response = client.get(LOGIN_PATH, follow_redirects=False)
    assert response.status_code == 302
    return parse_qs(urlparse(response.headers["location"]).query)["state"][0]


def _admission_generation() -> int:
    with Session(get_sync_engine()) as session:
        return session.execute(
            select(GithubOAuthAdmission.unlink_generation).where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
        ).scalar_one()


def _tombstone_generation(github_id: int):
    with Session(get_sync_engine()) as session:
        return session.execute(
            select(GithubUnlinkTombstone.unlink_generation).where(GithubUnlinkTombstone.github_user_id == github_id)
        ).scalar_one_or_none()


def _fresh_github_id() -> int:
    return random.randint(10 ** 8, 10 ** 9 - 1)


# ---------------------------------------------------------------------------
# A. OAuth started BEFORE unlink
# ---------------------------------------------------------------------------


def test_oauth_started_before_unlink_is_rejected_once_unlink_completes(monkeypatch, postgres_db):
    github_id = _fresh_github_id()
    user_id = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    _install_mock_github(monkeypatch, github_id=github_id)

    assert _admission_generation() == 0
    client = TestClient(create_app())
    state = _do_login(client)  # captures auth_generation=0

    real_resolver = github_identity.resolve_user_uuid_for_oauth
    paused = threading.Event()
    release = threading.Event()

    async def _paused_resolver(*, github_user_id, auth_generation):
        # "Paused after verified provider authentication but before
        # identity resolution" — this seam runs strictly after the
        # callback's token exchange + /user call have already succeeded.
        paused.set()
        assert release.wait(timeout=5), "test never released the paused callback"
        return await real_resolver(github_user_id=github_user_id, auth_generation=auth_generation)

    monkeypatch.setattr(github_identity, "resolve_user_uuid_for_oauth", _paused_resolver)

    callback_outcome = {}

    def _run_callback():
        callback_outcome["record"] = capture(lambda: client.get(
            CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False
        ))

    callback_thread = threading.Thread(target=_run_callback)
    callback_thread.start()
    assert paused.wait(timeout=5), "callback never reached identity resolution"

    unlink_outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert unlink_outcome == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert _admission_generation() == 1
    assert _tombstone_generation(github_id) == 1

    release.set()
    callback_thread.join(timeout=5)
    assert not callback_thread.is_alive()

    assert callback_outcome["record"].exception is None
    response = callback_outcome["record"].result
    assert response.status_code == 400
    assert client.cookies.get(web_config.session_cookie_name()) is None
    # No user, no mapping, no session was (re)created for this GitHub id.
    assert db_github_identity.lookup_user_by_github_id_sync(github_id) is None


# ---------------------------------------------------------------------------
# B. OAuth started AFTER unlink
# ---------------------------------------------------------------------------


def test_oauth_started_after_unlink_creates_a_fresh_mapping_normally(monkeypatch, postgres_db):
    github_id = _fresh_github_id()
    old_user_id = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    _install_mock_github(monkeypatch, github_id=github_id)

    assert db_telegram_link.unlink_github_sync(user_id=old_user_id) == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert _admission_generation() == 1

    client = TestClient(create_app())
    state = _do_login(client)  # captures auth_generation=1 (current)
    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)

    assert response.status_code == 302
    assert client.cookies.get(web_config.session_cookie_name()) is not None

    new_user_id = db_github_identity.lookup_user_by_github_id_sync(github_id)
    assert new_user_id is not None
    assert new_user_id != old_user_id


# ---------------------------------------------------------------------------
# C. Old callback after a newer flow already recreated a mapping
# ---------------------------------------------------------------------------


def test_old_generation_callback_cannot_attach_to_a_newer_recreated_mapping(monkeypatch, postgres_db):
    github_id = _fresh_github_id()
    old_user_id = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    _install_mock_github(monkeypatch, github_id=github_id)

    old_client = TestClient(create_app())
    old_state = _do_login(old_client)  # captures auth_generation=0

    assert db_telegram_link.unlink_github_sync(user_id=old_user_id) == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert _admission_generation() == 1

    new_client = TestClient(create_app())
    new_state = _do_login(new_client)  # captures auth_generation=1 (current)
    new_response = new_client.get(CALLBACK_PATH, params={"code": "c2", "state": new_state}, follow_redirects=False)
    assert new_response.status_code == 302
    new_user_id = db_github_identity.lookup_user_by_github_id_sync(github_id)
    assert new_user_id is not None and new_user_id != old_user_id

    # The OLD callback's own transaction (still unconsumed, captured at
    # generation 0) must be rejected — EVEN THOUGH a mapping for this
    # GitHub id now genuinely exists (pointing at new_user_id).
    old_response = old_client.get(CALLBACK_PATH, params={"code": "c1", "state": old_state}, follow_redirects=False)
    assert old_response.status_code == 400
    assert old_client.cookies.get(web_config.session_cookie_name()) is None

    # And the new mapping/session are unaffected by the old callback.
    me_as_new = new_client.get("/api/me")
    assert me_as_new.status_code == 200
    assert me_as_new.json()["id"] == str(new_user_id)


# ---------------------------------------------------------------------------
# D. Callback resolved before unlink but has not yet issued a session
# ---------------------------------------------------------------------------


def test_resolved_before_unlink_but_not_yet_issued_session_fails_closed(monkeypatch, postgres_db):
    """Proves the EXISTING (pre-Stage-6C) race-safe create_for_github_sync()
    protection remains effective now that identity resolution is
    generation-gated: the resolver already succeeded (this GitHub id was
    not stale), but before create_session_for_github() locks/re-resolves
    the mapping, a concurrent unlink deletes it — session issuance must
    fail closed, never minting a session for a mapping that no longer
    exists."""
    github_id = _fresh_github_id()
    user_id = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    _install_mock_github(monkeypatch, github_id=github_id)

    client = TestClient(create_app())
    state = _do_login(client)

    real_create_session_for_github = auth_session.create_session_for_github
    paused = threading.Event()
    release = threading.Event()

    async def _paused_create_session(gh_id, *, issued_secure):
        paused.set()
        assert release.wait(timeout=5), "test never released the paused callback"
        return await real_create_session_for_github(gh_id, issued_secure=issued_secure)

    monkeypatch.setattr(auth_session, "create_session_for_github", _paused_create_session)

    callback_outcome = {}

    def _run_callback():
        callback_outcome["record"] = capture(lambda: client.get(
            CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False
        ))

    callback_thread = threading.Thread(target=_run_callback)
    callback_thread.start()
    assert paused.wait(timeout=5), "callback never reached session issuance"

    assert db_telegram_link.unlink_github_sync(user_id=user_id) == db_telegram_link.UnlinkOutcome.USER_DELETED

    release.set()
    callback_thread.join(timeout=5)
    assert not callback_thread.is_alive()

    assert callback_outcome["record"].exception is None
    response = callback_outcome["record"].result
    assert response.status_code == 503
    assert client.cookies.get(web_config.session_cookie_name()) is None


# ---------------------------------------------------------------------------
# E. Two successive unlink/relink cycles — monotonic generation
# ---------------------------------------------------------------------------


def test_two_successive_unlink_relink_cycles_generation_increases_monotonically(monkeypatch, postgres_db):
    github_id = _fresh_github_id()
    user0 = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    _install_mock_github(monkeypatch, github_id=github_id)

    client0 = TestClient(create_app())
    state0 = _do_login(client0)  # auth_generation=0

    assert db_telegram_link.unlink_github_sync(user_id=user0) == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert _admission_generation() == 1
    assert _tombstone_generation(github_id) == 1

    # Generation 0 is now stale relative to tombstone@1.
    response0 = client0.get(CALLBACK_PATH, params={"code": "c0", "state": state0}, follow_redirects=False)
    assert response0.status_code == 400
    assert db_github_identity.lookup_user_by_github_id_sync(github_id) is None

    client1 = TestClient(create_app())
    state1 = _do_login(client1)  # auth_generation=1 (current)

    # A SECOND login also captures generation 1 here, BEFORE the second
    # unlink below — used to prove that a generation which was valid a
    # moment ago can become stale once a LATER unlink advances the
    # counter again (the check always compares against the LATEST
    # tombstone value, never a cached/earlier one).
    client1b = TestClient(create_app())
    state1b = _do_login(client1b)  # also auth_generation=1

    response1 = client1.get(CALLBACK_PATH, params={"code": "c1", "state": state1}, follow_redirects=False)
    assert response1.status_code == 302
    user1 = db_github_identity.lookup_user_by_github_id_sync(github_id)
    assert user1 is not None and user1 != user0

    assert db_telegram_link.unlink_github_sync(user_id=user1) == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert _admission_generation() == 2
    assert _tombstone_generation(github_id) == 2

    response1b = client1b.get(CALLBACK_PATH, params={"code": "c1b", "state": state1b}, follow_redirects=False)
    assert response1b.status_code == 400
    assert db_github_identity.lookup_user_by_github_id_sync(github_id) is None
