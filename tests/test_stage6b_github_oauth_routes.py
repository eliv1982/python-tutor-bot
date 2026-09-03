"""
Stage 6B regression tests: the GitHub OAuth login flow end to end —
actual FastAPI app (web.app.create_app()) + Starlette TestClient + a REAL
disposable PostgreSQL container (tests/conftest.py's postgres_db) + GitHub
HTTP mocked at the httpx transport layer (services/github_oauth_client.py's
`_client()` factory, see that module's own docstring) — no real network
call ever happens here; pytest.ini's --disable-socket would fail the run
outright if one somehow did.

Mirrors tests/test_stage6a_fastapi_app.py's own real-Postgres,
insecure-posture-for-testing convention (COOKIE_SECURE=False +
apply_startup_posture_sync(requested_secure=False)) so session creation
never needs a real TLS connection.
"""

import random
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from starlette.testclient import TestClient

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
import services.github_oauth_client as github_oauth_client
import web_config
from web.app import create_app
from web.github_oauth import _oauth_state_cookie_name

LOGIN_PATH = "/api/auth/github/login"
CALLBACK_PATH = "/api/auth/github/callback"


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    needs REAL `users`/`github_accounts`/`github_oauth_transactions`/
    `web_sessions` rows, never the offline in-memory fake."""
    yield


@pytest.fixture(autouse=True)
def _insecure_posture_for_testing(monkeypatch, postgres_db):
    """Every test in this module needs a real Postgres-backed session
    mint — mirrors test_stage6a_fastapi_app.py's own per-test posture
    setup exactly."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    yield


def _real_telegram_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _install_mock_github(monkeypatch, *, github_id=None, token_status=200, user_status=200,
                          token_calls=None, user_calls=None, token_payload=None, user_payload=None):
    """Installs a fake GitHub transport intercepting BOTH the token
    exchange and the /user endpoint, dispatched by URL — see
    services/github_oauth_client.py's `_client()` factory seam."""
    if github_id is None:
        github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    token_calls = token_calls if token_calls is not None else []
    user_calls = user_calls if user_calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            token_calls.append(request)
            payload = token_payload if token_payload is not None else {
                "access_token": "gho_faketoken123", "token_type": "bearer"
            }
            return httpx.Response(token_status, json=payload)
        if request.url.path == "/user":
            user_calls.append(request)
            payload = user_payload if user_payload is not None else {"id": github_id, "login": "octocat"}
            return httpx.Response(user_status, json=payload)
        raise AssertionError(f"unexpected outbound GitHub request: {request.url}")

    def _client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(github_oauth_client, "_client", _client)
    return github_id, token_calls, user_calls


def _extract_state_from_authorize_redirect(location: str) -> str:
    params = parse_qs(urlparse(location).query)
    return params["state"][0]


def _do_login(client: TestClient) -> tuple[str, str]:
    """Performs GET /login, returns (state, authorize_url)."""
    response = client.get(LOGIN_PATH, follow_redirects=False)
    assert response.status_code == 302
    location = response.headers["location"]
    state = _extract_state_from_authorize_redirect(location)
    assert client.cookies.get(_oauth_state_cookie_name(secure=False)) == state
    return state, location


# --- successful flow (Section 24) -------------------------------------------


def test_full_successful_login_flow(monkeypatch):
    github_id, token_calls, user_calls = _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    state, authorize_url = _do_login(client)
    assert authorize_url.startswith("https://github.com/login/oauth/authorize?")

    callback_response = client.get(
        CALLBACK_PATH, params={"code": "test-authorization-code", "state": state}, follow_redirects=False
    )

    assert callback_response.status_code == 302
    assert callback_response.headers["location"] == "/api/me"
    assert len(token_calls) == 1
    assert len(user_calls) == 1

    session_cookie_name = web_config.session_cookie_name()
    assert client.cookies.get(session_cookie_name) is not None
    assert client.cookies.get(web_config.csrf_cookie_name()) is not None

    me_response = client.get("/api/me")
    assert me_response.status_code == 200
    canonical_uuid = me_response.json()["id"]

    from db.github_identity import lookup_user_by_github_id_sync
    assert lookup_user_by_github_id_sync(github_id) == uuid.UUID(canonical_uuid)


def test_login_response_sets_expected_oauth_state_cookie_attributes(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = client.get(LOGIN_PATH, follow_redirects=False)
    set_cookie_headers = response.headers.get_list("set-cookie")
    oauth_cookie_header = next(h for h in set_cookie_headers if h.startswith(_oauth_state_cookie_name(secure=False) + "="))

    assert "HttpOnly" in oauth_cookie_header
    assert "samesite=lax" in oauth_cookie_header.lower()
    assert "secure" not in oauth_cookie_header.lower()  # COOKIE_SECURE=False in this fixture posture


# --- repeat login: same GitHub id -> same canonical uuid, fresh session ----


def test_repeat_login_same_github_id_resolves_same_user_with_a_fresh_session(monkeypatch):
    github_id, _, _ = _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    state1, _ = _do_login(client)
    client.get(CALLBACK_PATH, params={"code": "code-1", "state": state1}, follow_redirects=False)
    first_session_token = client.cookies.get(web_config.session_cookie_name())
    first_me = client.get("/api/me").json()

    _install_mock_github(monkeypatch, github_id=github_id)
    state2, _ = _do_login(client)
    client.get(CALLBACK_PATH, params={"code": "code-2", "state": state2}, follow_redirects=False)
    second_session_token = client.cookies.get(web_config.session_cookie_name())
    second_me = client.get("/api/me").json()

    assert first_me["id"] == second_me["id"]
    assert first_session_token != second_session_token


# --- different GitHub ids -> different canonical users ----------------------


def test_different_github_ids_resolve_to_different_canonical_users(monkeypatch):
    client_a = TestClient(create_app())
    _install_mock_github(monkeypatch, github_id=111222333)
    state_a, _ = _do_login(client_a)
    client_a.get(CALLBACK_PATH, params={"code": "c", "state": state_a}, follow_redirects=False)
    user_a = client_a.get("/api/me").json()["id"]

    client_b = TestClient(create_app())
    _install_mock_github(monkeypatch, github_id=444555666)
    state_b, _ = _do_login(client_b)
    client_b.get(CALLBACK_PATH, params={"code": "c", "state": state_b}, follow_redirects=False)
    user_b = client_b.get("/api/me").json()["id"]

    assert user_a != user_b


# --- bad / missing state ----------------------------------------------------


def test_callback_with_unknown_state_fails_and_never_calls_github(monkeypatch):
    _, token_calls, user_calls = _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = client.get(
        CALLBACK_PATH, params={"code": "irrelevant", "state": "s" * 43}, follow_redirects=False
    )

    assert response.status_code == 400
    assert token_calls == []
    assert user_calls == []
    assert client.cookies.get(web_config.session_cookie_name()) is None


def test_callback_with_missing_state_fails(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = client.get(CALLBACK_PATH, params={"code": "irrelevant"}, follow_redirects=False)

    assert response.status_code == 400
    assert client.cookies.get(web_config.session_cookie_name()) is None


def test_callback_with_missing_code_fails(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    state, _ = _do_login(client)
    response = client.get(CALLBACK_PATH, params={"state": state}, follow_redirects=False)

    assert response.status_code == 400
    assert client.cookies.get(web_config.session_cookie_name()) is None


# --- replay: state can be redeemed at most once (Section 6/24) -------------


def test_replayed_state_succeeds_once_then_fails(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    state, _ = _do_login(client)
    first = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert first.status_code == 302

    first_session_token = client.cookies.get(web_config.session_cookie_name())

    second = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert second.status_code == 400

    # The session from the FIRST (legitimate) callback remains valid —
    # the replay attempt must not have disturbed it.
    assert client.cookies.get(web_config.session_cookie_name()) == first_session_token


def test_concurrent_double_callback_only_one_succeeds(monkeypatch):
    """Section 6: two concurrent callbacks for the same state must not
    both proceed through successful authentication. Uses real threads
    against the real database claim (services/github_oauth_client is
    mocked identically for both)."""
    import threading

    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state, _ = _do_login(client)

    results = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _attempt():
        barrier.wait(timeout=5)
        # Each thread uses its own TestClient sharing the same app/DB so
        # the raw HTTP call itself doesn't serialize on TestClient's own
        # internals — only the real database claim should serialize this.
        local_client = TestClient(create_app())
        local_client.cookies.set(_oauth_state_cookie_name(secure=False), state)
        response = local_client.get(
            CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False
        )
        with lock:
            results.append(response.status_code)

    threads = [threading.Thread(target=_attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert sorted(results) == [302, 400]


# --- denied / error callback -------------------------------------------------


def test_denied_callback_fails_and_never_calls_github(monkeypatch):
    _, token_calls, user_calls = _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    state, _ = _do_login(client)
    response = client.get(
        CALLBACK_PATH, params={"error": "access_denied", "state": state}, follow_redirects=False
    )

    assert response.status_code == 400
    assert token_calls == []
    assert user_calls == []
    assert client.cookies.get(web_config.session_cookie_name()) is None


# --- provider failure ---------------------------------------------------------


def test_token_exchange_provider_failure_yields_no_session(monkeypatch):
    _install_mock_github(monkeypatch, token_status=401)
    client = TestClient(create_app())

    state, _ = _do_login(client)
    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)

    assert response.status_code == 502
    assert client.cookies.get(web_config.session_cookie_name()) is None


def test_user_lookup_provider_failure_yields_no_session(monkeypatch):
    _install_mock_github(monkeypatch, user_status=500)
    client = TestClient(create_app())

    state, _ = _do_login(client)
    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)

    assert response.status_code == 502
    assert client.cookies.get(web_config.session_cookie_name()) is None


def test_malformed_github_identity_payload_yields_no_session(monkeypatch):
    _install_mock_github(monkeypatch, user_payload={"id": "not-an-integer"})
    client = TestClient(create_app())

    state, _ = _do_login(client)
    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)

    assert response.status_code == 502
    assert client.cookies.get(web_config.session_cookie_name()) is None


# --- stale posture (Section 12/18/24) ---------------------------------------


def test_stale_posture_during_session_mint_fails_closed_no_cookies(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state, _ = _do_login(client)

    async def _boom(github_user_id, *, issued_secure):
        raise auth_session.StalePostureError("stale for test")

    # Stage 6C: the callback now mints its session through
    # create_session_for_github() (see web/github_oauth.py/db/auth_sessions.py's
    # create_for_github_sync() docstrings) — never the older, plain
    # create_session() this test used to patch.
    monkeypatch.setattr(auth_session, "create_session_for_github", _boom)

    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)

    assert response.status_code == 503
    assert client.cookies.get(web_config.session_cookie_name()) is None
    assert all(
        not h.startswith(web_config.session_cookie_name() + "=")
        for h in response.headers.get_list("set-cookie")
    )


# --- existing-session behavior (Section 25) ---------------------------------


@pytest.mark.asyncio
async def test_existing_browser_session_is_left_untouched_by_a_new_github_login(monkeypatch):
    """A browser already carrying a valid session for user A performs a
    fresh GitHub login as a DIFFERENT identity (user B). The outcome must
    be a brand-new session resolving to B, and A's original session must
    remain independently valid and unaffected — no implicit merge, no
    silent revocation as a side effect of an unrelated login."""
    user_a = _real_telegram_user()
    issued_a = await auth_session.create_session(user_a, issued_secure=False)

    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    # domain="testserver.local" matches the domain Python's cookiejar
    # normalizes TestClient's bare "testserver" host to (see
    # http.cookiejar's handling of dot-less hostnames) — the same domain
    # every Set-Cookie response from the app itself is later scoped to.
    # Without this, this manually-seeded cookie and the server's own
    # Set-Cookie for the SAME name would live under different (name,
    # domain) jar keys, a TestClient/http.cookiejar artifact that a real
    # browser (single origin, one cookie jar entry per name+domain+path)
    # would never exhibit.
    client.cookies.set(web_config.session_cookie_name(), issued_a.raw_token, domain="testserver.local")

    # Confirm the pre-existing session actually works before the new login.
    pre_login_me = client.get("/api/me")
    assert pre_login_me.status_code == 200
    assert pre_login_me.json()["id"] == str(user_a)

    state, _ = _do_login(client)
    callback_response = client.get(
        CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False
    )
    assert callback_response.status_code == 302

    new_session_token = client.cookies.get(web_config.session_cookie_name())
    assert new_session_token != issued_a.raw_token

    me_after_login = client.get("/api/me")
    new_user_id = me_after_login.json()["id"]
    assert new_user_id != str(user_a)

    # The OLD session (user A's) must still independently resolve — never
    # revoked as a side effect of the new, unrelated GitHub login.
    old_still_valid = await auth_session.resolve_session_user_id(issued_a.raw_token, expected_secure=False)
    assert old_still_valid == user_a


# --- login-CSRF defense (Section 13/25) -------------------------------------


def test_callback_without_the_matching_oauth_state_cookie_fails_closed(monkeypatch):
    """Simulates the classic OAuth login-CSRF: a victim's browser visits
    the real callback URL (valid code+state, straight from an attacker's
    own legitimately-started flow) WITHOUT ever having visited /login
    itself, so it never received the matching oauth-state cookie."""
    _, token_calls, user_calls = _install_mock_github(monkeypatch)

    attacker_client = TestClient(create_app())
    state, _ = _do_login(attacker_client)

    victim_client = TestClient(create_app())  # never called /login — no matching cookie
    response = victim_client.get(
        CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False
    )

    assert response.status_code == 400
    assert token_calls == []
    assert user_calls == []
    assert victim_client.cookies.get(web_config.session_cookie_name()) is None

    # The transaction must remain UNCONSUMED — the legitimate holder
    # (attacker_client, in this simulation) can still complete it
    # normally afterward.
    legitimate_completion = attacker_client.get(
        CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False
    )
    assert legitimate_completion.status_code == 302


def test_callback_with_wrong_oauth_state_cookie_value_fails_closed(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state, _ = _do_login(client)

    client.cookies.set(_oauth_state_cookie_name(secure=False), "a-completely-different-value")
    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)

    assert response.status_code == 400
    assert client.cookies.get(web_config.session_cookie_name()) is None
