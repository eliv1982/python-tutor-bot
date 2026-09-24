"""
Stage 7B-1 regression tests: the GitHub OAuth callback's ONLY behavioral
change — the fixed post-login destination is now "/" (the React application
root) instead of "/api/me" — and proof that every accepted security/privacy
guarantee of the callback is untouched by it.

Same harness as tests/test_stage6b_github_oauth_routes.py: the real
FastAPI app + Starlette TestClient + a REAL disposable PostgreSQL container
(tests/conftest.py's postgres_db) + GitHub HTTP mocked at the httpx
transport layer. No real network call ever happens (pytest.ini's
--disable-socket would fail the run outright).
"""

import inspect
import random
import re
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from starlette.testclient import TestClient

import db.auth_sessions as db_auth_sessions
import services.github_oauth_client as github_oauth_client
import web.frontend as frontend
import web.github_oauth as github_oauth
import web_config
from web.app import create_app
from web.github_oauth import _oauth_state_cookie_name

LOGIN_PATH = "/api/auth/github/login"
CALLBACK_PATH = "/api/auth/github/callback"


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module needs
    REAL `users`/`github_accounts`/`web_sessions` rows."""
    yield


@pytest.fixture(autouse=True)
def _insecure_posture_for_testing(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    yield


def _install_mock_github(monkeypatch, *, github_id=None):
    if github_id is None:
        github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    calls = {"token": 0, "user": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": "gho_faketoken123", "token_type": "bearer"})
        if request.url.path == "/user":
            calls["user"] += 1
            return httpx.Response(200, json={"id": github_id, "login": "octocat"})
        raise AssertionError(f"unexpected outbound GitHub request: {request.url}")

    monkeypatch.setattr(
        github_oauth_client,
        "_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False),
    )
    return calls


def _start_login(client: TestClient) -> str:
    response = client.get(LOGIN_PATH, follow_redirects=False)
    assert response.status_code == 302
    return parse_qs(urlparse(response.headers["location"]).query)["state"][0]


def _complete_login(client: TestClient, **extra_params) -> httpx.Response:
    state = _start_login(client)
    return client.get(
        CALLBACK_PATH,
        params={"code": "test-authorization-code", "state": state, **extra_params},
        follow_redirects=False,
    )


def _set_cookie_headers(response: httpx.Response) -> dict[str, str]:
    """Set-Cookie headers keyed by cookie name."""
    headers = {}
    for header in response.headers.get_list("set-cookie"):
        headers[header.split("=", 1)[0]] = header
    return headers


def _attributes(header: str) -> set[str]:
    """Lower-cased attribute names of a Set-Cookie header (excluding name=value)."""
    return {part.strip().split("=", 1)[0].lower() for part in header.split(";")[1:]}


# --- the redirect itself -------------------------------------------------------


def test_successful_callback_redirects_to_the_application_root(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = _complete_login(client)

    assert response.status_code == 302
    assert response.headers["location"] == "/"


def test_redirect_destination_is_a_fixed_same_origin_constant():
    assert github_oauth._POST_LOGIN_REDIRECT_PATH == "/"


@pytest.mark.parametrize(
    "extra_params",
    [
        {"next": "https://evil.example/"},
        {"return_to": "//evil.example/steal"},
        {"redirect": "https://evil.example"},
        {"redirect_uri": "https://evil.example/api/auth/github/callback"},
        {"continue": "/api/me"},
        {"url": "javascript:alert(1)"},
        {"next": "/\\evil.example", "return_to": "%2F%2Fevil.example"},
    ],
)
def test_callback_destination_is_never_caller_controlled(monkeypatch, extra_params):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = _complete_login(client, **extra_params)

    assert response.status_code == 302
    assert response.headers["location"] == "/"
    assert "evil.example" not in response.headers["location"]


def test_callback_reads_only_the_oauth_protocol_parameters():
    """No `next`/`return_to`/`redirect`-style input exists to be honored."""
    source = inspect.getsource(github_oauth.github_callback)

    assert re.findall(r'query_params\.get\("(\w+)"\)', source) == ["state", "error", "code"]
    assert source.count("RedirectResponse(") == 1
    assert "RedirectResponse(_POST_LOGIN_REDIRECT_PATH" in source


def test_existing_redirect_on_login_start_still_targets_github_only(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = client.get(LOGIN_PATH, follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"].startswith("https://github.com/login/oauth/authorize?")


# --- cookies (insecure local-development posture) ------------------------------


def test_session_and_csrf_cookies_are_still_issued_with_the_accepted_attributes(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    cookies = _set_cookie_headers(_complete_login(client))

    session = cookies[web_config.session_cookie_name()]
    csrf = cookies[web_config.csrf_cookie_name()]
    assert web_config.session_cookie_name() == "session"
    assert web_config.csrf_cookie_name() == "csrf_token"

    assert "httponly" in _attributes(session)
    assert "httponly" not in _attributes(csrf)  # the frontend must be able to read it
    for header in (session, csrf):
        assert "path=/" in header.lower().replace(" ", "")
        assert "samesite=lax" in header.lower().replace(" ", "")
        assert "domain" not in _attributes(header)
        assert "secure" not in _attributes(header)  # explicit local-development posture only


def test_the_frontend_can_read_the_csrf_cookie_but_not_the_session_cookie(monkeypatch):
    """The readable CSRF cookie is what the React client echoes; the session
    cookie must stay unreadable to JavaScript."""
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    cookies = _set_cookie_headers(_complete_login(client))

    assert "httponly" in _attributes(cookies["session"])
    assert "httponly" not in _attributes(cookies["csrf_token"])


def test_callback_success_sends_the_oauth_binding_cookie_deletion(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    cookies = _set_cookie_headers(_complete_login(client))

    state_cookie = cookies[_oauth_state_cookie_name(secure=False)]
    assert "max-age=0" in state_cookie.lower().replace(" ", "")
    assert "httponly" in _attributes(state_cookie)
    assert client.cookies.get(_oauth_state_cookie_name(secure=False)) is None


# --- cookies (production secure posture) ---------------------------------------


def test_secure_posture_callback_still_issues_host_prefixed_secure_cookies(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=True)
    _install_mock_github(monkeypatch)
    # https base URL so the client's cookie jar sends the Secure state cookie back.
    client = TestClient(create_app(), base_url="https://testserver")

    response = _complete_login(client)

    assert response.status_code == 302
    assert response.headers["location"] == "/"
    cookies = _set_cookie_headers(response)
    session = cookies["__Host-session"]
    csrf = cookies["__Host-csrf_token"]

    assert "httponly" in _attributes(session)
    assert "httponly" not in _attributes(csrf)
    for header in (session, csrf):
        assert "secure" in _attributes(header)
        assert "path=/" in header.lower().replace(" ", "")
        assert "samesite=lax" in header.lower().replace(" ", "")
        assert "domain" not in _attributes(header)  # required by the __Host- prefix


# --- privacy headers ------------------------------------------------------------


def test_success_redirect_keeps_the_privacy_headers(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = _complete_login(client)

    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"


def test_success_redirect_does_not_leak_code_or_state(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _start_login(client)

    response = client.get(CALLBACK_PATH, params={"code": "secret-code-value", "state": state}, follow_redirects=False)

    assert "secret-code-value" not in response.text + response.headers["location"]
    assert state not in response.headers["location"]
    assert "?" not in response.headers["location"]


# --- state / PKCE / error behavior unchanged ------------------------------------


def test_state_mismatch_still_fails_closed_with_no_redirect_and_no_session(monkeypatch):
    calls = _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    _start_login(client)

    response = client.get(
        CALLBACK_PATH, params={"code": "c", "state": "s" * 43}, follow_redirects=False
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "OAuth state mismatch"}
    assert "location" not in response.headers
    assert calls == {"token": 0, "user": 0}
    assert web_config.session_cookie_name() not in _set_cookie_headers(response)
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"


def test_missing_state_cookie_still_fails_closed(monkeypatch):
    calls = _install_mock_github(monkeypatch)
    victim = TestClient(create_app())
    attacker = TestClient(create_app())
    attacker_state = _start_login(attacker)  # transaction is valid, but victim's browser never began it

    response = victim.get(
        CALLBACK_PATH, params={"code": "c", "state": attacker_state}, follow_redirects=False
    )

    assert response.status_code == 400
    assert "location" not in response.headers
    assert calls == {"token": 0, "user": 0}
    assert web_config.session_cookie_name() not in _set_cookie_headers(response)


def test_replayed_state_still_fails_after_the_redirect_change(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _start_login(client)
    params = {"code": "c", "state": state}
    first = client.get(CALLBACK_PATH, params=params, follow_redirects=False)
    # Re-arm the browser binding so the second attempt is judged on the
    # (now consumed) transaction rather than the cleared cookie.
    client.cookies.set(_oauth_state_cookie_name(secure=False), state)

    second = client.get(CALLBACK_PATH, params=params, follow_redirects=False)

    assert first.status_code == 302 and first.headers["location"] == "/"
    assert second.status_code == 400
    assert second.json() == {"detail": "Invalid or expired OAuth state"}
    assert "location" not in second.headers


def test_provider_failure_is_still_a_sanitized_error_without_redirect(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded: client_secret=leak-me")

    monkeypatch.setattr(
        github_oauth_client,
        "_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False),
    )
    client = TestClient(create_app())

    response = _complete_login(client)

    assert response.status_code == 502
    assert response.json() == {"detail": "GitHub authentication failed"}
    assert "leak-me" not in response.text
    assert "location" not in response.headers
    assert web_config.session_cookie_name() not in _set_cookie_headers(response)
    assert response.headers["cache-control"] == "no-store"


def test_github_denial_is_still_a_400_without_redirect(monkeypatch):
    calls = _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _start_login(client)

    response = client.get(
        CALLBACK_PATH, params={"error": "access_denied", "state": state}, follow_redirects=False
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "GitHub authorization was not granted"}
    assert "location" not in response.headers
    assert calls == {"token": 0, "user": 0}


# --- the whole browser journey ---------------------------------------------------


def test_after_login_the_redirect_target_is_the_react_shell_and_the_session_works(monkeypatch, tmp_path):
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id='root'></div><!--react-shell-->", encoding="utf-8")
    monkeypatch.setattr(frontend, "FRONTEND_DIST_DIR", dist)
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    callback = _complete_login(client)
    landing = client.get(callback.headers["location"])
    me = client.get("/api/me")

    assert callback.headers["location"] == "/"
    assert landing.status_code == 200
    assert "react-shell" in landing.text
    assert me.status_code == 200
    assert set(me.json()) == {"id", "created_at", "telegram_linked"}


def test_logout_with_the_readable_csrf_cookie_works_end_to_end(monkeypatch):
    """The exact call the React client makes: POST /api/logout echoing the
    CSRF cookie as X-CSRF-Token."""
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    _complete_login(client)
    csrf_token = client.cookies.get(web_config.csrf_cookie_name())

    without_header = client.post("/api/logout")
    logout = client.post("/api/logout", headers={"X-CSRF-Token": csrf_token})
    after = client.get("/api/me")

    assert without_header.status_code == 403
    assert logout.status_code == 204
    assert logout.content == b""
    assert after.status_code == 401
