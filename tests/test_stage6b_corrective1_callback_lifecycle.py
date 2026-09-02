"""
Stage 6B independent-audit corrective pass #1 — callback lifecycle/error
hardening (MINOR 4) and callback privacy headers (MAJOR 1 item 4), proven
end to end against the real FastAPI app + Starlette TestClient + a REAL
disposable PostgreSQL container, mirroring
tests/test_stage6b_github_oauth_routes.py's own conventions exactly.
GitHub itself is always mocked at the httpx transport layer — no real
network call.

Covers:
  A. OAuth binding cookie cleanup — cleared on terminal flows for the
     CURRENTLY bound transaction; never cleared by a mismatched/older
     callback (multi-tab safety).
  B. Non-ASCII / malformed state — safe 400, never 500, before any
     hmac/DB interaction.
  C. Missing code — fails safely without consuming a still-valid
     transaction, and without disturbing its binding cookie.
  D. Referrer-Policy / Cache-Control headers on every callback response.
  E. GET /login admission-control 429 (MAJOR 2, exercised at the route
     layer here as an integration proof; the exhaustive database-layer
     proof lives in tests/test_stage6b_corrective1_oauth_admission.py).
"""

import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from starlette.testclient import TestClient

import app.oauth_transaction as oauth_transaction
import db.auth_sessions as db_auth_sessions
import db.oauth_transactions as db_oauth_transactions
import github_oauth_config
import services.github_oauth_client as github_oauth_client
import web_config
from web.app import create_app
from web.github_oauth import _oauth_state_cookie_name

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


def _install_mock_github(monkeypatch, *, github_id=None):
    if github_id is None:
        github_id = 55555

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gho_faketoken", "token_type": "bearer"})
        if request.url.path == "/user":
            return httpx.Response(200, json={"id": github_id})
        raise AssertionError(f"unexpected outbound GitHub request: {request.url}")

    def _client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(github_oauth_client, "_client", _client)


def _extract_state(location: str) -> str:
    return parse_qs(urlparse(location).query)["state"][0]


def _do_login(client: TestClient) -> str:
    response = client.get(LOGIN_PATH, follow_redirects=False)
    assert response.status_code == 302
    return _extract_state(response.headers["location"])


def _oauth_cookie_set_cookie_header(response) -> str | None:
    name_prefix = _oauth_state_cookie_name(secure=False) + "="
    return next((h for h in response.headers.get_list("set-cookie") if h.startswith(name_prefix)), None)


def _oauth_cookie_is_deleted(set_cookie_header: str) -> bool:
    """Same check tests/test_stage6a_corrective1_cookie_cleanup.py's own
    `_is_delete_cookie_header()` uses."""
    return "Max-Age=0" in set_cookie_header or "expires" in set_cookie_header.lower()


# --- A. OAuth binding cookie cleanup -----------------------------------------


def test_success_clears_the_oauth_binding_cookie(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert response.status_code == 302

    header = _oauth_cookie_set_cookie_header(response)
    assert header is not None
    assert _oauth_cookie_is_deleted(header)
    assert client.cookies.get(_oauth_state_cookie_name(secure=False)) is None


def test_github_denial_clears_the_oauth_binding_cookie(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    response = client.get(CALLBACK_PATH, params={"error": "access_denied", "state": state}, follow_redirects=False)
    assert response.status_code == 400

    header = _oauth_cookie_set_cookie_header(response)
    assert header is not None
    assert _oauth_cookie_is_deleted(header)


def test_github_denial_consumes_the_transaction(monkeypatch):
    """Documented semantics (MINOR 4C): a denied transaction is dead
    either way, so it is consumed (claim-and-discard) rather than left
    replayable."""
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    client.get(CALLBACK_PATH, params={"error": "access_denied", "state": state}, follow_redirects=False)

    # A follow-up with the SAME state and a real code must now fail —
    # the transaction is gone, not merely "already handled".
    second = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert second.status_code == 400


def test_invalid_or_expired_state_after_matching_binding_clears_the_cookie(monkeypatch):
    """Binding matched (browser owns this transaction) but the claim
    itself fails (e.g. a replay after the transaction was already used
    elsewhere) — still a terminal outcome for THIS browser's own dead
    transaction, so the cookie is cleared."""
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    first = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert first.status_code == 302
    # First success already cleared the cookie and consumed the
    # transaction; manually restore a matching cookie to isolate THIS
    # test's actual target: a matching-binding claim failure clears the
    # cookie again.
    client.cookies.set(_oauth_state_cookie_name(secure=False), state)

    second = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert second.status_code == 400
    header = _oauth_cookie_set_cookie_header(second)
    assert header is not None
    assert _oauth_cookie_is_deleted(header)


def test_provider_failure_after_binding_and_claim_clears_the_cookie(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad_verification_code"})

    def _client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(github_oauth_client, "_client", _client)
    client = TestClient(create_app())
    state = _do_login(client)

    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert response.status_code == 502
    header = _oauth_cookie_set_cookie_header(response)
    assert header is not None
    assert _oauth_cookie_is_deleted(header)


def test_mismatched_callback_never_clears_a_different_still_valid_binding(monkeypatch):
    """Multi-tab safety: tab A's still-active binding cookie must survive
    an unrelated/forged callback (tab B's state, or a completely unknown
    state) arriving while A's cookie is current."""
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state_a = _do_login(client)  # sets the binding cookie to state_a

    # A forged/unrelated callback with a DIFFERENT (but canonical-shaped)
    # state arrives — must fail, and must NOT clear the existing cookie.
    import secrets

    unrelated_state = secrets.token_urlsafe(32)
    response = client.get(
        CALLBACK_PATH, params={"code": "c", "state": unrelated_state}, follow_redirects=False
    )
    assert response.status_code == 400
    header = _oauth_cookie_set_cookie_header(response)
    assert header is None, "a mismatched callback must never touch the OAuth-binding cookie at all"

    # Tab A's own transaction must still be completable afterward.
    completion = client.get(CALLBACK_PATH, params={"code": "c", "state": state_a}, follow_redirects=False)
    assert completion.status_code == 302


# --- B. Non-ASCII / malformed state ------------------------------------------


@pytest.mark.parametrize(
    "bad_state",
    [
        "état-non-ascii-" + "x" * 30,  # non-ASCII
        "文字化け" * 10,  # non-ASCII, CJK
        "s" * 5,  # wrong length
        "!" * 43,  # right length, invalid alphabet
        "a" * 200,  # wildly too long
    ],
)
def test_malformed_or_non_ascii_state_returns_safe_400_never_500(monkeypatch, bad_state):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = client.get(CALLBACK_PATH, params={"code": "c", "state": bad_state}, follow_redirects=False)

    assert response.status_code == 400


def test_malformed_state_makes_zero_database_calls(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    called = {"value": False}
    real_claim = db_oauth_transactions.claim_sync

    def _spy(**kwargs):
        called["value"] = True
        return real_claim(**kwargs)

    monkeypatch.setattr(db_oauth_transactions, "claim_sync", _spy)

    response = client.get(
        CALLBACK_PATH, params={"code": "c", "state": "文字化け" * 10}, follow_redirects=False
    )

    assert response.status_code == 400
    assert called["value"] is False


@pytest.mark.asyncio
async def test_non_ascii_cookie_value_does_not_crash_either(monkeypatch):
    """Defense-in-depth: even a forged non-ASCII Cookie header must not
    reach hmac.compare_digest unguarded.

    Starlette's TestClient (backed by httpx2 in this repo's pinned
    Starlette version — see starlette/testclient.py) refuses client-side
    to even ENCODE a raw non-ASCII Python str into its persistent cookie
    jar's `Cookie:` header (a real `UnicodeEncodeError` at request-build
    time) — which itself already proves this exact code point sequence
    can never reach the server via ANY compliant high-level HTTP client.
    A real attacker forging a raw header would send raw BYTES on the wire
    instead, which the ASGI server decodes per-spec as latin-1 (RFC 7230),
    so this test reconstructs that exact scenario directly: this
    repository's own directly-depended-on classic `httpx` package (used
    elsewhere for services/github_oauth_client.py's own tests) talking to
    the real app in-process via `httpx.ASGITransport`, with the `Cookie`
    header supplied as raw bytes (bypassing httpx2's stricter ASCII-only
    str encoding entirely, exactly like a real non-compliant/forged raw
    request would)."""
    import httpx as classic_httpx

    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    non_ascii_cookie_value = ("文字化け" * 10).encode("utf-8")
    raw_cookie_header = _oauth_state_cookie_name(secure=False).encode("ascii") + b"=" + non_ascii_cookie_value

    transport = classic_httpx.ASGITransport(app=create_app())
    async with classic_httpx.AsyncClient(transport=transport, base_url="http://testserver") as raw_client:
        response = await raw_client.get(
            CALLBACK_PATH,
            params={"code": "c", "state": state},
            headers=[(b"cookie", raw_cookie_header)],
        )

    assert response.status_code == 400


# --- C. Missing code semantics -----------------------------------------------


def test_missing_code_with_matching_binding_does_not_consume_the_transaction(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    incomplete = client.get(CALLBACK_PATH, params={"state": state}, follow_redirects=False)
    assert incomplete.status_code == 400

    # The transaction must still be claimable — a legitimate follow-up
    # (e.g. the real GitHub redirect, if this was some transient artifact)
    # can still complete normally.
    completion = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert completion.status_code == 302


def test_missing_code_does_not_clear_the_binding_cookie(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    response = client.get(CALLBACK_PATH, params={"state": state}, follow_redirects=False)
    assert response.status_code == 400
    header = _oauth_cookie_set_cookie_header(response)
    assert header is None


# --- D. Privacy headers -------------------------------------------------------


def test_success_response_carries_privacy_headers(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    response = client.get(CALLBACK_PATH, params={"code": "c", "state": state}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers.get("referrer-policy") == "no-referrer"
    assert response.headers.get("cache-control") == "no-store"


@pytest.mark.parametrize(
    "params",
    [
        {"code": "c", "state": "s" * 43},  # unknown state
        {"state": "文字化け" * 10},  # malformed state
    ],
)
def test_error_responses_carry_privacy_headers(monkeypatch, params):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = client.get(CALLBACK_PATH, params=params, follow_redirects=False)
    assert response.status_code == 400
    assert response.headers.get("referrer-policy") == "no-referrer"
    assert response.headers.get("cache-control") == "no-store"


def test_denial_error_response_carries_privacy_headers(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())
    state = _do_login(client)

    response = client.get(CALLBACK_PATH, params={"error": "access_denied", "state": state}, follow_redirects=False)
    assert response.status_code == 400
    assert response.headers.get("referrer-policy") == "no-referrer"
    assert response.headers.get("cache-control") == "no-store"


def test_login_redirect_carries_privacy_headers(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    response = client.get(LOGIN_PATH, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers.get("referrer-policy") == "no-referrer"
    assert response.headers.get("cache-control") == "no-store"


def test_response_never_reflects_the_raw_code_or_state_in_body(monkeypatch):
    _install_mock_github(monkeypatch)
    client = TestClient(create_app())

    fake_code = "FAKE-CODE-should-never-be-reflected"
    fake_state = "s" * 43
    response = client.get(CALLBACK_PATH, params={"code": fake_code, "state": fake_state}, follow_redirects=False)

    assert fake_code not in response.text
    assert fake_state not in response.text


# --- E. GET /login admission control (integration proof) --------------------


def test_login_returns_429_when_admission_control_rejects(monkeypatch):
    async def _reject():
        raise oauth_transaction.OAuthAdmissionRejected("test-forced rejection")

    monkeypatch.setattr(oauth_transaction, "create_transaction", _reject)
    client = TestClient(create_app())

    response = client.get(LOGIN_PATH, follow_redirects=False)

    assert response.status_code == 429


def test_login_429_response_creates_no_transaction_row(monkeypatch, postgres_db):
    from sqlalchemy import func, select

    from db.engine import get_sync_engine
    from db.models import GithubOAuthTransaction

    async def _reject():
        raise oauth_transaction.OAuthAdmissionRejected("test-forced rejection")

    monkeypatch.setattr(oauth_transaction, "create_transaction", _reject)
    client = TestClient(create_app())

    client.get(LOGIN_PATH, follow_redirects=False)

    engine = get_sync_engine()
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(GithubOAuthTransaction)).scalar_one()
    assert count == 0


def test_login_rejected_via_real_admission_control_returns_429_and_inserts_nothing(monkeypatch):
    """End-to-end (not monkeypatched) proof: real admission-control
    rejection, reached by exhausting a tiny configured cap, surfaces as a
    429 through the actual route."""
    monkeypatch.setattr(github_oauth_config, "OAUTH_MAX_OUTSTANDING_TRANSACTIONS", 1)
    monkeypatch.setattr(github_oauth_config, "OAUTH_MAX_STARTS_PER_MINUTE", 10_000)
    client = TestClient(create_app())

    first = client.get(LOGIN_PATH, follow_redirects=False)
    assert first.status_code == 302

    second = client.get(LOGIN_PATH, follow_redirects=False)
    assert second.status_code == 429

    from sqlalchemy import func, select

    from db.engine import get_sync_engine
    from db.models import GithubOAuthTransaction

    engine = get_sync_engine()
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(GithubOAuthTransaction)).scalar_one()
    assert count == 1
