"""
Stage 6B regression tests: services/github_oauth_client.py — the only
module that ever talks to github.com/api.github.com. Every HTTP
interaction here is intercepted via httpx.MockTransport (in-process, opens
no real socket at all — pytest-socket's --disable-socket, see pytest.ini,
never even needs to allow anything for these tests). Never contacts real
GitHub — see this module's own docstring.
"""

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import github_oauth_config
import services.github_oauth_client as github_oauth_client
from services.github_oauth_client import GithubOAuthError


def _install_transport(monkeypatch, handler):
    def _client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(github_oauth_client, "_client", _client)


# --- build_authorize_url (Section 4/20) -------------------------------------


def test_authorize_url_targets_the_official_github_endpoint():
    url = github_oauth_client.build_authorize_url(state="s" * 43, code_challenge="c" * 43)
    assert url.startswith(github_oauth_config.GITHUB_AUTHORIZE_URL + "?")


def test_authorize_url_contains_exactly_the_expected_security_parameters():
    state = "state-value-xyz"
    code_challenge = "challenge-value-abc"
    url = github_oauth_client.build_authorize_url(state=state, code_challenge=code_challenge)

    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert params["client_id"] == [github_oauth_config.GITHUB_CLIENT_ID]
    assert params["redirect_uri"] == [github_oauth_config.GITHUB_REDIRECT_URI]
    assert params["state"] == [state]
    assert params["code_challenge"] == [code_challenge]
    assert params["code_challenge_method"] == ["S256"]


def test_authorize_url_never_contains_the_client_secret():
    url = github_oauth_client.build_authorize_url(state="s" * 43, code_challenge="c" * 43)
    assert github_oauth_config.GITHUB_CLIENT_SECRET not in url


def test_authorize_url_never_contains_a_scope_parameter():
    url = github_oauth_client.build_authorize_url(state="s" * 43, code_challenge="c" * 43)
    params = parse_qs(urlparse(url).query)
    assert "scope" not in params


def test_authorize_url_never_contains_a_canonical_uuid_shaped_value():
    """No canonical application UUID is ever a legitimate value in this
    URL — a sanity check that only the expected five parameters exist."""
    url = github_oauth_client.build_authorize_url(state="s" * 43, code_challenge="c" * 43)
    params = parse_qs(urlparse(url).query)
    assert set(params.keys()) == {
        "client_id", "redirect_uri", "state", "code_challenge", "code_challenge_method"
    }


# --- exchange_code_for_token (Section 8/21) ---------------------------------


@pytest.mark.asyncio
async def test_token_exchange_sends_expected_request_shape(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = parse_qs(request.content.decode("utf-8"))
        return httpx.Response(200, json={"access_token": "gho_faketoken", "token_type": "bearer"})

    _install_transport(monkeypatch, handler)

    token = await github_oauth_client.exchange_code_for_token(
        code="the-code", code_verifier="the-verifier", redirect_uri="https://example.com/callback"
    )

    assert token == "gho_faketoken"
    assert captured["method"] == "POST"
    assert captured["url"] == github_oauth_config.GITHUB_TOKEN_URL
    assert captured["headers"]["accept"] == "application/json"
    assert captured["body"]["client_id"] == [github_oauth_config.GITHUB_CLIENT_ID]
    assert captured["body"]["client_secret"] == [github_oauth_config.GITHUB_CLIENT_SECRET]
    assert captured["body"]["code"] == ["the-code"]
    assert captured["body"]["redirect_uri"] == ["https://example.com/callback"]
    assert captured["body"]["code_verifier"] == ["the-verifier"]


@pytest.mark.asyncio
async def test_token_exchange_timeout_raises_generic_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom", request=request)

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_non_2xx_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad_verification_code"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_5xx_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_malformed_json_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all {{{")

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_missing_access_token_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token_type": "bearer"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_empty_access_token_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "", "token_type": "bearer"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_wrong_token_type_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "gho_x", "token_type": "mac"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_response_over_size_limit_raises(monkeypatch):
    huge_payload = json.dumps({"access_token": "x" * 2_000_000, "token_type": "bearer"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=huge_payload.encode("utf-8"))

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )


@pytest.mark.asyncio
async def test_token_exchange_error_message_never_contains_secret_code_or_verifier(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad_verification_code"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError) as excinfo:
        await github_oauth_client.exchange_code_for_token(
            code="THE-SECRET-CODE", code_verifier="THE-SECRET-VERIFIER", redirect_uri="https://example.com/callback"
        )

    message = str(excinfo.value)
    assert "THE-SECRET-CODE" not in message
    assert "THE-SECRET-VERIFIER" not in message
    assert github_oauth_config.GITHUB_CLIENT_SECRET not in message


# --- fetch_github_user_id (Section 9/22) ------------------------------------


@pytest.mark.asyncio
async def test_fetch_user_id_sends_expected_request_shape(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"id": 123456, "login": "octocat"})

    _install_transport(monkeypatch, handler)

    github_id = await github_oauth_client.fetch_github_user_id(access_token="gho_faketoken")

    assert github_id == 123456
    assert captured["method"] == "GET"
    assert captured["url"] == github_oauth_config.GITHUB_USER_API_URL
    assert captured["headers"]["authorization"] == "Bearer gho_faketoken"


@pytest.mark.asyncio
async def test_fetch_user_id_username_change_is_irrelevant_only_id_is_used(monkeypatch):
    """Section 22: a username change with the same numeric id must still
    resolve — proven here at the client layer by showing the login field
    is never even inspected/returned."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 777, "login": "brand-new-username"})

    _install_transport(monkeypatch, handler)

    github_id = await github_oauth_client.fetch_github_user_id(access_token="tok")
    assert github_id == 777


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing id
        {"id": None},  # null id
        {"id": "123"},  # string where integer expected
        {"id": True},  # bool (subclass of int) must not pass as an id
        {"id": False},
        {"id": 0},  # zero
        {"id": -5},  # negative
        {"id": 3.5},  # float
        {"id": [1, 2, 3]},  # absurd malformed value
    ],
)
@pytest.mark.asyncio
async def test_fetch_user_id_rejects_invalid_id_shapes(monkeypatch, payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_fetch_user_id_non_2xx_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_fetch_user_id_403_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "rate limited"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_fetch_user_id_5xx_status_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_fetch_user_id_malformed_json_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_fetch_user_id_non_object_json_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_fetch_user_id_network_error_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_fetch_user_id_never_requires_email_field(monkeypatch):
    """Section 9: 'Do NOT require email' — a payload with no email field
    at all must still succeed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": 999, "login": "no-email-user"})

    _install_transport(monkeypatch, handler)

    assert await github_oauth_client.fetch_github_user_id(access_token="tok") == 999


@pytest.mark.asyncio
async def test_fetch_user_id_error_message_never_contains_the_access_token(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GithubOAuthError) as excinfo:
        await github_oauth_client.fetch_github_user_id(access_token="THE-SECRET-ACCESS-TOKEN")

    assert "THE-SECRET-ACCESS-TOKEN" not in str(excinfo.value)
