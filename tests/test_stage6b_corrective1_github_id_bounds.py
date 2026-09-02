"""
Stage 6B independent-audit corrective pass #1, MINOR 1 — the previous
provider-response validation in services/github_oauth_client.py accepted
ANY positive Python `int` for GitHub's numeric `id`, including values
above `2**63 - 1`. `github_user_id` is persisted as a PostgreSQL
`BigInteger` (db/models.py's GithubAccount) — a signed 64-bit `bigint`,
whose durable domain is exactly `1 <= id <= 2**63 - 1`. A value outside
that range used to sail through this client and only fail later, deep
inside db/github_identity.py, as an opaque `DataError`.

fetch_github_user_id() now rejects anything outside this exact range at
the provider-response validation boundary, as a safe GithubOAuthError —
never letting an out-of-range id reach a database call at all. Every
GitHub HTTP interaction here is mocked at the httpx transport layer (see
services/github_oauth_client.py's own module docstring) — no real network
call.
"""

import httpx
import pytest

import services.github_oauth_client as github_oauth_client
from services.github_oauth_client import GithubOAuthError

_MAX_BIGINT = 2 ** 63 - 1


def _install(monkeypatch, raw_id):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": raw_id, "login": "octocat"})

    def _client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    monkeypatch.setattr(github_oauth_client, "_client", _client)


@pytest.mark.asyncio
async def test_id_of_one_is_accepted(monkeypatch):
    _install(monkeypatch, 1)
    assert await github_oauth_client.fetch_github_user_id(access_token="tok") == 1


@pytest.mark.asyncio
async def test_ordinary_id_is_accepted(monkeypatch):
    _install(monkeypatch, 123456789)
    assert await github_oauth_client.fetch_github_user_id(access_token="tok") == 123456789


@pytest.mark.asyncio
async def test_id_at_exactly_max_signed_bigint_is_accepted(monkeypatch):
    _install(monkeypatch, _MAX_BIGINT)
    assert await github_oauth_client.fetch_github_user_id(access_token="tok") == _MAX_BIGINT


@pytest.mark.asyncio
async def test_id_one_above_max_signed_bigint_is_rejected_before_any_db_call(monkeypatch):
    _install(monkeypatch, _MAX_BIGINT + 1)
    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_id_far_above_max_signed_bigint_is_rejected(monkeypatch):
    _install(monkeypatch, 2 ** 63 + 10 ** 9)
    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_absurdly_large_id_is_rejected(monkeypatch):
    _install(monkeypatch, 10 ** 40)
    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_zero_id_is_rejected(monkeypatch):
    _install(monkeypatch, 0)
    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_negative_id_is_rejected(monkeypatch):
    _install(monkeypatch, -5)
    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")


@pytest.mark.asyncio
async def test_out_of_range_rejection_never_reaches_the_identity_layer(monkeypatch):
    """The rejection must happen at provider-response validation — this
    module never calls into db/github_identity.py at all, so an
    out-of-range id can, by construction, never reach a DataError."""
    import app.github_identity as app_github_identity

    called = {"value": False}

    async def _boom(github_user_id):
        called["value"] = True
        raise AssertionError("resolve_user_uuid must not be reached for an out-of-range id")

    monkeypatch.setattr(app_github_identity, "resolve_user_uuid", _boom)
    _install(monkeypatch, _MAX_BIGINT + 1)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")

    assert called["value"] is False
