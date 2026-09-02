"""
GitHub OAuth HTTP client (Stage 6B) — the only module in this codebase
that ever talks to github.com/api.github.com. Two server-to-server calls
only: the Authorization Code + PKCE token exchange, and a single
authenticated `GET /user` identity fetch. Both use httpx (already a
repository dependency via Stage 6A's FastAPI TestClient — see
requirements-dev.txt) with an explicit `trust_env=False`, the same
proxy-immunity posture services/openai_client.py and services/
anthropic_client.py already use for every provider HTTP client this
application builds (see tests/test_stage1f_offline_enforcement.py) — no
GitHub request can ever be silently redirected through an
environment/OS-discovered proxy.

`_client()` is a private factory, not a shared client instance — tests
monkeypatch it to return an httpx.AsyncClient built with
`transport=httpx.MockTransport(...)` instead of a real network transport,
the same "inject a fake instead of the network" idiom this repository
already applies for its LLM provider clients. Since a MockTransport never
opens a real socket, this needs no special pytest-socket allowance (see
pytest.ini) to stay compliant with the suite's offline guarantee.

GithubOAuthError is the ONE exception type this module ever raises for a
GitHub-side failure (timeout, network error, non-2xx status, malformed
JSON, or a missing/invalid field) — its message is always a short, fixed,
generic string. It never interpolates the authorization code, the PKCE
verifier, the client secret, or the access token, so it is always safe to
let propagate up to web/github_oauth.py's callback and from there into a
generic HTTP error response, exactly like db.auth_sessions.
StalePostureError is safe to surface through app/auth_session.py.
"""

import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import httpx

import github_oauth_config

# Explicit connect/read/write/pool timeouts (Section 8 of the Stage 6B
# spec: "explicit connect/read/overall timeout") — never the httpx
# library default, and never unbounded.
_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 10.0
_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_TIMEOUT_SECONDS,
    read=_READ_TIMEOUT_SECONDS,
    write=_READ_TIMEOUT_SECONDS,
    pool=_READ_TIMEOUT_SECONDS,
)

# Bounded response size (Section 8: "bounded response sizes where
# practical") — GitHub's token-exchange and /user responses are both a
# few KB at most; anything past this is treated as a provider failure
# rather than handed to json.loads().
#
# Stage 6B independent-audit corrective pass #1, MINOR 2: this bound used
# to be enforced AFTER `client.get()`/`client.post()` had already fully
# buffered the entire response body in memory — so it was an advertised
# bound, never an actual memory bound. `_read_bounded_json_response()`
# below enforces it with a TRUE streamed read instead: it opens the
# response via `client.stream()`, rejects up front if a declared
# Content-Length already exceeds this value (never reading any body at
# all in that case), and otherwise reads chunks incrementally, aborting
# the instant the running total exceeds this value — so a dishonest/
# absent Content-Length, or a chunked/unbounded response, can never cause
# more than roughly one chunk past this bound to ever sit in memory at
# once, regardless of how much more the server has queued to send.
_MAX_RESPONSE_BYTES = 1_000_000

# GitHub's own numeric account ids are always positive, but Section 22/
# MINOR 1 requires the accepted range to also fit the application's
# durable PostgreSQL domain — `github_user_id` is a `BigInteger`
# (`db/models.py`'s GithubAccount), i.e. PostgreSQL's signed 64-bit
# `bigint`. A value inside `int` range but outside `bigint` range must be
# rejected HERE, at provider-response validation, never allowed to reach
# db/github_identity.py and fail as an opaque DB-layer `DataError`.
_MAX_GITHUB_USER_ID = 2 ** 63 - 1


class GithubOAuthError(Exception):
    """Generic, safe-to-surface failure for any GitHub OAuth HTTP
    interaction — see this module's own docstring for the no-secrets-in-
    the-message contract every raise site here follows."""


def _client() -> httpx.AsyncClient:
    # follow_redirects is left at httpx's own default (False) but stated
    # explicitly (Stage 6B independent-audit corrective pass #1, MINOR 2)
    # so it can never silently change: a redirect followed automatically
    # here would resend the `Authorization: Bearer <access_token>` header
    # (fetch_github_user_id) or the client secret (exchange_code_for_token)
    # to whatever Location a compromised/misbehaving provider endpoint
    # returned.
    return httpx.AsyncClient(trust_env=False, timeout=_TIMEOUT, follow_redirects=False)


def build_authorize_url(*, state: str, code_challenge: str) -> str:
    """
    The GitHub authorization redirect URL (Section 4/20 of the Stage 6B
    spec) — deliberately NO `scope` parameter at all (never an empty
    string either): omitting it entirely requests the minimum GitHub OAuth
    Apps grant by default (read access to the authenticated user's public
    profile, exactly what `GET /user`'s numeric `id` needs), never private
    repo/org access. Carries only `client_id`/`redirect_uri`/`state`/
    `code_challenge`/`code_challenge_method` — never the client secret,
    never the PKCE verifier, never any canonical application UUID.
    """
    params = {
        "client_id": github_oauth_config.GITHUB_CLIENT_ID,
        "redirect_uri": github_oauth_config.GITHUB_REDIRECT_URI,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{github_oauth_config.GITHUB_AUTHORIZE_URL}?{urlencode(params)}"


async def _read_bounded_json_response(
    client: httpx.AsyncClient, method: str, url: str, *, context: str, **kwargs
) -> dict:
    """
    True bounded-memory streamed read (Stage 6B independent-audit
    corrective pass #1, MINOR 2) shared by both provider calls below.

    1. Opens the response via `client.stream()` — headers/status arrive
       without buffering any body yet.
    2. Checks the status code from headers alone; a non-2xx response is
       rejected WITHOUT ever reading its body.
    3. If a `Content-Length` header is present and already exceeds
       `_MAX_RESPONSE_BYTES`, rejects immediately — no body is read at
       all in that case (never trust a body read to a declared-oversized
       length).
    4. Otherwise reads the body incrementally via `aiter_bytes()`,
       tracking a running total, and aborts the instant that total
       exceeds `_MAX_RESPONSE_BYTES` — this is what actually bounds
       memory for a dishonest/absent Content-Length or a chunked/
       streamed response of unknown length: at most one chunk past the
       limit is ever held, never the full body.
    5. Only once the FULL (bounded) body has been collected is it handed
       to `json.loads()` — never partial/streamed JSON parsing.

    Network/connection errors (during either the initial connect or a
    later chunk read) and read-time size-limit aborts both surface as the
    same generic GithubOAuthError — see this module's own docstring for
    why every raise site here uses a short, fixed, no-secrets message.
    """
    try:
        async with client.stream(method, url, **kwargs) as response:
            if response.status_code != 200:
                raise GithubOAuthError(f"GitHub {context} returned HTTP {response.status_code}")

            declared_length = response.headers.get("content-length")
            if declared_length is not None:
                try:
                    declared = int(declared_length)
                except ValueError:
                    declared = None
                if declared is not None and declared > _MAX_RESPONSE_BYTES:
                    raise GithubOAuthError(f"GitHub {context} response exceeded the maximum allowed size")

            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise GithubOAuthError(f"GitHub {context} response exceeded the maximum allowed size")
    except httpx.HTTPError:
        raise GithubOAuthError(f"GitHub {context} request failed") from None

    try:
        payload = json.loads(bytes(body))
    except ValueError:
        raise GithubOAuthError(f"GitHub {context} response was not valid JSON") from None
    if not isinstance(payload, dict):
        raise GithubOAuthError(f"GitHub {context} response was not a JSON object")
    return payload


async def exchange_code_for_token(*, code: str, code_verifier: str, redirect_uri: str) -> str:
    """
    Server-to-server Authorization Code + PKCE token exchange (Section 8).
    Sent exactly once per call — this function performs no retry of its
    own (Section 8: "do not silently retry authorization-code exchange in
    a way that may replay a one-time code"; a caller that wants another
    attempt must obtain a fresh code via a fresh login). Returns ONLY the
    bearer access token as a plain string — an ephemeral, in-memory value
    the caller (web/github_oauth.py) uses exactly once (the following
    `GET /user` call) and then discards; nothing here persists it, logs
    it, or returns it inside any larger structure a caller might
    accidentally serialize/log whole.
    """
    async with _client() as client:
        payload = await _read_bounded_json_response(
            client,
            "POST",
            github_oauth_config.GITHUB_TOKEN_URL,
            context="token exchange",
            data={
                "client_id": github_oauth_config.GITHUB_CLIENT_ID,
                "client_secret": github_oauth_config.GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"Accept": "application/json"},
        )

    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise GithubOAuthError("GitHub token exchange response was missing a valid access_token")

    token_type = payload.get("token_type")
    if token_type is not None and not (isinstance(token_type, str) and token_type.lower() == "bearer"):
        raise GithubOAuthError("GitHub token exchange returned an unexpected token_type")

    return access_token


async def fetch_github_user_id(*, access_token: str) -> int:
    """
    The one and only use of the ephemeral access token (Section 9/10):
    a single authenticated `GET /user` call, whose ONLY extracted,
    persisted value is the stable numeric `id` — never the login,
    display name, avatar URL, profile URL, or email. Validates strictly:
    `id` must be a real (non-bool) integer within
    `1 <= id <= 2**63 - 1` — matching Section 22's "missing / null /
    string / bool / zero / negative" rejection requirements, PLUS (Stage
    6B independent-audit corrective pass #1, MINOR 1) an upper bound
    matching the exact durable PostgreSQL domain this value is later
    stored in (`github_user_id BigInteger` — db/models.py's GithubAccount):
    GitHub's own numeric ids are always positive and always fit a signed
    64-bit bigint today, so anything outside this range can only be a
    malformed/spoofed payload — rejected HERE, before it ever reaches
    db/github_identity.py and fails there as an opaque DB `DataError`.
    """
    async with _client() as client:
        payload = await _read_bounded_json_response(
            client,
            "GET",
            github_oauth_config.GITHUB_USER_API_URL,
            context="user lookup",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    raw_id = payload.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise GithubOAuthError("GitHub user lookup returned a non-integer id")
    if raw_id <= 0:
        raise GithubOAuthError("GitHub user lookup returned a non-positive id")
    if raw_id > _MAX_GITHUB_USER_ID:
        raise GithubOAuthError("GitHub user lookup returned an id outside the supported range")

    return raw_id
