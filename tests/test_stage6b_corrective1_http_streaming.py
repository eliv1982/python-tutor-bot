"""
Stage 6B independent-audit corrective pass #1, MINOR 2 — the previous
client used `client.get()`/`client.post()`, which fully buffers the
ENTIRE response body before `_parse_json_response()` ever checked its
size. The advertised 1 MB bound (`_MAX_RESPONSE_BYTES`) was therefore
never an actual memory bound: a malicious/misbehaving GitHub endpoint (or
anything on that network path) could still force this process to buffer
an arbitrarily large body first.

services/github_oauth_client.py's `_read_bounded_json_response()` now
opens the response via `client.stream()` and reads it incrementally,
aborting the instant the running total exceeds the limit. This file
proves that TRUE streaming behavior directly: a hand-written
`httpx.AsyncBaseTransport` that yields the response body one chunk at a
time (via a custom async byte stream, never a single pre-built
`httpx.Response(content=...)` the way httpx.MockTransport normally
works) and records exactly how many bytes/chunks the client actually
pulled before giving up — proving the client stops reading near the
configured limit instead of ever consuming a full oversized body,
regardless of whether Content-Length is absent, present-and-honest, or
present-and-dishonest.

Never contacts real GitHub — every transport here is entirely in-process,
same as every other Stage 6B HTTP-client test (see
services/github_oauth_client.py's own module docstring).
"""

import pytest
import httpx

import services.github_oauth_client as github_oauth_client
from services.github_oauth_client import GithubOAuthError, _MAX_RESPONSE_BYTES


class _RecordingAsyncByteStream(httpx.AsyncByteStream):
    """A hand-rolled async byte stream (httpx's `stream=` response
    constructor parameter contract — must subclass `httpx.AsyncByteStream`
    itself; httpx asserts `isinstance(response.stream, AsyncByteStream)`
    when sending a request) that yields `total_bytes` of filler content in
    `chunk_size`-sized pieces, recording exactly how many bytes and chunks
    were ACTUALLY pulled by whatever iterates it — the whole point being
    to prove the client stops asking for more chunks once its own bound is
    exceeded, never that it received a small response to begin with."""

    def __init__(self, total_bytes: int, chunk_size: int = 8192):
        self._total_bytes = total_bytes
        self._chunk_size = chunk_size
        self.bytes_yielded = 0
        self.chunks_yielded = 0
        self.closed = False

    async def __aiter__(self):
        remaining = self._total_bytes
        while remaining > 0:
            n = min(self._chunk_size, remaining)
            yield b"x" * n
            self.bytes_yielded += n
            self.chunks_yielded += 1
            remaining -= n

    async def aclose(self) -> None:
        self.closed = True


class _StreamingTransport(httpx.AsyncBaseTransport):
    """Serves one fixed streamed response for every request — enough for
    these single-call client functions. `headers` lets a test control
    (or omit) Content-Length independently of the ACTUAL byte count the
    stream will yield, including a dishonest one."""

    def __init__(self, *, total_bytes: int, chunk_size: int = 8192, headers: dict | None = None, status_code: int = 200):
        self.stream = _RecordingAsyncByteStream(total_bytes, chunk_size)
        self._headers = headers or {}
        self._status_code = status_code
        self.request_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        return httpx.Response(self._status_code, headers=self._headers, stream=self.stream, request=request)


def _install(monkeypatch, transport: httpx.AsyncBaseTransport) -> None:
    def _client():
        return httpx.AsyncClient(transport=transport, trust_env=False)

    monkeypatch.setattr(github_oauth_client, "_client", _client)


# --- true incremental abort: no Content-Length, huge chunked body ----------


@pytest.mark.asyncio
async def test_fetch_user_id_aborts_near_the_limit_for_an_unbounded_chunked_response(monkeypatch):
    """No Content-Length at all (simulating a genuinely chunked/streamed
    response of unknown length) — the ONLY thing that can possibly bound
    memory here is the incremental abort during iteration."""
    huge_total = _MAX_RESPONSE_BYTES * 50
    transport = _StreamingTransport(total_bytes=huge_total, chunk_size=8192, headers={})
    _install(monkeypatch, transport)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")

    # The client must have stopped WELL short of the full (50x-oversized)
    # body — at most a small number of chunks past the configured limit,
    # never anywhere close to the total the "server" had queued.
    assert transport.stream.bytes_yielded <= _MAX_RESPONSE_BYTES + 8192
    assert transport.stream.bytes_yielded < huge_total


@pytest.mark.asyncio
async def test_token_exchange_aborts_near_the_limit_for_an_unbounded_chunked_response(monkeypatch):
    huge_total = _MAX_RESPONSE_BYTES * 50
    transport = _StreamingTransport(total_bytes=huge_total, chunk_size=8192, headers={})
    _install(monkeypatch, transport)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )

    assert transport.stream.bytes_yielded <= _MAX_RESPONSE_BYTES + 8192
    assert transport.stream.bytes_yielded < huge_total


# --- dishonest Content-Length: small declared, huge actual body -------------


@pytest.mark.asyncio
async def test_fetch_user_id_does_not_trust_a_dishonest_small_content_length(monkeypatch):
    """Content-Length claims a tiny body, but the transport actually
    streams far more — the running-total abort during iteration must
    still catch this; the client must not finish "successfully" just
    because it stopped trusting Content-Length past the header check."""
    huge_total = _MAX_RESPONSE_BYTES * 10
    transport = _StreamingTransport(total_bytes=huge_total, chunk_size=8192, headers={"content-length": "10"})
    _install(monkeypatch, transport)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")

    assert transport.stream.bytes_yielded <= _MAX_RESPONSE_BYTES + 8192


# --- honest oversized Content-Length: rejected BEFORE any body read --------


@pytest.mark.asyncio
async def test_fetch_user_id_rejects_oversized_content_length_before_reading_any_body(monkeypatch):
    transport = _StreamingTransport(
        total_bytes=_MAX_RESPONSE_BYTES * 5,
        chunk_size=8192,
        headers={"content-length": str(_MAX_RESPONSE_BYTES * 5)},
    )
    _install(monkeypatch, transport)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")

    # Declared length alone was enough to reject — no body should have
    # been read at all.
    assert transport.stream.bytes_yielded == 0
    assert transport.stream.chunks_yielded == 0


@pytest.mark.asyncio
async def test_token_exchange_rejects_oversized_content_length_before_reading_any_body(monkeypatch):
    transport = _StreamingTransport(
        total_bytes=_MAX_RESPONSE_BYTES * 5,
        chunk_size=8192,
        headers={"content-length": str(_MAX_RESPONSE_BYTES * 5)},
    )
    _install(monkeypatch, transport)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.exchange_code_for_token(
            code="c", code_verifier="v", redirect_uri="https://example.com/callback"
        )

    assert transport.stream.bytes_yielded == 0


# --- non-2xx status: rejected without reading the body ----------------------


@pytest.mark.asyncio
async def test_fetch_user_id_non_2xx_status_does_not_read_the_body(monkeypatch):
    transport = _StreamingTransport(total_bytes=_MAX_RESPONSE_BYTES * 5, chunk_size=8192, status_code=502)
    _install(monkeypatch, transport)

    with pytest.raises(GithubOAuthError):
        await github_oauth_client.fetch_github_user_id(access_token="tok")

    assert transport.stream.bytes_yielded == 0


# --- a genuinely small, valid response still works end to end ---------------


@pytest.mark.asyncio
async def test_fetch_user_id_small_valid_streamed_response_still_succeeds(monkeypatch):
    import json

    payload = json.dumps({"id": 4242}).encode("utf-8")

    class _SmallStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield payload

        async def aclose(self) -> None:
            pass

    class _SmallTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-length": str(len(payload))}, stream=_SmallStream(), request=request)

    _install(monkeypatch, _SmallTransport())

    github_id = await github_oauth_client.fetch_github_user_id(access_token="tok")
    assert github_id == 4242


# --- redirects are never followed (Authorization/secret leakage guard) -----


@pytest.mark.asyncio
async def test_client_factory_disables_redirects():
    async with github_oauth_client._client() as client:
        assert client.follow_redirects is False
