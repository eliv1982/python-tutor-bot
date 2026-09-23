"""
Stage 7A-2 regression tests: web.body_limit.RequestBodyLimitMiddleware —
the per-route 64 KiB ACTUAL request-body cap for POST /api/chat and
PATCH /api/settings.

Section A drives the middleware directly with raw ASGI messages, so the
actual-byte counting can be proven independently of any HTTP client's own
Content-Length handling: chunked bodies with no Content-Length, a
dishonestly small Content-Length, and early rejection that stops reading.
Section B proves the same behavior through the real create_app() stack,
including that the cap is route-scoped, not global.

Offline: no database, provider, Qdrant, or Telegram access.
"""

import asyncio
import json
import uuid

import pytest
from fastapi import Request
from starlette.testclient import TestClient

import app.text_chat as text_chat
import web_config
from web.app import JSON_BODY_LIMITED_ROUTES, create_app
from web.body_limit import RequestBodyLimitMiddleware
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME, get_current_user_id

LIMIT = web_config.MAX_JSON_BODY_BYTES
TOO_LARGE = {"detail": "Request body too large"}
SESSION_TOKEN = "stage7a2-body-limit-dummy-token"


def test_limit_is_64_kib():
    assert LIMIT == 65536


def test_limited_routes_are_exactly_the_small_json_routes():
    """Stage 7A-3 added POST /api/retrieval/search to this same 64 KiB
    cap (a small JSON query, not a file upload) — see
    tests/test_stage7a3_retrieval_api.py for that route's own behavior;
    POST /api/documents deliberately stays OUT of this set (it has its
    own, much larger DOCUMENT_BODY_LIMITED_ROUTES cap instead — see
    web/app.py)."""
    assert JSON_BODY_LIMITED_ROUTES == frozenset(
        {("POST", "/api/chat"), ("PATCH", "/api/settings"), ("POST", "/api/retrieval/search")}
    )


# ============================================================================
# A. Raw ASGI.
# ============================================================================


class _RecordingApp:
    """Inner ASGI app: drains the body it is given and records it."""

    def __init__(self):
        self.called = False
        self.body = None
        self.receive = None

    async def __call__(self, scope, receive, send):
        self.called = True
        self.receive = receive
        chunks = []
        while True:
            message = await receive()
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        self.body = b"".join(chunks)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class _Receive:
    """Feeds `chunks` as http.request messages; counts calls. After the
    body, returns http.disconnect."""

    def __init__(self, chunks, *, fail_if_called=False):
        self.chunks = list(chunks)
        self.calls = 0
        self.fail_if_called = fail_if_called

    async def __call__(self):
        if self.fail_if_called:
            raise AssertionError("body must not be read")
        self.calls += 1
        if self.chunks:
            chunk = self.chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(self.chunks)}
        return {"type": "http.disconnect"}


def _scope(method="POST", path="/api/chat", content_length=None):
    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


def _run(middleware, scope, receive):
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def _status_and_body(sent):
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], body


def _middleware(inner):
    return RequestBodyLimitMiddleware(inner, max_body_bytes=LIMIT, limited_routes=JSON_BODY_LIMITED_ROUTES)


def _chunks(total, size=4096):
    data = b"a" * total
    return [data[i:i + size] for i in range(0, total, size)] or [b""]


@pytest.mark.parametrize("content_length", [None, LIMIT], ids=["no-content-length", "honest-content-length"])
def test_exactly_the_limit_is_passed_through_byte_for_byte(content_length):
    inner = _RecordingApp()
    sent = _run(_middleware(inner), _scope(content_length=content_length), _Receive(_chunks(LIMIT)))
    assert inner.called
    assert inner.body == b"a" * LIMIT
    assert _status_and_body(sent) == (200, b"ok")


@pytest.mark.parametrize("content_length", [None, LIMIT + 1], ids=["no-content-length", "honest-content-length"])
def test_one_byte_over_the_limit_is_413_and_the_app_never_runs(content_length):
    inner = _RecordingApp()
    sent = _run(_middleware(inner), _scope(content_length=content_length), _Receive(_chunks(LIMIT + 1)))
    assert not inner.called
    status, body = _status_and_body(sent)
    assert status == 413
    assert json.loads(body) == TOO_LARGE


def test_declared_oversized_content_length_is_rejected_before_reading_any_body():
    inner = _RecordingApp()
    receive = _Receive([b"x"], fail_if_called=True)
    sent = _run(_middleware(inner), _scope(content_length=LIMIT + 1), receive)
    assert receive.calls == 0
    assert not inner.called
    assert _status_and_body(sent)[0] == 413


def test_chunked_oversized_body_without_content_length_stops_reading_at_the_limit():
    inner = _RecordingApp()
    chunk_size = 4096
    receive = _Receive(_chunks(LIMIT * 4, chunk_size))
    sent = _run(_middleware(inner), _scope(content_length=None), receive)
    assert _status_and_body(sent)[0] == 413
    assert not inner.called
    # Stopped as soon as the running total exceeded the limit — never
    # drained the remaining ~3x limit worth of chunks.
    assert receive.calls == LIMIT // chunk_size + 1


def test_dishonest_small_content_length_is_not_trusted():
    inner = _RecordingApp()
    sent = _run(_middleware(inner), _scope(content_length=10), _Receive(_chunks(LIMIT + 1)))
    assert not inner.called
    assert _status_and_body(sent)[0] == 413


def test_unparseable_content_length_falls_back_to_actual_byte_counting():
    inner = _RecordingApp()
    scope = _scope()
    scope["headers"].append((b"content-length", b"not-a-number"))
    sent = _run(_middleware(inner), scope, _Receive(_chunks(LIMIT + 1)))
    assert not inner.called
    assert _status_and_body(sent)[0] == 413


def test_disconnect_mid_body_sends_nothing_and_never_runs_the_app():
    inner = _RecordingApp()

    class _Disconnecting(_Receive):
        async def __call__(self):
            self.calls += 1
            if self.calls == 1:
                return {"type": "http.request", "body": b"abc", "more_body": True}
            return {"type": "http.disconnect"}

    sent = _run(_middleware(inner), _scope(), _Disconnecting([]))
    assert sent == []
    assert not inner.called


def test_after_the_replayed_body_receive_delegates_to_the_server_channel():
    seen = []

    async def inner(scope, receive, send):
        seen.append(await receive())
        seen.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    _run(_middleware(inner), _scope(), _Receive([b"ab", b"cd"]))
    assert seen[0] == {"type": "http.request", "body": b"abcd", "more_body": False}
    assert seen[1] == {"type": "http.disconnect"}


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/api/settings"),
        ("POST", "/api/settings"),
        ("PATCH", "/api/chat"),
        ("POST", "/api/documents"),
        ("POST", "/api/chat/"),
        ("GET", "/api/auth/github/callback"),
        ("POST", "/api/logout"),
    ],
)
def test_non_target_routes_pass_through_untouched_and_uncapped(method, path):
    inner = _RecordingApp()
    original_receive = _Receive(_chunks(LIMIT * 3))
    sent = _run(_middleware(inner), _scope(method=method, path=path, content_length=LIMIT * 3), original_receive)
    assert inner.receive is original_receive  # not wrapped at all
    assert inner.body == b"a" * (LIMIT * 3)
    assert _status_and_body(sent)[0] == 200


def test_non_http_scopes_pass_through():
    called = []

    async def inner(scope, receive, send):
        called.append(scope["type"])

    asyncio.run(_middleware(inner)({"type": "lifespan"}, None, None))
    assert called == ["lifespan"]


# ============================================================================
# B. Through the real application stack.
# ============================================================================


@pytest.fixture(autouse=True)
def _insecure_cookie_posture(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)


@pytest.fixture
def provider(monkeypatch):
    calls = []

    async def fake_generate(messages, max_tokens=None):
        calls.append(messages)
        return "reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)
    return calls


def _padded_json(obj, total_bytes):
    """Valid JSON of exactly `total_bytes` bytes (trailing JSON whitespace)."""
    raw = json.dumps(obj).encode("utf-8")
    assert len(raw) <= total_bytes
    return raw + b" " * (total_bytes - len(raw))


def _authed_client(app=None):
    app = app or create_app()
    app.dependency_overrides[get_current_user_id] = lambda: uuid.uuid4()
    client = TestClient(app)
    client.cookies.set(web_config.session_cookie_name(), SESSION_TOKEN)
    return client


def _headers():
    return {CSRF_HEADER_NAME: derive_csrf_token(SESSION_TOKEN), "content-type": "application/json"}


def test_chat_body_of_exactly_64_kib_is_accepted(provider):
    response = _authed_client().post("/api/chat", content=_padded_json({"message": "q"}, LIMIT), headers=_headers())
    assert response.status_code == 200
    assert len(provider) == 1


def test_chat_body_one_byte_over_is_413(provider):
    response = _authed_client().post(
        "/api/chat", content=_padded_json({"message": "q"}, LIMIT + 1), headers=_headers()
    )
    assert response.status_code == 413
    assert response.json() == TOO_LARGE
    assert provider == []


def test_unauthenticated_oversized_chat_is_413_not_401(provider):
    """Rejected before FastAPI buffers/parses anything and before any
    authentication/CSRF dependency runs."""
    response = TestClient(create_app()).post(
        "/api/chat", content=b"x" * (LIMIT + 1), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json() == TOO_LARGE
    assert provider == []


def test_chunked_oversized_chat_without_content_length_is_413(provider):
    def body():
        for _ in range(20):
            yield b" " * 8192

    response = _authed_client().post("/api/chat", content=body(), headers=_headers())
    assert response.status_code == 413
    assert provider == []


def test_settings_patch_is_capped(monkeypatch):
    import db.preferences as db_preferences

    writes = []
    monkeypatch.setattr(db_preferences, "set_mode_sync", lambda user_id, mode: writes.append(mode))
    client = _authed_client()

    ok = client.patch("/api/settings", content=_padded_json({"mode": "rag"}, LIMIT), headers=_headers())
    assert ok.status_code == 200
    over = client.patch("/api/settings", content=_padded_json({"mode": "voice"}, LIMIT + 1), headers=_headers())
    assert over.status_code == 413
    assert over.json() == TOO_LARGE
    assert writes == ["rag"]


def test_other_existing_api_routes_are_not_capped():
    """POST /api/logout with a large body reaches its normal auth gate
    (401), not the 64 KiB cap."""
    response = TestClient(create_app()).post(
        "/api/logout", content=b"x" * (LIMIT * 2), headers={CSRF_HEADER_NAME: "irrelevant"}
    )
    assert response.status_code == 401


def test_a_future_non_target_api_route_is_not_capped():
    """Stand-in for a future endpoint registered on the same app that is
    scoped into NEITHER body-limit middleware instance. (Stage 7A-3 gave
    POST /api/documents its own, much larger, dedicated limit — see
    test_stage7a3_documents_api.py's own body-size tests for that route's
    actual behavior — so this stand-in now uses a path neither middleware
    instance has ever heard of.)"""
    app = create_app()

    @app.post("/api/not-a-real-route")
    async def _stand_in(request: Request):
        return {"size": len(await request.body())}

    size = LIMIT * 4
    response = TestClient(app).post("/api/not-a-real-route", content=b"x" * size)
    assert response.status_code == 200
    assert response.json() == {"size": size}
