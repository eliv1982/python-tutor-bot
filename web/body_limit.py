"""
Per-route request-body size limit (Stage 7A-2) — a small pure-ASGI
middleware, because FastAPI reads and parses a route's JSON body BEFORE any
route dependency (authentication, CSRF) runs: a dependency-level check
would come too late to stop an oversized body from being buffered.

Scoped to an explicit set of (method, path) pairs — deliberately NOT every
/api/* request (future document uploads have their own, larger limit).
Every other request passes through completely untouched.

For a limited route:
  - a declared Content-Length greater than the limit is rejected
    immediately with 413, without reading any body bytes;
  - otherwise the ACTUAL `http.request` body bytes are counted as they
    arrive (never trusting Content-Length alone — it may be absent, as for
    a chunked request, or dishonestly small), and the request is rejected
    with 413 as soon as the running total exceeds the limit, without
    reading any further;
  - a body within the limit is buffered (bounded by the limit itself) and
    replayed to the application as one `http.request` message, so nothing
    is consumed irreversibly: FastAPI receives exactly the bytes the client
    sent. Any later receive() call (e.g. waiting for `http.disconnect`) is
    delegated to the server's original receive channel.

The 413 is produced here, before the application runs at all — so it is
the same for an authenticated and an unauthenticated caller, and never
depends on how FastAPI would treat an exception raised from inside its own
body-parsing code.
"""

import json
from typing import Iterable, Tuple

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_BODY_TOO_LARGE_DETAIL = "Request body too large"


class _Disconnected(Exception):
    """The client disconnected before its full body was received."""


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int, limited_routes: Iterable[Tuple[str, str]]) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.limited_routes = frozenset(limited_routes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (scope["method"], scope["path"]) not in self.limited_routes:
            await self.app(scope, receive, send)
            return

        if self._declared_length_exceeds_limit(scope):
            await self._send_too_large(send)
            return

        try:
            body = await self._read_bounded_body(receive)
        except _Disconnected:
            return
        if body is None:
            await self._send_too_large(send)
            return

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    def _declared_length_exceeds_limit(self, scope: Scope) -> bool:
        for name, value in scope.get("headers", ()):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    # Unparseable: never trusted either way — the actual
                    # received bytes are still counted below.
                    continue
                if declared > self.max_body_bytes:
                    return True
        return False

    async def _read_bounded_body(self, receive: Receive):
        """Returns the full body as bytes, or None as soon as the actual
        received byte count exceeds the limit (without reading further)."""
        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _Disconnected()
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks)

    async def _send_too_large(self, send: Send) -> None:
        payload = json.dumps({"detail": REQUEST_BODY_TOO_LARGE_DETAIL}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})
