"""
Stage 6B independent-audit corrective pass #1, MAJOR 1 — OAuth callback
credentials leaking into access logs.

Uvicorn's DEFAULT access logging includes the request path WITH its query
string, so a callback request (`GET /api/auth/github/callback?code=...&
state=...`) used to have the app server itself write the raw authorization
`code`/`state` into an access log. web_main.py now launches Uvicorn with
`access_log=False` (the PRIMARY guarantee); web/app.py's create_app() ALSO
disables the `"uvicorn.access"` logger directly (defense-in-depth for a
deployment that launches this ASGI app through the bare `uvicorn`
CLI/config instead of web_main.py).

This file proves, in order:
  1. web_main.py's real `__main__` block actually passes
     `access_log=False` to `uvicorn.run()` (a monkeypatched capture, not a
     source-text grep).
  2. create_app() disables the `"uvicorn.access"` logger by itself,
     independent of web_main.py — the CLI/config-launch defense-in-depth
     layer.
  3. A REAL Uvicorn server (a genuine bound socket, a genuine HTTP request
     over the wire) with `access_log=False` never emits a log line
     containing a distinctive fake `code`/`state`, searching ALL captured
     log records/messages — not merely grepping `logger.*` source lines.
  4. A callback rejected BEFORE the transaction is claimed (a login-CSRF
     binding mismatch) never causes this application's OWN logging (the
     `"bot"` logger / anything else) to record the raw fake code/state
     either — proven via TestClient plus a log-capturing handler attached
     across every relevant logger.
  5. Uvicorn's `"uvicorn.error"` logger (startup/crash diagnostics) is
     left completely functional — MAJOR 1 explicitly forbids silencing
     application/error logging as a side effect of this fix.
"""

import contextlib
import io
import logging
import runpy
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from web.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_MAIN_PATH = str(PROJECT_ROOT / "web_main.py")

FAKE_CODE = "FAKE-AUTH-CODE-4f8a9c2e-do-not-appear-in-logs"
FAKE_STATE = "FAKE-STATE-b91d7a3c-do-not-appear-in-logs"


# --- 1. web_main.py's real __main__ block disables access logging ----------


def test_web_main_launches_uvicorn_with_access_log_disabled(monkeypatch):
    captured = {}

    def _fake_run(app, **kwargs):
        captured.update(kwargs)
        captured["app"] = app

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr("utils.logging.configure_logging", lambda *a, **kw: None)
    monkeypatch.setenv("WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("WEB_PORT", "8000")

    runpy.run_path(WEB_MAIN_PATH, run_name="__main__")

    assert captured.get("access_log") is False


# --- 2. create_app() disables the uvicorn.access logger itself -------------


def test_create_app_disables_the_uvicorn_access_logger():
    access_logger = logging.getLogger("uvicorn.access")
    # Simulate Uvicorn's own default setup having already (re-)armed this
    # logger, the way Config.__init__() does before this factory ever runs
    # (see web/app.py's own docstring on call ordering) — proves create_app()
    # actively overrides it rather than merely observing an
    # already-disabled default.
    access_logger.disabled = False
    access_logger.handlers = [logging.StreamHandler()]
    access_logger.propagate = True

    create_app()

    assert access_logger.disabled is True
    assert access_logger.handlers == []
    assert access_logger.propagate is False


def test_create_app_never_disables_uvicorn_error_or_the_application_logger():
    error_logger = logging.getLogger("uvicorn.error")
    error_logger.disabled = False
    app_logger = logging.getLogger("bot")
    app_logger.disabled = False

    create_app()

    assert error_logger.disabled is False
    assert app_logger.disabled is False


# --- 3. real Uvicorn server: no access-log line ever contains credentials --


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# Every test in this module drives its request through an HTTP client
# library (httpx/httpx2 for the real-server tests below, and — since this
# repo's pinned Starlette version prefers httpx2 over the deprecated
# plain `httpx` for its TestClient, see starlette/testclient.py — httpx2
# again for the in-process TestClient test) whose OWN client-side request
# logger (an INFO-level "HTTP Request: GET <url> ..." line, logged by the
# client BEFORE the server ever sees the request) necessarily contains the
# exact URL being requested, query string included. That is the TEST
# HARNESS logging its own outbound call, not this application logging
# anything — it would appear identically for ANY httpx-based test client
# hitting ANY URL, regardless of what this corrective pass changed, and
# saying nothing about whether the CODE UNDER TEST ever logs the secret.
# Excluded by logger name so the assertion stays meaningful; every other
# logger (this application's own "bot" logger, "uvicorn"/"uvicorn.access"/
# "uvicorn.error", root, anything else) remains fully captured.
_CLIENT_LIBRARY_LOGGER_PREFIXES = ("httpx", "httpcore")


class _ExcludeClientLibraryNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(_CLIENT_LIBRARY_LOGGER_PREFIXES)


@contextlib.contextmanager
def _capturing_all_loggers():
    """Attaches one StreamHandler to the ROOT logger for the duration of
    the block, capturing every record from every logger in the process
    that propagates (which every logger does by default unless explicitly
    disabled) — deliberately broad ("search ALL log records/messages", not
    a targeted grep of known call sites) — except the outbound HTTP CLIENT
    library's own request-logging noise (see _CLIENT_LIBRARY_LOGGER_PREFIXES
    above)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    handler.addFilter(_ExcludeClientLibraryNoise())
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)


def test_real_uvicorn_server_with_access_log_disabled_never_logs_the_callback_query_string(postgres_db):
    """`postgres_db` is required here: create_app()'s lifespan startup
    (app.auth_session.apply_startup_posture()) needs a REAL reachable
    PostgreSQL to complete — without it, this real Uvicorn server would
    never finish starting at all against tests/conftest.py's deliberately
    poisoned default DATABASE_URL."""
    port = _free_port()

    with _capturing_all_loggers() as stream:
        config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, access_log=False, log_level="info")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.05)
            assert server.started, "real Uvicorn server never reported started"

            response = httpx.get(
                f"http://127.0.0.1:{port}/api/auth/github/callback",
                params={"code": FAKE_CODE, "state": FAKE_STATE},
                timeout=10,
            )
            # Rejected (no matching OAuth-binding cookie) — the request
            # itself is real; what matters here is what got LOGGED.
            assert response.status_code == 400

            time.sleep(0.3)  # give any would-be access-log write a chance to land
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    log_output = stream.getvalue()
    assert FAKE_CODE not in log_output
    assert FAKE_STATE not in log_output


def test_real_uvicorn_server_launched_like_bare_cli_still_never_logs_credentials(postgres_db):
    """Defense-in-depth proof: even with `access_log` left at Uvicorn's
    own default (True — simulating a deployer running the bare `uvicorn`
    CLI/config without an explicit `--no-access-log` flag),
    create_app()'s own `"uvicorn.access"` logger disable must still
    prevent the leak."""
    port = _free_port()

    with _capturing_all_loggers() as stream:
        config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="info")
        assert config.access_log is True, "test setup must actually exercise Uvicorn's default-enabled access log"
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.05)
            assert server.started

            response = httpx.get(
                f"http://127.0.0.1:{port}/api/auth/github/callback",
                params={"code": FAKE_CODE, "state": FAKE_STATE},
                timeout=10,
            )
            assert response.status_code == 400

            time.sleep(0.3)
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    log_output = stream.getvalue()
    assert FAKE_CODE not in log_output
    assert FAKE_STATE not in log_output


# --- 4. application-level logging never records a pre-claim rejection ------


def test_pre_claim_rejection_never_logs_the_fake_code_or_state_via_testclient(monkeypatch):
    """Complements the real-server proof above with a fast, TestClient-
    based check of THIS application's own logging (never Uvicorn's access
    log, which TestClient never exercises at all — it talks to the ASGI
    app in-process, with no HTTP wire protocol/access-log call site
    involved)."""
    from starlette.testclient import TestClient

    with _capturing_all_loggers() as stream:
        client = TestClient(create_app())
        response = client.get(
            "/api/auth/github/callback",
            params={"code": FAKE_CODE, "state": FAKE_STATE},
            follow_redirects=False,
        )
        assert response.status_code == 400

    log_output = stream.getvalue()
    assert FAKE_CODE not in log_output
    assert FAKE_STATE not in log_output
