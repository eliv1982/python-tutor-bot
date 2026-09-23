"""
FastAPI application factory (Stage 6A) — the web adapter's composition
root, playing the same role bot.py/main.py play for the Telegram adapter.

create_app() performs NO I/O: no database engine construction, no network
calls. It only assembles routes (and registers the lifespan hook below,
which performs no I/O merely by being registered — it only runs later,
when something actually starts the app, e.g. uvicorn or `with
TestClient(...)`). get_sync_engine() (db/engine.py) is still constructed
lazily, on its own first real call from inside a request, exactly like
every other adapter in this codebase. This is what lets "the application
can be constructed offline" hold as a plain, unconditional fact,
independent of whether a database is reachable.

Stage 6B: web.github_oauth (GitHub login) is imported unconditionally
below, the same as web.routes — GitHub login is a core route of this
adapter, not an optional feature flag. That import transitively requires
github_oauth_config.py's checks to pass (GITHUB_CLIENT_ID/SECRET/
REDIRECT_URI configured and valid) — the same fail-closed posture
web_config.py's SESSION_SECRET_KEY check already established for every
route in this module, including ones (like /healthz) that don't
themselves touch either credential. This has no effect on the Telegram
adapter: main.py/bot.py never import anything under web/.

Deliberately no module-level `app = create_app()` singleton (unlike
bot.py's eager `bot = AsyncTeleBot(...)`): a FastAPI application built
purely from route registration has no equivalent reason to be a singleton,
and an explicit factory keeps every test constructing its own isolated
instance, exactly like Telegram's own tests build their own bot/update
fixtures rather than sharing process-wide state.

Lifespan startup (independent-audit corrective pass #2, Major 1, hardened
by corrective pass #3): transactionally establishes this process's
posture as the database's AUTHORITATIVE current posture and revokes every
still-active web_sessions row bound to the other posture
(app.auth_session.apply_startup_posture()), BEFORE the app begins serving
any request. See db/models.py's WebSessionPolicy docstring and
db/auth_sessions.py's apply_startup_posture_sync()/create_sync() for the
full transactional protocol — pass #2's version merely ran an
unsynchronized UPDATE here, which raced against a concurrent create_sync()
call from an old, still-running opposite-posture process; pass #3 closes
that by making session creation itself acquire the exact same database
lock this call does, so the two can never interleave unsafely regardless
of which process reaches it first. See
tests/test_stage6a_corrective2_posture_transition.py for the real-Postgres,
real-FastAPI sequential regression proof (the original audit's exact
secure->insecure->secure reproduction) and
tests/test_stage6a_corrective3_policy_race.py for the real-Postgres,
real-thread proof of the concurrent-process race. Reads
web_config.COOKIE_SECURE HERE, at the web-adapter boundary — never inside
app/auth_session.py itself (see that module's own docstring on why).

Lifespan shutdown (independent-audit corrective pass #1, minor finding #3):
disposes the shared sync DB engine (db.engine.close_db()) on ASGI
shutdown — the web adapter's own equivalent of main.py's shutdown_bot()
call for the Telegram adapter. Safe and adapter-owned: web_main.py is
documented as its own separate OS process from main.py (see that module's
own docstring — "running this does not start the Telegram bot, and
running main.py does not start this web server"), so disposing the engine
here can never race or interfere with a Telegram process's own engine,
which it does not share. get_sync_engine() reconstructs a fresh Engine
lazily on the next call regardless (see db/engine.py), so this is a
clean, idempotent, "nothing left running past shutdown" cleanup, not a
destructive one. See tests/test_stage6a_corrective1_lifecycle.py for the
proof.

No CORS middleware is registered — Stage 6A's CSRF defense (web/csrf.py)
relies on the browser's Same-Origin Policy to prevent a cross-site page
from ever reading the CSRF cookie/deriving a valid X-CSRF-Token itself
(see that module's own docstring). A FUTURE permissive/credentialed CORS
policy (out of scope here — Stage 7+) must never expose the CSRF cookie
value to another origin or otherwise let an attacker origin construct a
valid authenticated state-changing request; whoever adds CORS here must
preserve that invariant explicitly, not merely default `allow_credentials`
to true alongside a wide `allow_origins`.

Stage 7A-2: web.body_limit.RequestBodyLimitMiddleware caps the ACTUAL
request body of the small mutating JSON routes (POST /api/chat, PATCH
/api/settings — and only those) at web_config.MAX_JSON_BODY_BYTES, before
FastAPI parses anything. FastAPI's default RequestValidationError response
(which can echo submitted input back) is replaced by a sanitized, fixed
422 {"detail": "Invalid request"}; only the error count is logged, never
the Pydantic error payload.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import web_config
from web.body_limit import RequestBodyLimitMiddleware
from web.github_oauth import router as github_oauth_router
from web.routes import INVALID_REQUEST_DETAIL, router

logger = logging.getLogger(__name__)

# Stage 7A-2: the only routes web_config.MAX_JSON_BODY_BYTES applies to.
JSON_BODY_LIMITED_ROUTES = frozenset({("POST", "/api/chat"), ("PATCH", "/api/settings")})


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from app.auth_session import apply_startup_posture

    await apply_startup_posture(requested_secure=web_config.COOKIE_SECURE)

    yield

    from db.engine import close_db

    await asyncio.to_thread(close_db)


def _disable_uvicorn_access_logging() -> None:
    """Defense-in-depth for MAJOR 1 (Stage 6B independent-audit corrective
    pass #1) — see web_main.py's own docstring for the PRIMARY guarantee
    (`uvicorn.run(..., access_log=False)`), which stops Uvicorn's access-
    log call site entirely regardless of logger configuration. This
    additionally silences the `"uvicorn.access"` logger itself, so a
    deployment that launches this exact ASGI app through the bare
    `uvicorn` CLI/config (e.g. `uvicorn web.app:create_app --factory`,
    where `access_log` defaults to True and a deployer may not think to
    add an equivalent `--no-access-log` flag) still never has this
    application's own callback query strings (`code`/`state`) written to
    an access log.

    Safe to call unconditionally from create_app(): Uvicorn installs its
    OWN default logging configuration (which (re)creates/configures the
    `"uvicorn.access"` logger) during `Config.__init__()` — which always
    runs BEFORE `Config.load()` imports and calls this factory — so by the
    time create_app() runs, Uvicorn's own setup has already happened and
    this reliably overrides it, regardless of launch method. Only
    `"uvicorn.access"` is touched; `"uvicorn"`/`"uvicorn.error"` (startup/
    crash diagnostics) and this application's own `"bot"` logger
    (utils/logging.py) are left completely alone — Do NOT disable
    application security/error logging here.
    """
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.disabled = True
    access_logger.handlers = []
    access_logger.propagate = False


async def _sanitized_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Never log or return exc.errors()/exc.body — both can carry submitted
    # input. The error count is safe metadata.
    logger.info(
        "request validation failed | method=%s, path=%s, error_count=%s",
        request.method, request.url.path, len(exc.errors()),
    )
    return JSONResponse(
        {"detail": INVALID_REQUEST_DETAIL}, status_code=422, headers={"Cache-Control": "no-store"}
    )


def create_app() -> FastAPI:
    _disable_uvicorn_access_logging()
    app = FastAPI(title="Python Tutor Bot — Web API", lifespan=_lifespan)
    app.add_exception_handler(RequestValidationError, _sanitized_validation_error_handler)
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=web_config.MAX_JSON_BODY_BYTES,
        limited_routes=JSON_BODY_LIMITED_ROUTES,
    )
    app.include_router(router)
    app.include_router(github_oauth_router)
    return app
