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
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

import web_config
from web.routes import router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from app.auth_session import apply_startup_posture

    await apply_startup_posture(requested_secure=web_config.COOKIE_SECURE)

    yield

    from db.engine import close_db

    await asyncio.to_thread(close_db)


def create_app() -> FastAPI:
    app = FastAPI(title="Python Tutor Bot — Web API", lifespan=_lifespan)
    app.include_router(router)
    return app
