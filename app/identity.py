"""
Canonical identity boundary for the Telegram adapter (Stage 5C).

resolve_user_uuid() is the ONLY place a Telegram numeric id is turned into
the application's canonical internal UUID. Handlers call it exactly once,
immediately AFTER utils.access_control.require_authorized() has already
let the request through — never before, and never as a substitute for
that check. Database identity and Telegram authorization are different
concerns: an unallowed Telegram id must never reach this function, and
this function is never itself an authorization decision (see
db/identity.py's own docstring for the same point from the persistence
side).

A thin async wrapper around the sync db.identity call, offloaded via
utils.helpers.submit_worker()/await_worker() — see db/engine.py's module
docstring for why DB access here is sync-in-thread rather than a native
async driver (psycopg's async mode is incompatible with Windows' default
ProactorEventLoop, which this application's own asyncio.run() uses), and
for why this is submit_worker()/await_worker() rather than a plain
asyncio.to_thread() (Stage 7A-3 unified-runtime corrective pass:
resolve_user_uuid() runs on every Telegram update, so its Task must not be
able to report itself "settled" to service_main.py's shutdown sequence
while its worker thread is still using the shared DB engine).
"""

import uuid

import db.identity as db_identity
from utils.helpers import await_worker, submit_worker


async def resolve_user_uuid(telegram_user_id: int) -> uuid.UUID:
    return await await_worker(submit_worker(db_identity.resolve_or_create_user_by_telegram_id_sync, telegram_user_id))


async def is_telegram_linked(user_id: uuid.UUID) -> bool:
    """Stage 6C, Section M — thin async wrapper around
    db.identity.has_telegram_account_sync(), used by web/routes.py's
    `/api/me` to expose a safe boolean without ever leaking the Telegram
    numeric id itself."""
    return await await_worker(submit_worker(db_identity.has_telegram_account_sync, user_id))
