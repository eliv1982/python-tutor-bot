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

A thin asyncio.to_thread() wrapper around the sync db.identity call — see
db/engine.py's module docstring for why DB access here is sync-in-thread
rather than a native async driver (psycopg's async mode is incompatible
with Windows' default ProactorEventLoop, which this application's own
asyncio.run() uses). This is the same offload idiom already used for
Qdrant reads (rag/query.py, handlers/start.py's /stats command).
"""

import asyncio
import uuid

import db.identity as db_identity


async def resolve_user_uuid(telegram_user_id: int) -> uuid.UUID:
    return await asyncio.to_thread(db_identity.resolve_or_create_user_by_telegram_id_sync, telegram_user_id)
