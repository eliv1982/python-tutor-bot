"""
Canonical identity resolution (Stage 5C) — SYNC (see db/engine.py's
module docstring for why: psycopg async mode is incompatible with
Windows' default ProactorEventLoop). app/identity.py wraps this in
asyncio.to_thread() so callers still `await` it, matching the codebase's
existing idiom for blocking-I/O boundaries (rag/query.py, handlers/start.py).

resolve_or_create_user_by_telegram_id_sync() is called by the Telegram
adapter (app/identity.py) ONLY AFTER utils.access_control.require_authorized()
has already let the request through — this function is never itself an
authorization decision, and an unallowed Telegram id must never reach it
(an unallowed id gaining an internal user merely by being resolved here
would silently weaken the fail-closed allowlist into "anyone who messages
once is now a known user").
"""

import uuid
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import TelegramAccount, User


def lookup_user_by_telegram_id_sync(telegram_id: int) -> Optional[uuid.UUID]:
    """
    Read-only lookup: the internal user UUID already mapped to
    `telegram_id`, or None if no mapping exists yet. Never creates
    anything — unlike resolve_or_create_user_by_telegram_id_sync() below,
    this is safe to call for a telegram_id that must NOT be silently
    onboarded merely by being looked up.

    Stage 5C corrective pass: scripts/migrate_sidecars_v2_to_v3.py needs
    exactly this distinction for its fail-closed ownership rule — "if a
    mapping already exists, use it; otherwise consult the allowlist before
    ever creating one" — which resolve_or_create_user_by_telegram_id_sync()
    alone cannot express (it always creates on a miss).
    """
    with Session(get_sync_engine()) as session:
        return session.execute(
            select(TelegramAccount.user_id).where(TelegramAccount.telegram_user_id == telegram_id)
        ).scalar_one_or_none()


def resolve_or_create_user_by_telegram_id_sync(telegram_id: int) -> uuid.UUID:
    """
    Resolve a trusted Telegram numeric user id to its stable internal user
    UUID, creating both the user and the mapping on first use.

    Race-safe via pg_advisory_xact_lock(telegram_id): Telegram ids already
    fit Postgres's bigint (the lock's native key type, no hashing needed).
    The lock serializes ONLY concurrent resolution attempts for this exact
    telegram_id (e.g. two racing worker threads/requests for the same
    not-yet-seen id) — unrelated ids proceed fully in parallel — and is
    released automatically at COMMIT/ROLLBACK, so it can never be leaked
    by a crash mid-transaction. Simpler and leaves no orphan `users` row
    on the losing side of a race, unlike an INSERT ... ON CONFLICT DO
    NOTHING approach (which would need a rollback-and-reread dance to
    avoid leaving an orphaned `users` insert behind for the loser).
    """
    with Session(get_sync_engine()) as session:
        session.execute(text("SELECT pg_advisory_xact_lock(CAST(:tid AS bigint))"), {"tid": telegram_id})
        existing = session.execute(
            select(TelegramAccount.user_id).where(TelegramAccount.telegram_user_id == telegram_id)
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            return row

        new_user_id = uuid.uuid4()
        session.add(User(id=new_user_id))
        # No relationship() is declared between User/TelegramAccount (this
        # schema deliberately uses plain FK columns, not an ORM object
        # graph) — without one, the unit-of-work flush has no dependency
        # edge telling it TelegramAccount.user_id's raw value depends on
        # this User row, and does NOT reliably order the two INSERTs by
        # table-level foreign key alone. An explicit flush() here forces
        # the `users` row to be inserted (visible within this same,
        # still-open transaction) before `telegram_accounts` is added,
        # avoiding a FK-violation race against SQLAlchemy's own flush
        # ordering — confirmed necessary against a real Postgres instance.
        session.flush()
        session.add(TelegramAccount(telegram_user_id=telegram_id, user_id=new_user_id))
        session.commit()
        return new_user_id
        # session.commit() here releases the advisory lock as part of the
        # same COMMIT (both the new `users` row and the `telegram_accounts`
        # row land atomically); the `with Session(...)` block's own close
        # on exit never re-triggers a rollback after a successful commit.
