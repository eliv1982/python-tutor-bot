"""
Canonical identity resolution (Stage 5C) — SYNC (see db/engine.py's
module docstring for why: psycopg async mode is incompatible with
Windows' default ProactorEventLoop). app/identity.py wraps this via
utils.helpers.submit_worker()/await_worker() so callers still `await` it
— see db/engine.py's own docstring (Stage 7A-3 unified-runtime corrective
pass) for why this is no longer a plain `asyncio.to_thread()`.

resolve_or_create_user_by_telegram_id_sync() is called by the Telegram
adapter (app/identity.py) ONLY AFTER utils.access_control.require_authorized()
has already let the request through — this function is never itself an
authorization decision, and an unallowed Telegram id must never reach it
(an unallowed id gaining an internal user merely by being resolved here
would silently weaken the fail-closed allowlist into "anyone who messages
once is now a known user").
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import TelegramAccount, User


@dataclass(frozen=True)
class UserRecord:
    """Minimal, safe-to-expose snapshot of a canonical `users` row — see
    get_user_by_id_sync() below."""
    id: uuid.UUID
    created_at: datetime


def get_user_by_id_sync(user_id: uuid.UUID) -> Optional[UserRecord]:
    """
    Plain read of the canonical `users` row by its internal UUID, or None
    if no such user exists. Adapter-agnostic (despite this module's other
    functions being Telegram-specific): used by the web adapter's
    authenticated "current user" endpoint via app/auth_session.py, and
    fine for any future adapter to reuse the same way. Deliberately
    returns only `id`/`created_at` — never a Telegram id or any other
    internal detail — mirroring db.documents.DocumentRecord's own
    "minimal, concrete snapshot" philosophy rather than exposing the raw
    ORM row.
    """
    with Session(get_sync_engine()) as session:
        row = session.get(User, user_id)
        if row is None:
            return None
        return UserRecord(id=row.id, created_at=row.created_at)


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


def has_telegram_account_sync(user_id: uuid.UUID) -> bool:
    """
    Plain existence check: does `user_id` have a `telegram_accounts` row?
    (Stage 6C) — used by web/routes.py's `/api/me` to expose a safe
    `telegram_linked: bool` field without ever revealing the Telegram
    numeric id itself. Read-only, no advisory lock needed: an ordinary
    "yes/no" read for a display field has none of resolve_or_create_user_by_
    telegram_id_sync()'s first-creation race to guard against.
    """
    with Session(get_sync_engine()) as session:
        return (
            session.execute(
                select(TelegramAccount.telegram_user_id).where(TelegramAccount.user_id == user_id).limit(1)
            ).first()
            is not None
        )


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
