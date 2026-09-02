"""
Canonical identity resolution for the GitHub OAuth provider (Stage 6B) —
mirrors db/identity.py's Telegram equivalent exactly: GitHub is only ever
an EXTERNAL identity, never itself the primary key of anything else in
this schema (see db/models.py's User docstring). app/github_identity.py
wraps this in asyncio.to_thread() the same way app/identity.py wraps
db/identity.py's Telegram functions — see db/engine.py's module docstring
for why DB access here is sync-in-thread rather than a native async
driver.

resolve_or_create_user_by_github_id_sync() is called ONLY after
web/github_oauth.py's callback has already completed a successful GitHub
token exchange and a successful, validated `GET /user` call — this
function is never itself an authorization decision, it only maps an
already-authenticated GitHub identity to the application's canonical
UUID, exactly mirroring db/identity.py's own docstring on the same point
for Telegram.

Stage 6C boundary: this module NEVER merges a GitHub identity with an
existing Telegram-backed canonical user — not by email, not by username,
not by any heuristic. A human who already has a Telegram-backed canonical
user and signs in with GitHub for the first time intentionally ends up
with a SECOND, separate canonical user until Stage 6C's explicit linking
ships. See web/github_oauth.py's own module docstring for the full
rationale.
"""

import uuid
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import GithubAccount, User


def lookup_user_by_github_id_sync(github_user_id: int) -> Optional[uuid.UUID]:
    """
    Read-only lookup: the internal user UUID already mapped to
    `github_user_id`, or None if no mapping exists yet. Never creates
    anything — mirrors db.identity.lookup_user_by_telegram_id_sync()'s own
    "safe to call for an id that must not be silently onboarded" contract.
    """
    with Session(get_sync_engine()) as session:
        return session.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_user_id)
        ).scalar_one_or_none()


def resolve_or_create_user_by_github_id_sync(github_user_id: int) -> uuid.UUID:
    """
    Resolve a GitHub numeric user id (the stable, durable external
    subject — never the login/username, which can change, and never
    email) to its stable internal user UUID, creating both the user and
    the mapping on first use.

    Race-safe via pg_advisory_xact_lock(-github_user_id) — the same
    pg_advisory_xact_lock pattern
    db.identity.resolve_or_create_user_by_telegram_id_sync() uses for
    Telegram ids, deliberately NEGATED here so the two providers can never
    collide in PostgreSQL's single 64-bit advisory-lock key space:
    Telegram numeric user ids are always positive and GitHub numeric user
    ids are always positive, so negating one of the two providers' keys
    makes the two lock namespaces disjoint by construction — no hashing,
    no ambiguity, no possibility of an unrelated Telegram/GitHub id pair
    ever accidentally serializing against each other. The lock is released
    automatically at COMMIT/ROLLBACK (never leaked by a crash mid-
    transaction), and — like the Telegram version — leaves no orphan
    `users` row on the losing side of a race.
    """
    with Session(get_sync_engine()) as session:
        session.execute(
            text("SELECT pg_advisory_xact_lock(CAST(:key AS bigint))"), {"key": -github_user_id}
        )
        existing = session.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_user_id)
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            return row

        new_user_id = uuid.uuid4()
        session.add(User(id=new_user_id))
        # See db.identity.resolve_or_create_user_by_telegram_id_sync()'s
        # own comment on this exact flush() — no relationship() is
        # declared between User/GithubAccount either, so the unit-of-work
        # flush needs this explicit nudge to insert `users` before
        # `github_accounts` within the same transaction.
        session.flush()
        session.add(GithubAccount(github_user_id=github_user_id, user_id=new_user_id))
        session.commit()
        return new_user_id
