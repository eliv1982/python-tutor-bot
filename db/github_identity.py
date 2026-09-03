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
from typing import Callable, Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import GithubAccount, GithubUnlinkTombstone, User


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


def resolve_or_create_user_by_github_id_for_oauth_sync(
    *,
    github_user_id: int,
    auth_generation: int,
    _test_hook_after_advisory_lock: Optional[Callable[[], None]] = None,
) -> Optional[uuid.UUID]:
    """
    Generation-aware GitHub identity resolution (Stage 6C corrective pass,
    independent-audit MAJOR 1) — the ONLY resolution path
    web/github_oauth.py's callback is allowed to use once it has a
    verified GitHub identity and a claimed OAuth transaction's
    `auth_generation` in hand. Every OTHER caller of "resolve or create a
    canonical user for a GitHub id" (there are none today outside the
    OAuth callback) must keep using
    resolve_or_create_user_by_github_id_sync() above unchanged — this
    function exists specifically to avoid silently inventing a generation
    of 0 for a caller that never actually captured one, which would
    quietly defeat the whole staleness check.

    Closes a real race: an OAuth callback that has already exchanged its
    code and fetched a verified GitHub identity, but has not yet reached
    identity resolution, could otherwise "undo" a concurrent unlink of the
    SAME GitHub identity by recreating (or re-attaching to) a mapping the
    unlink just tore down — see db/telegram_link.py's module docstring and
    unlink_github_sync() for the other half of this protocol.

    Validates both inputs defensively (the database CHECK constraints on
    `github_oauth_transactions.auth_generation`/
    `github_unlink_tombstones.unlink_generation` are the authoritative
    backstop; this is fail-fast for an obviously-wrong caller, never a
    substitute for them): `auth_generation` must be non-negative — a
    negative value could never have been legitimately captured by
    db.oauth_transactions.create_sync() and would make the staleness
    comparison below meaningless.

    Serialization: acquires the EXACT SAME per-GitHub-id advisory lock
    (`pg_advisory_xact_lock(-github_user_id)`) resolve_or_create_user_by_
    github_id_sync() above already uses, and db.telegram_link.
    unlink_github_sync() also acquires (in that exact order — advisory
    lock strictly before any `github_accounts`/`users` row lock) before
    writing a tombstone for this same id. Because both sides always take
    this lock FIRST, whichever of {this resolver, a concurrent unlink of
    the same GitHub id} reaches it first runs to completion (commit,
    releasing the lock) before the other can even read the tombstone —
    there is no window where this resolver could read a tombstone that is
    concurrently being written for the SAME id (it either sees the
    tombstone fully committed, or the unlink is serialized to run after
    this resolver's own commit).

    Reads the durable tombstone (GithubUnlinkTombstone) for
    `github_user_id`, if any, and rejects (returns None — creates NO user,
    NO GitHub mapping, NO session; the caller must map this to a generic
    "login must be restarted" response and mint no session) whenever
    `tombstone.unlink_generation > auth_generation` — i.e. this GitHub
    identity has been unlinked at some point AFTER this OAuth transaction
    was created. This check runs and can reject EVEN WHEN a
    `github_accounts` mapping already (again) exists for `github_user_id`
    (Section D of the corrective pass): a stale callback must never attach
    to a mapping a NEWER login already recreated, so the generation
    comparison always happens before the existing-mapping fast path below,
    never only when no mapping exists.

    Otherwise, resolves or creates the canonical user/mapping using the
    EXACT SAME concurrency-safe existing/create logic
    resolve_or_create_user_by_github_id_sync() above already implements
    (read under the advisory lock; create-and-flush-then-map on a miss),
    and commits atomically.
    """
    if auth_generation < 0:
        raise ValueError(f"auth_generation must be non-negative, got {auth_generation}")

    with Session(get_sync_engine()) as session:
        session.execute(
            text("SELECT pg_advisory_xact_lock(CAST(:key AS bigint))"), {"key": -github_user_id}
        )
        if _test_hook_after_advisory_lock is not None:
            _test_hook_after_advisory_lock()

        tombstone_generation = session.execute(
            select(GithubUnlinkTombstone.unlink_generation).where(
                GithubUnlinkTombstone.github_user_id == github_user_id
            )
        ).scalar_one_or_none()
        if tombstone_generation is not None and tombstone_generation > auth_generation:
            session.rollback()
            return None

        existing = session.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_user_id)
        ).scalar_one_or_none()
        if existing is not None:
            session.commit()
            return existing

        new_user_id = uuid.uuid4()
        session.add(User(id=new_user_id))
        session.flush()
        session.add(GithubAccount(github_user_id=github_user_id, user_id=new_user_id))
        session.commit()
        return new_user_id
