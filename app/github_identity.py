"""
Canonical identity boundary for the GitHub OAuth provider (Stage 6B) —
mirrors app/identity.py's Telegram equivalent exactly.

resolve_user_uuid() is the ONLY place a GitHub numeric user id is turned
into the application's canonical internal UUID. web/github_oauth.py's
callback calls it exactly once, immediately AFTER a successful GitHub
token exchange and a successful, validated `GET /user` call — never
before. This function is never itself an authorization decision: GitHub
having authenticated the browser IS the authorization decision (unlike
the Telegram adapter's separate allowlist gate), and by the time this is
called, that has already happened.

A thin asyncio.to_thread() wrapper around the sync db.github_identity
call — see db/engine.py's module docstring for why DB access here is
sync-in-thread rather than a native async driver, and
app/identity.py's own docstring for the same offload idiom already used
for Telegram identity resolution.
"""

import asyncio
import uuid

import db.github_identity as db_github_identity


async def resolve_user_uuid(github_user_id: int) -> uuid.UUID:
    return await asyncio.to_thread(db_github_identity.resolve_or_create_user_by_github_id_sync, github_user_id)
