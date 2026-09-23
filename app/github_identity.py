"""
Canonical identity boundary for the GitHub OAuth provider (Stage 6B) —
mirrors app/identity.py's Telegram equivalent exactly.

Two distinct resolution functions live here, for two distinct callers:

  - resolve_user_uuid() is the ORDINARY, non-generation-aware GitHub id ->
    canonical UUID lookup/create — creates a `github_accounts` row (and
    its canonical user) on first sight, exactly like
    app/identity.py's Telegram equivalent. It performs no staleness check
    of any kind and is never itself an authorization decision.
  - resolve_user_uuid_for_oauth() (Stage 6C corrective pass, independent-
    audit MAJOR 1, below) is the ONLY resolution call
    web/github_oauth.py's callback may use once it holds a verified GitHub
    identity and a claimed OAuth transaction's `auth_generation` — see its
    own docstring for the generation/tombstone protocol this enforces
    (never resolve_user_uuid() plain, which the callback used pre-Stage
    6C and does not reject a stale, already-in-flight login against a
    concurrent unlink).

Neither function is itself an authorization decision: GitHub having
authenticated the browser IS the authorization decision (unlike the
Telegram adapter's separate allowlist gate), and by the time either is
called, that has already happened.

A thin async wrapper around the sync db.github_identity calls, offloaded
via utils.helpers.submit_worker()/await_worker() — see db/engine.py's
module docstring for why DB access here is sync-in-thread rather than a
native async driver (and why this is submit_worker()/await_worker()
rather than a plain asyncio.to_thread() as of the Stage 7A-3
unified-runtime corrective pass), and app/identity.py's own docstring for
the same offload idiom already used for Telegram identity resolution.
"""

import uuid
from typing import Optional

import db.github_identity as db_github_identity
from utils.helpers import await_worker, submit_worker


async def resolve_user_uuid(github_user_id: int) -> uuid.UUID:
    return await await_worker(submit_worker(db_github_identity.resolve_or_create_user_by_github_id_sync, github_user_id))


async def resolve_user_uuid_for_oauth(*, github_user_id: int, auth_generation: int) -> Optional[uuid.UUID]:
    """
    Generation-aware wrapper (Stage 6C corrective pass, independent-audit
    MAJOR 1) — the ONLY resolution call web/github_oauth.py's callback may
    use once it holds a verified GitHub identity and a claimed OAuth
    transaction's `auth_generation`. Returns None if the transaction is
    stale relative to a later unlink of this same GitHub identity — see
    db.github_identity.resolve_or_create_user_by_github_id_for_oauth_sync()
    for the full protocol this delegates to. The callback must treat None
    exactly like any other terminal callback failure: a generic "login
    must be restarted" response, the OAuth-binding cookie cleared, and no
    session minted.
    """
    return await await_worker(submit_worker(
        db_github_identity.resolve_or_create_user_by_github_id_for_oauth_sync,
        github_user_id=github_user_id,
        auth_generation=auth_generation,
    ))
