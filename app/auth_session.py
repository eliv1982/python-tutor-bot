"""
Server-side web-session lifecycle (Stage 6A) — the application-layer
boundary between the FastAPI adapter (web/*.py) and db.auth_sessions'
SYNC persistence, exactly mirroring the existing split app/identity.py
(Telegram) / app/session.py (durable preferences) already use over
db/identity.py / db/preferences.py (see db/engine.py's module docstring
for why DB access is sync-in-thread rather than a native async driver).

Deliberately its own module, not folded into either existing one:
app/identity.py's whole contract is scoped to "a Telegram numeric id ->
internal UUID" (see its own header) and app/session.py's `UserSession` is
ephemeral chat history plus durable mode/voice preferences — an HTTP
authentication session is a third, distinct concern from both, with its
own lifecycle (create/resolve/revoke/expire) that has nothing to do with
Telegram or tutoring state.

Session token generation (CSPRNG) and hashing (SHA-256) both live HERE, in
the application layer — never in db/auth_sessions.py, which only ever
sees/stores a digest (see that module's own docstring). The RAW token
exists in memory only transiently: inside create_session()'s return value
(handed straight to the FastAPI route, which places it in a Set-Cookie
header and never logs it) and inside resolve_session_user_id()/
revoke_session()'s own stack frame while it is hashed. It is never
persisted anywhere.

create_session() is intentionally NOT wired to any HTTP endpoint in Stage
6A (there is no login flow yet — GitHub OAuth is Stage 6B). It exists as a
stable seam: Stage 6B's OAuth callback will call this exact function after
verifying the provider identity, and tests today call it directly the same
way, standing in for that not-yet-built callback.

Imports session_config.py (TTL policy only), never web_config.py —
independent-audit corrective pass #1, minor finding #1: this module has no
need for SESSION_SECRET_KEY/COOKIE_SECURE (web_config.py's own concerns),
so it must not require them merely to import. See session_config.py's own
docstring for the full rationale, and
tests/test_stage6a_corrective1_lifecycle.py for the import-isolation
regression proof.

Cookie posture (Stage 6A independent-audit corrective pass #2, Major 1):
create_session()/resolve_session_user_id() take an explicit
`issued_secure`/`expected_secure` boolean rather than importing
web_config.COOKIE_SECURE themselves — same import-isolation rationale as
the TTL split above, extended to this new parameter: the CALLER (the web
adapter, which already legitimately imports web_config) passes the current
posture in explicitly. See db/models.py's WebSession docstring and
db/auth_sessions.py's get_active_sync()/apply_startup_posture_sync() for
the full mechanism.

Stage 6A independent-audit corrective pass #3: create_session() can now
raise StalePostureError (re-exported below from db.auth_sessions, never
caught/swallowed here) — see db/models.py's WebSessionPolicy docstring and
db/auth_sessions.py's create_sync()/apply_startup_posture_sync() for the
transactional protocol this reflects. A caller (Stage 6B's future OAuth
callback) that receives it should treat it exactly like any other failed
mint attempt: this process's own web_config.COOKIE_SECURE no longer
matches the database's authoritative posture, so no session was created.
"""

import asyncio
import base64
import binascii
import hashlib
import re
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
import session_config

# Re-exported so Stage 6B (and tests) can write
# `except auth_session.StalePostureError:` against this module's own
# stable seam, never needing to import db.auth_sessions directly merely
# to catch it.
StalePostureError = db_auth_sessions.StalePostureError

# 256 bits of CSPRNG entropy (secrets.token_urlsafe uses os.urandom) —
# comfortably beyond brute-force feasibility for a bearer credential.
_TOKEN_BYTES = 32

# secrets.token_urlsafe(n) encodes exactly n bytes as unpadded base64url:
# ceil(n * 4 / 3) characters from [A-Za-z0-9_-]. For _TOKEN_BYTES=32 that is
# always exactly 43 characters — base64 length is a function of the INPUT
# BYTE LENGTH only, never of the bytes' values, so this is exact, not a
# heuristic estimate. Independent-audit corrective pass #1, required
# hardening: a raw bearer credential that doesn't match this exact shape is
# rejected BEFORE it is hashed or reaches the database — an attacker-
# controlled cookie value must never drive an unbounded-cost hash/DB
# lookup. See tests/test_stage6a_corrective1_repr_and_token_bounds.py.
_TOKEN_LENGTH = -(-(_TOKEN_BYTES * 4) // 3)  # ceil division, no float
_TOKEN_SHAPE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _is_canonical_token(raw_token: str) -> bool:
    """Independent-audit corrective pass #2, minor finding: the pass #1
    version only checked length + base64url alphabet, which — for a fixed
    43-character length — accepts strings whose FINAL character encodes an
    "impossible" quantum: 32 bytes (256 bits) only need 4 of the last
    base64 character's 6 bits, so a canonical encoding always has that
    character's low 2 bits zero. A 43-char, alphabet-valid string with
    those bits set decodes successfully (base64 decoders discard unused
    padding bits rather than rejecting them) to the SAME 32 bytes another,
    different 43-char string would also decode to — i.e. multiple
    non-equal strings would hash/look-up as if they were the one real
    issued token, which is not the intended semantics of "this exact
    output of secrets.token_urlsafe(32)". Fixed with a canonical
    decode-and-re-encode round trip: `raw_token` is accepted only if
    base64url-decoding it (with restored '=' padding) yields exactly 32
    bytes AND re-encoding those 32 bytes reproduces `raw_token` exactly.
    Any other 43-char alphabet-valid string decodes to different bytes or
    fails the round-trip and is rejected. See
    tests/test_stage6a_corrective2_secret_and_token_validation.py."""
    if len(raw_token) != _TOKEN_LENGTH or not _TOKEN_SHAPE_RE.fullmatch(raw_token):
        return False
    padding = "=" * (-len(raw_token) % 4)
    try:
        decoded = base64.urlsafe_b64decode(raw_token + padding)
    except (binascii.Error, ValueError):
        return False
    if len(decoded) != _TOKEN_BYTES:
        return False
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    return canonical == raw_token


def _hash_token(raw_token: str) -> bytes:
    return hashlib.sha256(raw_token.encode("utf-8")).digest()


@dataclass(frozen=True)
class IssuedSession:
    """Returned ONLY by create_session() — the one place the raw bearer
    token is ever handed to a caller.

    `raw_token` is excluded from the dataclass-generated `repr()`
    (`field(repr=False)`) — independent-audit corrective pass #1, Major 1:
    the raw bearer credential must never appear in a repr/debug/logging
    path, even incidentally (a caught-exception log, a debugger, a stray
    `logger.debug(session)`). It is NOT removed from the object itself —
    Stage 6B's OAuth callback still needs the actual value to set the
    browser cookie; only its textual REPRESENTATION is suppressed. See
    tests/test_stage6a_corrective1_repr_and_token_bounds.py."""
    raw_token: str = field(repr=False)
    expires_at: datetime


@dataclass(frozen=True)
class UserProfile:
    """Safe, session-facing snapshot of a canonical user — see
    db.identity.UserRecord's own docstring for what is deliberately
    excluded (Telegram id, provider credentials, internal persistence
    detail)."""
    id: uuid.UUID
    created_at: datetime


async def create_session(user_id: uuid.UUID, *, issued_secure: bool) -> IssuedSession:
    """
    Issue a brand-new server-side session for `user_id`. See this module's
    own docstring for why nothing in Stage 6A calls this over HTTP yet.

    `issued_secure` (Stage 6A independent-audit corrective pass #2,
    Major 1) is the cookie posture (web_config.COOKIE_SECURE) in effect
    right now, supplied explicitly by the caller (Stage 6B's OAuth
    callback will pass it the same way tests do today) — see this
    module's own docstring for why it is a parameter here rather than an
    import of web_config. Persisted verbatim onto the new row; see
    db/models.py's WebSession docstring for what it's for.

    Raises StalePostureError (Stage 6A independent-audit corrective pass
    #3) if `issued_secure` no longer matches the database's authoritative
    posture at the moment db.auth_sessions.create_sync()'s transaction
    acquires its lock — see that function's own docstring for the full
    transactional protocol. No row is inserted; this is a normal, expected
    outcome for a stale process, not an internal error.

    `expires_at` is a timezone-AWARE UTC datetime, persisted as-is (no
    tzinfo stripping) into `web_sessions.expires_at`, a `TIMESTAMP WITH
    TIME ZONE` column (see db/models.py's WebSession docstring —
    independent-audit corrective pass #1, Blocker 1: a naive timestamp
    compared against PostgreSQL's `now()` is only timezone-safe when the
    PostgreSQL session's TimeZone GUC happens to be UTC, which the auditor
    proved is not a safe assumption to depend on). The same aware value is
    also what the returned IssuedSession carries — Starlette's
    Response.set_cookie() requires an aware UTC datetime for its `expires`
    cookie attribute (a naive one raises ValueError, see web/cookies.py) —
    so, unlike before this pass, caller and persistence now see the exact
    same value, not two independently-derived copies.

    Note on cancellation: if the awaiting caller is cancelled after the
    worker thread (asyncio.to_thread below) has already committed the INSERT
    but before this coroutine resumes and returns, the new row becomes
    unreachable garbage — nobody ever received `raw_token`, so nothing can
    ever resolve/revoke it; it simply sits until `expires_at` (now
    bounded — see session_config.py's upper bound) passes. This is the same
    accepted risk category db/engine.py's own module docstring documents
    for db/identity.py/db/preferences.py's plain (unshielded)
    asyncio.to_thread() usage ("at worst nothing was created/updated and
    the caller's request simply fails and can be retried") — deliberately
    NOT given the heavier submit_worker()/await_worker() shielding
    app/documents.py uses, since that mechanism exists to protect
    multi-step physical-file/sidecar/catalog consistency that a single
    INSERT here has no equivalent of, and adding cross-thread
    cancellation/reconciliation machinery for a bounded, unreachable,
    already-expiring row would be disproportionate complexity for no
    correctness gain.
    """
    raw_token = secrets.token_urlsafe(_TOKEN_BYTES)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=session_config.SESSION_TTL_SECONDS)
    await asyncio.to_thread(
        db_auth_sessions.create_sync,
        token_hash=_hash_token(raw_token),
        user_id=user_id,
        issued_secure=issued_secure,
        expires_at=expires_at,
    )
    return IssuedSession(raw_token=raw_token, expires_at=expires_at)


async def create_session_for_github(github_user_id: int, *, issued_secure: bool) -> Optional[IssuedSession]:
    """
    GitHub-backed session issuance (Stage 6C, Section K) — the async
    wrapper around db.auth_sessions.create_for_github_sync(), mirroring
    create_session() above's own asyncio.to_thread() offload idiom. Unlike
    create_session(), this does not take an already-resolved `user_id`: it
    re-resolves `github_user_id` -> canonical UUID FRESH, under a lock, in
    the SAME transaction as the session insert, so the session is always
    bound to whichever UUID is CURRENTLY mapped, never a value the caller
    resolved moments earlier through a separate transaction (see that
    function's own docstring for the exact race this closes — a Stage 6C
    merge can move a GitHub mapping onto a different, Telegram-backed UUID
    between an earlier resolve and session issuance).

    Returns None if `github_user_id` has no current mapping at all (fail
    closed — no session created; Section K: "returns a fail-closed result
    if the mapping disappeared"). The caller (web/github_oauth.py) must
    treat None exactly like a failed login attempt: no cookie is set, and
    it is the caller's job to have already ensured a mapping exists via
    app.github_identity.resolve_user_uuid() before calling this — this
    function itself never creates one (Section K: "never recreates a
    GitHub mapping").

    Raises StalePostureError exactly like create_session() (unchanged
    semantics, same re-exported exception) if this process's own cookie
    posture is no longer authoritative.
    """
    raw_token = secrets.token_urlsafe(_TOKEN_BYTES)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=session_config.SESSION_TTL_SECONDS)
    user_id = await asyncio.to_thread(
        db_auth_sessions.create_for_github_sync,
        github_user_id=github_user_id,
        token_hash=_hash_token(raw_token),
        issued_secure=issued_secure,
        expires_at=expires_at,
    )
    if user_id is None:
        return None
    return IssuedSession(raw_token=raw_token, expires_at=expires_at)


async def resolve_session_user_id(raw_token: Optional[str], *, expected_secure: bool) -> Optional[uuid.UUID]:
    """Fail closed for anything that isn't a plausible, currently-active,
    CURRENT-POSTURE session token: a missing/empty/non-canonical value
    never even reaches the database. `_is_canonical_token()`
    (independent-audit corrective pass #1/#2, required hardening) rejects
    anything that couldn't possibly be a real
    secrets.token_urlsafe(_TOKEN_BYTES) output — wrong length, a character
    outside the base64url alphabet, or a non-canonical encoding of some
    32-byte value — BEFORE hashing/DB lookup, so an oversized or malformed
    cookie value can never drive an unbounded-cost hash or a wasted
    database round trip.

    `expected_secure` (Stage 6A independent-audit corrective pass #2,
    Major 1) is the CURRENT cookie posture, supplied explicitly by the
    caller — see this module's own docstring. Passed straight through to
    db.auth_sessions.get_active_sync(); see that function's own docstring
    for the full posture-matching contract and why it alone is not the
    authoritative revival guarantee (apply_startup_posture() below,
    combined with create_sync()'s own transactional posture check, is).
    An unknown, expired, revoked, or wrong-posture token (still of
    canonical shape) all resolve to None alike."""
    if not raw_token or not _is_canonical_token(raw_token):
        return None
    record = await asyncio.to_thread(
        db_auth_sessions.get_active_sync, token_hash=_hash_token(raw_token), expected_secure=expected_secure
    )
    return record.user_id if record is not None else None


async def revoke_session(raw_token: Optional[str]) -> None:
    """Logout: idempotent no-op for a missing/unknown/non-canonical token —
    see db.auth_sessions.revoke_sync()'s own contract and
    resolve_session_user_id()'s docstring above for the same shape-check
    rationale. Deliberately posture-agnostic (no `expected_secure`
    parameter) — see db.auth_sessions.revoke_sync()'s own docstring for
    why revocation doesn't need one."""
    if not raw_token or not _is_canonical_token(raw_token):
        return
    await asyncio.to_thread(db_auth_sessions.revoke_sync, token_hash=_hash_token(raw_token))


async def apply_startup_posture(*, requested_secure: bool) -> int:
    """Transactional posture transition — thin async wrapper, called ONCE
    from web/app.py's FastAPI lifespan at every startup, BEFORE the app
    begins serving requests. See
    db.auth_sessions.apply_startup_posture_sync()'s own docstring for the
    full transactional protocol (Stage 6A independent-audit corrective
    pass #3 — this is what makes create_sync()'s posture check
    authoritative against a concurrent, cross-process transition, closing
    the race pass #2's simpler, unsynchronized version left open); this
    wrapper exists only to apply this module's usual asyncio.to_thread()
    offload convention. Returns the number of sessions revoked
    (informational)."""
    return await asyncio.to_thread(
        db_auth_sessions.apply_startup_posture_sync, requested_secure=requested_secure
    )


async def get_user_profile(user_id: uuid.UUID) -> Optional[UserProfile]:
    """Safe, session-facing snapshot of a canonical user for the web
    adapter's "current user" endpoint. None if the row is somehow gone
    (defensive — no user-deletion path exists yet, but a resolved session
    must never be trusted blindly into a profile that turns out not to
    exist)."""
    row = await asyncio.to_thread(db_identity.get_user_by_id_sync, user_id)
    if row is None:
        return None
    return UserProfile(id=row.id, created_at=row.created_at)
