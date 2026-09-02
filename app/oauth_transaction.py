"""
Application-layer boundary for the GitHub OAuth transaction (state + PKCE)
lifecycle (Stage 6B) — mirrors app/auth_session.py's own split from
db/auth_sessions.py: token/state generation, hashing, and PKCE challenge
derivation all live HERE, in the application layer; db/oauth_transactions.py
only ever sees/stores a digest of `state` (never the raw value) plus the
(cleartext) PKCE verifier — see that module's own docstring for why the
verifier's cleartext storage is an acceptable trust boundary.

Imports github_oauth_config.py ONLY for OAUTH_TRANSACTION_TTL_SECONDS —
mirrors app/auth_session.py importing session_config.py rather than
web_config.py for the identical import-isolation reason: nothing else in
this module needs GITHUB_CLIENT_ID/SECRET/REDIRECT_URI.

The raw `state` and raw `code_verifier` values exist in memory only
transiently, exactly like app/auth_session.py's raw bearer token: `state`
is handed straight to the caller (web/github_oauth.py, which puts it in
the GitHub authorize redirect URL and a short-lived cookie — see that
module's own docstring for why) and `code_verifier` is recovered from the
database only inside claim_transaction()'s own return value, to be handed
straight to services/github_oauth_client.py's token exchange. Neither is
ever logged.
"""

import asyncio
import base64
import binascii
import hashlib
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import db.oauth_transactions as db_oauth_transactions
import github_oauth_config

# 256 bits of CSPRNG entropy for `state` — matches app/auth_session.py's
# own _TOKEN_BYTES rationale (comfortably beyond brute-force feasibility
# for a value an attacker could try to guess/replay).
_STATE_BYTES = 32

# secrets.token_urlsafe(n) encodes exactly n bytes as unpadded base64url:
# ceil(n * 4 / 3) characters — see app/auth_session.py's identical
# comment for _TOKEN_LENGTH, which this mirrors exactly for `state`.
_STATE_LENGTH = -(-(_STATE_BYTES * 4) // 3)
_STATE_SHAPE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# RFC 7636 requires a code_verifier of 43-128 characters from
# [A-Za-z0-9-._~]. secrets.token_urlsafe(48) yields exactly 64 base64url
# characters (a subset of that allowed alphabet), comfortably inside the
# required range with 384 bits of entropy — a fixed, RFC-compliant shape
# by construction, never a value that needs runtime length validation.
_VERIFIER_BYTES = 48


def _hash_state(raw_state: str) -> bytes:
    return hashlib.sha256(raw_state.encode("utf-8")).digest()


def is_canonical_state(raw_state: str) -> bool:
    """
    Same rationale and shape as app/auth_session._is_canonical_token():
    reject anything that couldn't possibly be a real
    secrets.token_urlsafe(_STATE_BYTES) output — wrong length, a character
    outside the base64url alphabet, a non-canonical encoding of some other
    32-byte value, or (Stage 6B independent-audit corrective pass #1,
    MINOR 4B) any non-ASCII/Unicode content at all — BEFORE hashing,
    touching the database, or ever being passed to `hmac.compare_digest()`
    (which raises TypeError for a non-ASCII `str` operand — web/
    github_oauth.py's callback calls this FIRST, on the raw query-string
    `state`, specifically to turn a would-be 500 into a safe 400 before
    that comparison ever runs). Public (no leading underscore): used both
    by claim_transaction() below and directly by web/github_oauth.py's
    callback for that same cheap pre-hmac/pre-DB shape check.

    An attacker-controlled `state` query parameter can therefore never
    drive an unbounded-cost hash/DB lookup, and two non-identical query
    strings can never be treated as the same transaction merely because
    they decode to the same underlying bytes.
    """
    if len(raw_state) != _STATE_LENGTH or not _STATE_SHAPE_RE.fullmatch(raw_state):
        return False
    padding = "=" * (-len(raw_state) % 4)
    try:
        decoded = base64.urlsafe_b64decode(raw_state + padding)
    except (binascii.Error, ValueError):
        return False
    if len(decoded) != _STATE_BYTES:
        return False
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    return canonical == raw_state


def code_challenge_for(code_verifier: str) -> str:
    """PKCE S256: BASE64URL_NO_PADDING(SHA256(code_verifier)) — RFC 7636
    section 4.2, exactly. Exposed as a standalone function so tests can
    validate actual challenge derivation independent of transaction
    creation (Section 7 of the Stage 6B spec)."""
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@dataclass(frozen=True)
class IssuedTransaction:
    """Returned ONLY by create_transaction() — the one place the raw
    `state` value is ever handed to a caller. Deliberately does NOT carry
    `code_verifier` — see this module's own docstring on why that value
    never needs to leave the database/this module's own stack frame until
    claim_transaction() recovers it.

    `state` uses `field(repr=False)` (Stage 6B independent-audit
    corrective pass #1, NOTE hardening): the default dataclass repr would
    otherwise print the raw OAuth state verbatim into anything that logs,
    prints, or exception-formats an IssuedTransaction instance whole — the
    same "never let a secret hitch a ride in an incidental repr" concern
    app/auth_session.py's own issued-session dataclass already avoids by
    never carrying the raw bearer token in the first place. `code_challenge`
    is not secret (it is sent to GitHub in the plaintext authorize URL) and
    keeps its default repr."""
    state: str = field(repr=False)
    code_challenge: str


class OAuthAdmissionRejected(Exception):
    """Raised by create_transaction() when database-authoritative OAuth
    admission control (db.oauth_transactions.create_sync() — Stage 6B
    independent-audit corrective pass #1, MAJOR 2: the global start-rate
    window or the outstanding-transaction hard cap) rejects a new `/login`
    attempt. No transaction row was created. A plain, clean domain signal
    — never a raw SQLAlchemy/DB exception — safe to surface all the way up
    through web/github_oauth.py into a generic 429 response with no
    further detail, mirroring db.auth_sessions.StalePostureError's own
    "safe to surface" contract."""


async def create_transaction() -> IssuedTransaction:
    """
    Start a brand-new OAuth transaction: a fresh CSPRNG `state`, a fresh
    CSPRNG PKCE `code_verifier`, and their derived `code_challenge` —
    persisted via db.oauth_transactions.create_sync() (state stored only
    as a digest; verifier stored in cleartext — see that module's own
    docstring) with an expiry `github_oauth_config.
    OAUTH_TRANSACTION_TTL_SECONDS` seconds from now, subject to the
    database-authoritative admission control described in
    github_oauth_config.OAUTH_MAX_STARTS_PER_MINUTE/
    OAUTH_MAX_OUTSTANDING_TRANSACTIONS (Stage 6B independent-audit
    corrective pass #1, MAJOR 2).

    Raises OAuthAdmissionRejected (no row created) if admission control
    rejects this attempt — web/github_oauth.py's login route turns that
    into a safe 429, never a 500.
    """
    raw_state = secrets.token_urlsafe(_STATE_BYTES)
    code_verifier = secrets.token_urlsafe(_VERIFIER_BYTES)
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=github_oauth_config.OAUTH_TRANSACTION_TTL_SECONDS
    )
    admitted = await asyncio.to_thread(
        db_oauth_transactions.create_sync,
        state_hash=_hash_state(raw_state),
        code_verifier=code_verifier,
        expires_at=expires_at,
        max_starts_per_window=github_oauth_config.OAUTH_MAX_STARTS_PER_MINUTE,
        max_outstanding=github_oauth_config.OAUTH_MAX_OUTSTANDING_TRANSACTIONS,
        window_seconds=github_oauth_config.OAUTH_RATE_WINDOW_SECONDS,
    )
    if not admitted:
        raise OAuthAdmissionRejected("GitHub OAuth login admission control rejected this request")
    return IssuedTransaction(state=raw_state, code_challenge=code_challenge_for(code_verifier))


async def claim_transaction(raw_state: Optional[str]) -> Optional[str]:
    """
    Atomically claim the transaction named by `raw_state` and return its
    original PKCE `code_verifier` — or None for anything that isn't a
    plausible, currently-valid, not-yet-consumed transaction: a missing,
    malformed, unknown, expired, or already-consumed `state` all fail
    closed alike (see db.oauth_transactions.claim_sync()'s own contract),
    exactly mirroring app/auth_session.resolve_session_user_id()'s
    "shape-check before hashing/DB lookup" design.
    """
    if not raw_state or not is_canonical_state(raw_state):
        return None
    return await asyncio.to_thread(db_oauth_transactions.claim_sync, state_hash=_hash_state(raw_state))
