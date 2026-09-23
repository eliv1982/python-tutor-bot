"""
Application-layer boundary for Telegram <-> GitHub/web identity linking
(Stage 6C) — mirrors app/auth_session.py's own split of concerns exactly:
secret generation (CSPRNG) and hashing (SHA-256) both live HERE, never in
db/telegram_link.py, which only ever sees/stores a digest (Section F:
"database functions receive exactly a 32-byte digest, never the raw
secret"). The raw bearer secret exists in memory only transiently — inside
start_link()'s return value (handed straight to the FastAPI route, which
places it in the deep-link URL and never logs it) and inside
redeem_link()'s own stack frame while it is validated and hashed. It is
never persisted anywhere, never appears in any exception text, and this
module never logs it (Section F).

Every function here is a thin async wrapper around db/telegram_link.py's
sync functions, offloaded via utils.helpers.submit_worker()/await_worker()
— the same offload idiom every other app/*.py module in this codebase
uses for its blocking-I/O boundary (see db/engine.py's module docstring,
including why this is submit_worker()/await_worker() rather than a plain
asyncio.to_thread() as of the Stage 7A-3 unified-runtime corrective pass).
"""

import base64
import binascii
import hashlib
import logging
import re
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import db.telegram_link as db_telegram_link
import telegram_link_config
from db.telegram_link import CreateAttemptOutcome, RedemptionOutcome, UnlinkOutcome
from utils.helpers import await_worker, submit_worker

logger = logging.getLogger(__name__)

# Re-exported so callers (web/routes.py, handlers/start.py) can write
# `telegram_link.RedemptionOutcome.MERGED` / `telegram_link.UnlinkOutcome.
# TELEGRAM_KEPT` without importing db.telegram_link directly — mirrors
# app/auth_session.py's own re-export of StalePostureError.
__all__ = [
    "RedemptionOutcome",
    "UnlinkOutcome",
    "LINK_PAYLOAD_PREFIX",
    "MAX_START_PAYLOAD_LENGTH",
    "LinkStartResult",
    "extract_link_secret",
    "start_link",
    "redeem_link",
    "unlink_github",
]

# 256 bits of CSPRNG entropy — same size as app/auth_session.py's own
# session bearer token (_TOKEN_BYTES), for the same reason: comfortably
# beyond brute-force feasibility for a bearer credential.
_SECRET_BYTES = 32

# secrets.token_urlsafe(32) always encodes to exactly 43 unpadded
# base64url characters — see app/auth_session.py's identical _TOKEN_LENGTH
# derivation/rationale.
_SECRET_LENGTH = -(-(_SECRET_BYTES * 4) // 3)  # ceil division, no float
_SECRET_SHAPE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

LINK_PAYLOAD_PREFIX = "link_"

# Telegram's own /start deep-link payload has a hard 64-character limit.
# "link_" (5 chars) + a canonical 43-char secret = 48 chars, comfortably
# under it — asserted at construction time in start_link() below so a
# future change to either constant can never silently produce an
# unusable deep link.
MAX_START_PAYLOAD_LENGTH = 64


def _is_canonical_secret(raw_secret: str) -> bool:
    """Same canonical decode-and-re-encode round-trip app/auth_session.py's
    _is_canonical_token() already uses (see that function's own docstring
    for why a naive length+alphabet check alone is insufficient) — rejects
    anything that couldn't possibly be a genuine
    secrets.token_urlsafe(_SECRET_BYTES) output BEFORE it is ever hashed or
    reaches the database."""
    if len(raw_secret) != _SECRET_LENGTH or not _SECRET_SHAPE_RE.fullmatch(raw_secret):
        return False
    padding = "=" * (-len(raw_secret) % 4)
    try:
        decoded = base64.urlsafe_b64decode(raw_secret + padding)
    except (binascii.Error, ValueError):
        return False
    if len(decoded) != _SECRET_BYTES:
        return False
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    return canonical == raw_secret


def _hash_secret(raw_secret: str) -> bytes:
    return hashlib.sha256(raw_secret.encode("utf-8")).digest()


def extract_link_secret(start_payload: str) -> Optional[str]:
    """
    Validates the SHAPE of a `/start` command's payload (the text after
    `/start `, e.g. `link_<secret>`) and returns the raw secret if — and
    only if — it is a canonical `link_<secret>` payload; None for
    anything else (including a plain `/start` with no payload, or a
    payload that merely happens to start with `link_` but isn't a
    canonical secret). Never logs `start_payload` (Section I.1: "validate
    shape without logging it") — callers (handlers/start.py) must not log
    it either.
    """
    if not start_payload.startswith(LINK_PAYLOAD_PREFIX):
        return None
    candidate = start_payload[len(LINK_PAYLOAD_PREFIX):]
    if not _is_canonical_secret(candidate):
        return None
    return candidate


@dataclass(frozen=True)
class LinkStartResult:
    """Returned ONLY by start_link() — the one place the raw bearer secret
    (embedded in `deep_link`) is ever handed to a caller. `deep_link` is
    excluded from repr() (Section F: never in a repr/debug/logging path,
    even incidentally) — see app/auth_session.py's IssuedSession for the
    identical precedent."""

    deep_link: str = field(repr=False)
    expires_at: datetime


async def start_link(user_id: uuid.UUID) -> Optional[LinkStartResult]:
    """
    Starts (or atomically supersedes) `user_id`'s outstanding Telegram
    link attempt (Section G). Returns None if `user_id` has no CURRENT
    GitHub mapping (revalidated under lock inside db.telegram_link.
    create_attempt_sync()) — the caller (web/routes.py) must treat that as
    a generic, safe rejection, never a raw 500.

    Runs a best-effort, separately-transacted expired-attempt cleanup pass
    FIRST (Section D: never inside the same transaction/lock graph as the
    creation below) — a failure there is caught and swallowed here rather
    than allowed to block or corrupt this call: cleanup is a housekeeping
    concern, never a precondition for issuing a bearer.
    """
    try:
        await await_worker(submit_worker(db_telegram_link.cleanup_expired_attempts_sync))
    except Exception:
        logger.warning("telegram_link: expired-attempt cleanup pass failed; continuing", exc_info=True)

    raw_secret = secrets.token_urlsafe(_SECRET_BYTES)
    payload = f"{LINK_PAYLOAD_PREFIX}{raw_secret}"
    assert len(payload) <= MAX_START_PAYLOAD_LENGTH, "link payload exceeds Telegram's /start payload limit"

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=telegram_link_config.LINK_ATTEMPT_TTL_SECONDS)
    outcome = await await_worker(submit_worker(
        db_telegram_link.create_attempt_sync,
        web_user_id=user_id,
        link_secret_hash=_hash_secret(raw_secret),
        expires_at=expires_at,
    ))
    if outcome != CreateAttemptOutcome.CREATED:
        return None

    return LinkStartResult(deep_link=telegram_link_config.telegram_deep_link(raw_secret), expires_at=expires_at)


async def redeem_link(*, telegram_user_id: int, raw_secret: str) -> RedemptionOutcome:
    """
    Redeems `raw_secret` for the sender `telegram_user_id` (already
    resolved/authorized by handlers/start.py's existing allowlist gate and
    Telegram-identity resolution — see db/telegram_link.py's
    redeem_attempt_sync() for why that ordering is required). `raw_secret`
    is validated for canonical shape here, in the application layer,
    BEFORE it is hashed or reaches the database (Section F) — an
    unrecognizable value never drives a database round trip at all.
    """
    if not _is_canonical_secret(raw_secret):
        return RedemptionOutcome.INVALID_OR_EXPIRED
    result = await await_worker(submit_worker(
        db_telegram_link.redeem_attempt_sync,
        link_secret_hash=_hash_secret(raw_secret),
        telegram_user_id=telegram_user_id,
    ))
    return result.outcome


async def unlink_github(user_id: uuid.UUID) -> UnlinkOutcome:
    """Thin wrapper around db.telegram_link.unlink_github_sync() (Section
    L) — see that function's own docstring for the three possible
    outcomes."""
    return await await_worker(submit_worker(db_telegram_link.unlink_github_sync, user_id=user_id))
