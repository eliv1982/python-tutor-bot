"""
Telegram-linking web-safe configuration (Stage 6C) — mirrors web_config.py's/
github_oauth_config.py's own isolation rationale: these values are needed
ONLY by the web adapter's Telegram-linking routes (web/routes.py's
POST /api/link/telegram/start), so they live in their own module rather than
web_config.py or telegram_config.py.

Deliberately DOES NOT import telegram_config.py (never requires
TELEGRAM_BOT_TOKEN): the web adapter must stay importable/startable even on
a deployment that never configures a Telegram bot at all, and the Telegram
adapter (bot.py/main.py) never imports anything under web/*.py or this
module either — the two adapters remain fully independent of each other's
credentials, exactly as telegram_config.py's own docstring already
establishes for TELEGRAM_BOT_TOKEN vs. every other credential in this
codebase.

Unlike web_config.py/github_oauth_config.py, TELEGRAM_BOT_USERNAME does NOT
fail closed at import time: a missing or malformed value must never prevent
web startup or break any other web route (GitHub login, /api/me, logout) —
only the one route that actually needs it
(POST /api/link/telegram/start) is affected, and it degrades to a generic,
safe 503 "linking unavailable" response (see web/routes.py) rather than an
import-time crash. This is a deliberate, narrower posture than every other
credential check in this codebase: Telegram linking is an OPTIONAL
enhancement on top of GitHub login, not a route every deployment must serve.
A missing/malformed value is logged once, at import time, as a plain safe
warning (no secret material is involved either way — a bot USERNAME is
already public, visible to anyone who opens a chat with the bot on
Telegram).
"""

import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ENV_VAR = "TELEGRAM_BOT_USERNAME"

# 5-32 characters (Telegram's own username length bounds), must start with a
# letter, ASCII letters/digits/underscore only, and end in "bot"
# (case-insensitively) — Telegram requires every bot's username to end in
# "bot", so this is also a cheap sanity check against a misconfigured
# non-bot username being pasted in here by mistake.
#
# Stage 6C corrective pass (Minor 1): the middle quantifier must allow the
# documented 5-character LOWER boundary too — 1 (leading letter) + middle +
# 3 ("bot" suffix) = 5 requires a middle of {1,...}, not {3,...}. The
# previous `{3,30}` silently raised the real minimum to 7 characters
# (1+3+3), rejecting genuinely valid 5-6 character usernames the
# `5 <= len(candidate) <= 32` check below would otherwise accept. `{1,28}`
# reproduces the exact same 5..32 total range the length check already
# enforces (1+1+3=5 .. 1+28+3=32).
_USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,28}[Bb][Oo][Tt]$")


def _normalize_bot_username(raw: str) -> str | None:
    """Strips an optional leading '@' and validates shape. Returns the
    normalized (no '@') username, or None if the value is missing/malformed
    — never raises (see this module's own docstring for why this config is
    fail-SAFE, not fail-closed, unlike every other credential check in this
    codebase)."""
    if raw is None:
        return None
    candidate = raw.strip()
    if candidate.startswith("@"):
        candidate = candidate[1:]
    if not (5 <= len(candidate) <= 32):
        return None
    if not _USERNAME_RE.fullmatch(candidate):
        return None
    return candidate


TELEGRAM_BOT_USERNAME: str | None = _normalize_bot_username(os.getenv(_ENV_VAR))

if TELEGRAM_BOT_USERNAME is None:
    logger.warning(
        "%s is not configured (or is malformed) — Telegram account-linking is unavailable "
        "(POST /api/link/telegram/start will return a generic 503); every other web route is "
        "unaffected.",
        _ENV_VAR,
    )


# --- Link-attempt TTL policy (Stage 6C, Section G) --------------------------
#
# Mirrors github_oauth_config.py's own OAUTH_TRANSACTION_TTL_SECONDS
# default/hard-maximum split: a short-lived bearer secret should stay
# short-lived, and an operator-supplied override is still bounded so a
# misconfiguration can never turn this into an effectively-unbounded-lived
# credential.
_DEFAULT_LINK_ATTEMPT_TTL_SECONDS = 600  # 10 minutes
_MAX_LINK_ATTEMPT_TTL_SECONDS = 900  # 15 minutes (hard maximum)

_raw_ttl = os.getenv("TELEGRAM_LINK_ATTEMPT_TTL_SECONDS")
if _raw_ttl is None:
    LINK_ATTEMPT_TTL_SECONDS: int = _DEFAULT_LINK_ATTEMPT_TTL_SECONDS
else:
    try:
        LINK_ATTEMPT_TTL_SECONDS = int(_raw_ttl.strip())
    except ValueError as e:
        raise ValueError(
            f"TELEGRAM_LINK_ATTEMPT_TTL_SECONDS must be an integer number of seconds, got {_raw_ttl!r}"
        ) from e
    if LINK_ATTEMPT_TTL_SECONDS <= 0:
        raise ValueError(
            f"TELEGRAM_LINK_ATTEMPT_TTL_SECONDS must be a positive number of seconds, got "
            f"{LINK_ATTEMPT_TTL_SECONDS}"
        )
    if LINK_ATTEMPT_TTL_SECONDS > _MAX_LINK_ATTEMPT_TTL_SECONDS:
        raise ValueError(
            f"TELEGRAM_LINK_ATTEMPT_TTL_SECONDS={LINK_ATTEMPT_TTL_SECONDS} exceeds the maximum "
            f"allowed ({_MAX_LINK_ATTEMPT_TTL_SECONDS}, 15 minutes) — a link attempt must stay "
            f"short-lived"
        )


def telegram_deep_link(secret: str) -> str:
    """Builds the `https://t.me/<bot_username>?start=link_<secret>` deep
    link — the ONE place this URL shape is constructed, so
    web/routes.py never hand-assembles it. Raises RuntimeError if
    TELEGRAM_BOT_USERNAME is unavailable — callers must check
    `TELEGRAM_BOT_USERNAME is not None` (or catch this) before calling, and
    the web route translates that into a generic 503, never a raw 500."""
    if TELEGRAM_BOT_USERNAME is None:
        raise RuntimeError("TELEGRAM_BOT_USERNAME is not configured — cannot build a Telegram deep link")
    return f"https://t.me/{TELEGRAM_BOT_USERNAME}?start=link_{secret}"
