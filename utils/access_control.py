"""
Temporary Telegram access gate (Stage 1C).

The bot is currently usable by anyone who finds it on Telegram. This module
adds a fail-closed allowlist of immutable Telegram numeric user IDs
(`from_user.id`) and a single `require_authorized` decorator that every
user-reachable handler is wrapped with, so the authorization check runs
before any handler body — and therefore before any paid API call, Telegram
file download, filesystem write, RAG query/ingestion, or UserSession
creation/mutation.

Deliberately NOT the future architecture. This is a stopgap ahead of the
later GitHub OAuth + internal UUID + Telegram-account-linking system, where
Telegram identity will be linked to an internal user and authorization will
be an application-level policy decision, not a Telegram-ID allowlist. To
keep that swap cheap, every Telegram-specific assumption here (the env var,
the "numeric user id" concept, the handler-wrapping decorator) is contained
in this one module — nothing elsewhere in the codebase imports
`TELEGRAM_ALLOWED_USER_IDS` or reasons about Telegram IDs directly.
"""

import functools
import os
from typing import Optional

from telebot import types

from bot import bot
from utils.logging import logger

_ENV_VAR = "TELEGRAM_ALLOWED_USER_IDS"

ACCESS_DENIED_MESSAGE = "\U0001F6AB Access is restricted."


def parse_allowed_user_ids(raw: Optional[str]) -> frozenset[int]:
    """
    Parse the TELEGRAM_ALLOWED_USER_IDS env value into a frozenset[int].

    Parsing policy (deliberate choice — Option B): entries are comma
    separated, surrounding whitespace is tolerated, duplicates collapse
    naturally via the set, and a non-integer entry is dropped (logged as a
    count only, never the raw entry) rather than invalidating the whole
    configuration. Rationale: this variable is hand-edited, unvalidated
    plain text; treating one fat-fingered entry as grounds to lock out every
    other already-correctly-configured ID would turn a typo into a full
    outage. The fail-closed guarantee this stage actually requires — an
    absent, empty, whitespace-only, or fully-malformed allowlist denies all
    access — still holds exactly: whenever no valid integer entries survive
    parsing (whether because the input was empty or because every entry was
    malformed), this returns an empty frozenset, and is_authorized() then
    rejects every user_id.

    Never logs the raw configured string or the resulting IDs.
    """
    if raw is None or not raw.strip():
        logger.warning(
            "%s is not configured — Telegram access is fully denied (fail closed).",
            _ENV_VAR,
        )
        return frozenset()

    valid_ids = set()
    invalid_count = 0
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            valid_ids.add(int(entry))
        except ValueError:
            invalid_count += 1

    if invalid_count:
        logger.warning(
            "%s contains invalid (non-numeric) entries — they are ignored | "
            "invalid_entry_count=%d, valid_entry_count=%d",
            _ENV_VAR, invalid_count, len(valid_ids),
        )

    if not valid_ids:
        logger.warning(
            "%s produced no valid numeric IDs — Telegram access is fully denied (fail closed).",
            _ENV_VAR,
        )

    return frozenset(valid_ids)


# Parsed once at import time, same pattern as config.py's own env reads.
# Tests monkeypatch this module attribute directly rather than the
# environment, matching this codebase's existing convention for
# import-time-bound singletons (see tests/conftest.py).
TELEGRAM_ALLOWED_USER_IDS = parse_allowed_user_ids(os.getenv(_ENV_VAR))

if TELEGRAM_ALLOWED_USER_IDS:
    logger.info(
        "Telegram access allowlist loaded | authorized_user_count=%d",
        len(TELEGRAM_ALLOWED_USER_IDS),
    )


def is_authorized(user_id: int) -> bool:
    """Authorization is based ONLY on the immutable numeric Telegram user id."""
    return user_id in TELEGRAM_ALLOWED_USER_IDS


def _extract_user_id(update) -> Optional[int]:
    """
    Defensively extract update.from_user.id as a genuine Telegram user id.

    Returns None (never raises) for any missing/malformed/unexpected shape:
    `from_user` absent, `id` absent, or `id` not a plain int. `bool` is
    deliberately rejected even though it is an `int` subclass in Python —
    it is never a valid Telegram user id and must not be accidentally
    accepted as one.

    Identity is derived ONLY from `from_user.id` — never from username,
    chat id, display name, or any other field.
    """
    from_user = getattr(update, "from_user", None)
    if from_user is None:
        return None
    user_id = getattr(from_user, "id", None)
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        return None
    return user_id


async def _deny(update) -> None:
    """
    Send the generic denial response and fail closed no matter how
    malformed `update` or the Telegram API call turns out to be.

    Routing is by concrete pyTelegramBotAPI type (`isinstance`), never by
    attribute sniffing: `types.Message` has its own top-level `.id` (an
    alias for `message_id` — see telebot's `Message.__init__`), so
    attribute presence alone cannot distinguish a Message from a
    CallbackQuery. Using the wrong id with the wrong Telegram operation is
    a routing bug even though it still fails closed (no handler runs
    either way), so each shape only ever reads the field that actually
    belongs to it.

    Never raises: a malformed update or a failed Telegram delivery must
    result in the protected handler simply not running, not in an
    exception propagating into pyTelegramBotAPI's dispatcher (which could
    log a token-bearing exception, e.g. a request URL from an HTTP error).
    """
    user_id = _extract_user_id(update)
    logger.warning("Telegram access denied | user_id=%s", user_id if user_id is not None else "unknown")

    if isinstance(update, types.Message):
        operation = "send_message"
        chat = getattr(update, "chat", None)
        target = getattr(chat, "id", None) if chat is not None else None
        send = lambda: bot.send_message(target, ACCESS_DENIED_MESSAGE)
    elif isinstance(update, types.CallbackQuery):
        operation = "answer_callback_query"
        target = getattr(update, "id", None)
        send = lambda: bot.answer_callback_query(target, ACCESS_DENIED_MESSAGE, show_alert=True)
    else:
        # Unknown/unexpected update shape: never guess a response target
        # from ad-hoc attributes, just fail closed with no outbound call.
        logger.warning("Telegram access denied | unrecognized update type, denial not sent | type=%s", type(update).__name__)
        return

    if target is None:
        logger.warning("Telegram access denied | no usable response target on this update, denial not sent")
        return

    try:
        await send()
    except Exception as exc:
        # Never interpolate the exception itself: Telegram HTTP exceptions
        # can embed the request URL, which contains the bot token. Only
        # fixed, safe metadata is logged.
        logger.warning(
            "Telegram access denied | denial delivery failed | operation=%s, exception_type=%s",
            operation,
            type(exc).__name__,
        )


def require_authorized(handler):
    """
    Decorator for pyTelegramBotAPI message/callback handlers.

    Must be applied directly to the handler function (innermost decorator,
    i.e. written *below* `@bot.message_handler(...)` /
    `@bot.callback_query_handler(...)`) so the authorization check runs
    before the handler body — and therefore before any `user_sessions`
    read/write the handler would otherwise perform. `bot.message_handler`
    registers whatever callable it is given and does no work of its own
    before invoking it, so wrapping the handler here is sufficient; no
    change to bot.py's dispatch setup is needed.
    """
    @functools.wraps(handler)
    async def wrapper(update, *args, **kwargs):
        user_id = _extract_user_id(update)
        if user_id is None or not is_authorized(user_id):
            await _deny(update)
            return None
        return await handler(update, *args, **kwargs)

    return wrapper
