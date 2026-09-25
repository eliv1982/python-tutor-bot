"""
Application-level effective preferences (Stage 7A-2, effective-default policy
finalized in Stage 7B-3P) — the ONE place that turns a stored
`user_preferences` row (or its absence) into the mode/voice a user actually
gets, for Telegram (app/session.py's UserSession.get_mode/get_voice) and the
web settings API (web/routes.py -> get_effective_mode() below) alike.

Effective-default policy. BOT_MODE (config.DEFAULT_MODE) and DEFAULT_VOICE
(config.DEFAULT_VOICE) are EFFECTIVE defaults, not initial values anyone
writes: they apply whenever nothing was stored, and no code path persists
them merely to "materialize" a default (Telegram's /start never writes a
preference row). `user_preferences` therefore holds persisted customization,
not initialization:

    MODE  no row                          -> DEFAULT_MODE
          row, mode NULL                  -> DEFAULT_MODE
          row, mode canonical (BotMode)   -> the stored mode
          row, mode legacy/noncanonical   -> DEFAULT_MODE
    VOICE no row                          -> DEFAULT_VOICE
          row, voice NULL                 -> DEFAULT_VOICE
          row, voice canonical (VoiceType)-> the stored voice
          row, voice legacy/noncanonical  -> DEFAULT_VOICE (the same fallback
                                             services/tts.py already applies)

A noncanonical stored value falls back at READ time only — a read never
repairs, deletes or rewrites it. (It is still MATERIAL for Telegram-link
merge classification, which is a separate, data-preserving question answered
in db/preferences.py — see classify_preference_sync().)

The defaults are read from `config` at CALL time, never captured at import,
so both read paths always see the same current configured value.

Deliberately mode-only on the WRITE side (the web adapter has no voice
setting): set_mode() persists an explicit user selection and preserves any
stored voice. It takes an explicit, already-authenticated canonical
`user_id: uuid.UUID` — never resolves identity itself, exactly like
app/text_chat.py, and never touches app.session.user_sessions or any other
process-global conversation state.

db.preferences is imported as a qualified module (never `from
db.preferences import ...`) so tests can monkeypatch its functions, the
same convention app/session.py uses.
"""

import uuid
from typing import Any, Optional

import config
import db.preferences as db_preferences
from config import BotMode, VoiceType
from utils.helpers import await_worker, submit_worker

__all__ = [
    "PreferenceValidationError",
    "default_mode",
    "default_voice",
    "get_effective_mode",
    "resolve_effective_mode",
    "resolve_effective_voice",
    "set_mode",
]

_CANONICAL_MODES = frozenset(BotMode.ALL)
_CANONICAL_VOICES = frozenset(VoiceType.ALL)


class PreferenceValidationError(ValueError):
    """Raised for a non-UUID user_id or a mode outside config.BotMode.ALL.
    Carries only a fixed, safe message; never echoes the rejected value."""


def default_mode() -> str:
    """The configured effective default mode (config.DEFAULT_MODE, already
    validated against BotMode.ALL at configuration load), read at call
    time."""
    return config.DEFAULT_MODE


def default_voice() -> str:
    """The configured effective default voice (config.DEFAULT_VOICE, already
    validated against VoiceType.ALL at configuration load), read at call
    time."""
    return config.DEFAULT_VOICE


def _validate_user_id(user_id: Any) -> None:
    # Exact type only (not isinstance) — a uuid.UUID subclass could
    # override __eq__/__hash__ and spoof equality with an unrelated UUID.
    if type(user_id) is not uuid.UUID:
        raise PreferenceValidationError("user_id must be a uuid.UUID instance")


def _is_canonical_mode(mode: Any) -> bool:
    # A genuine `str` is required BEFORE the membership test — same
    # rationale as app/text_chat.py's _validate_mode().
    return type(mode) is str and mode in _CANONICAL_MODES


def _is_canonical_voice(voice: Any) -> bool:
    return type(voice) is str and voice in _CANONICAL_VOICES


def resolve_effective_mode(stored_mode: Optional[str]) -> str:
    """Pure resolution of a stored `mode` column value (None for a NULL mode
    OR a user with no row — the two are deliberately the same here) to the
    effective mode. See this module's docstring for the full table."""
    if _is_canonical_mode(stored_mode):
        return stored_mode
    return default_mode()


def resolve_effective_voice(stored_voice: Optional[str]) -> str:
    """Pure resolution of a stored `voice` column value (None for a NULL
    voice OR a user with no row) to the effective voice."""
    if _is_canonical_voice(stored_voice):
        return stored_voice
    return default_voice()


async def get_effective_mode(user_id: uuid.UUID) -> str:
    """Returns the user's effective mode. Never creates a row. Offloaded
    via utils.helpers.submit_worker()/await_worker() — see db/engine.py's
    module docstring (Stage 7A-3 unified-runtime corrective pass) for why
    this is no longer a plain asyncio.to_thread(): this is called on every
    GET /api/settings request, so its Task must not report itself
    "settled" to service_main.py's shutdown sequence while its worker
    thread is still using the shared DB engine."""
    _validate_user_id(user_id)
    stored_mode, _voice = await await_worker(submit_worker(db_preferences.get_preferences_sync, user_id))
    return resolve_effective_mode(stored_mode)


async def set_mode(user_id: uuid.UUID, mode: str) -> str:
    """Validates and persists `mode` — an explicit user selection, the one
    thing that ever writes a mode — returning the resulting effective mode.
    Nothing is written if validation fails; a stored voice is preserved."""
    _validate_user_id(user_id)
    if not _is_canonical_mode(mode):
        raise PreferenceValidationError("mode must be one of the canonical BotMode values")
    await await_worker(submit_worker(db_preferences.set_mode_sync, user_id, mode))
    return mode
