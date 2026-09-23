"""
Adapter-independent mode-preference service (Stage 7A-2) — the web
adapter's read/write path for the durable `mode` preference.

Deliberately tiny and mode-only: it reads/writes through db.preferences
(the same durable PostgreSQL row Telegram's app.session.UserSession.
get_mode/set_mode use) but never touches app.session.user_sessions or any
other process-global conversation state. Takes an explicit, already-
authenticated canonical `user_id: uuid.UUID` — never resolves identity
itself, exactly like app/text_chat.py.

Effective-mode semantics mirror UserSession.get_mode(): no stored value ->
BotMode.TEXT. Additionally, a stored value that is not one of the
canonical config.BotMode.ALL values (a legacy/noncanonical row) is
reported as BotMode.TEXT at read time — the same behavior Telegram's own
routing already gives such a value (it matches no non-text branch) — and
is never rewritten by a read.

db.preferences is imported as a qualified module (never `from
db.preferences import ...`) so tests can monkeypatch its functions, the
same convention app/session.py uses.
"""

import uuid
from typing import Any

import db.preferences as db_preferences
from config import BotMode
from utils.helpers import await_worker, submit_worker

__all__ = ["PreferenceValidationError", "get_effective_mode", "set_mode"]

DEFAULT_EFFECTIVE_MODE = BotMode.TEXT
_CANONICAL_MODES = frozenset(BotMode.ALL)


class PreferenceValidationError(ValueError):
    """Raised for a non-UUID user_id or a mode outside config.BotMode.ALL.
    Carries only a fixed, safe message; never echoes the rejected value."""


def _validate_user_id(user_id: Any) -> None:
    # Exact type only (not isinstance) — a uuid.UUID subclass could
    # override __eq__/__hash__ and spoof equality with an unrelated UUID.
    if type(user_id) is not uuid.UUID:
        raise PreferenceValidationError("user_id must be a uuid.UUID instance")


def _is_canonical_mode(mode: Any) -> bool:
    # A genuine `str` is required BEFORE the membership test — same
    # rationale as app/text_chat.py's _validate_mode().
    return type(mode) is str and mode in _CANONICAL_MODES


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
    if _is_canonical_mode(stored_mode):
        return stored_mode
    return DEFAULT_EFFECTIVE_MODE


async def set_mode(user_id: uuid.UUID, mode: str) -> str:
    """Validates and persists `mode`, returning the resulting effective
    mode. Nothing is written if validation fails."""
    _validate_user_id(user_id)
    if not _is_canonical_mode(mode):
        raise PreferenceValidationError("mode must be one of the canonical BotMode values")
    await await_worker(submit_worker(db_preferences.set_mode_sync, user_id, mode))
    return mode
