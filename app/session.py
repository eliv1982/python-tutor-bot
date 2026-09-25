"""
Application conversation-state ownership (Stage 5B, identity migrated
Stage 5C).

Moved out of utils/helpers.py (a generic, Telegram-oriented utility
module) into its own application-layer module: conversation history,
tutoring/RAG mode, voice selection, and pending-image state are
application concerns, not Telegram-transport utilities.

Canonical user identity (Stage 5C) is the internal UUID resolved by
app/identity.py — this module is handed a uuid.UUID by whichever adapter
calls it; it does not know or care that identity ultimately originates
from Telegram.

History/pending-image state stays EPHEMERAL and in-memory (Stage 5C
explicitly keeps chat history unpersisted; pending-image state may remain
ephemeral) — only re-keyed from int to uuid.UUID. Mode/voice are now
durable (PostgreSQL, via db/preferences.py), so get_mode/set_mode/
get_voice/set_voice become async — the minimum interface change actually
required by that persistence, not a wholesale rewrite of this class.
db.preferences is imported as a qualified module (never `from
db.preferences import ...`) so tests can monkeypatch its functions the
same way the rest of this codebase already does for
rag_constants/app_config.

Thread-safe history mutation (Stage 7A-1 second corrective pass): every
method that reads or mutates `self.sessions` (get_history, add_message,
add_exchange, clear_history) now does so under one `threading.Lock`
(`self._lock`). This is what makes add_exchange() a GENUINE atomic
exchange to concurrent OS-thread observers, not merely "two appends with
nothing awaited in between" (which is atomic against other asyncio TASKS
on the same event loop, since nothing yields between them, but says
nothing about a genuinely different OS thread calling get_history()
concurrently): both the user and assistant halves are appended while
holding `self._lock`, and get_history() takes its snapshot copy while
holding the SAME lock — so a concurrent get_history() call from any
thread either sees the state strictly before an in-progress add_exchange()
started, or strictly after it fully completed, and never a state with
only one of the two messages appended. The lock guards ONLY these small,
purely in-memory dict/list operations — never anything awaited — so it is
never held across an `await`/provider call, exactly like
app/generation_limits.py's own `threading.Lock` usage. get_mode/set_mode/
get_voice/set_voice are unaffected (durable PostgreSQL state via
db.preferences, not `self.sessions`).
"""

import threading
import uuid
from typing import Optional

import app.preferences as app_preferences
import db.preferences as db_preferences
from utils.helpers import await_worker, submit_worker


class UserSession:
    """Simple user session manager to store conversation history."""

    def __init__(self):
        self.sessions = {}
        # Guards every read/mutation of `self.sessions` below — never held
        # across an `await` (see this module's own docstring, "Thread-safe
        # history mutation").
        self._lock = threading.Lock()

    def get_history(self, user_id: uuid.UUID) -> list:
        """
        Get a snapshot of the conversation history for a user.

        Returns a shallow COPY, never the live list stored internally
        (Stage 5A finding / Stage 5B fix): a caller that captures this
        list and then calls add_message() for the current turn before
        building its provider request must not see that same turn
        silently appear in the snapshot it already captured. Returning
        the live list previously did exactly that from the second turn
        onward — see tests/test_stage5b_tutor_session.py.

        The copy is taken under `self._lock` (Stage 7A-1 second corrective
        pass), the SAME lock add_exchange() holds across both of its
        appends — so a concurrent call from any OS thread can never
        observe a history with only one half of an in-progress
        add_exchange() applied.
        """
        with self._lock:
            return list(self.sessions.get(user_id, []))

    def add_message(self, user_id: uuid.UUID, role: str, content: str):
        """Add a message to user's conversation history."""
        from config import MAX_HISTORY_LENGTH

        with self._lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = []

            self.sessions[user_id].append({
                "role": role,
                "content": content
            })

            # Limit history length
            if len(self.sessions[user_id]) > MAX_HISTORY_LENGTH * 2:
                self.sessions[user_id] = self.sessions[user_id][-MAX_HISTORY_LENGTH * 2:]

    def add_exchange(self, user_id: uuid.UUID, user_text: str, assistant_text: str):
        """
        Atomically record a completed user/assistant turn pair as ONE
        adapter-owned operation (Stage 7A-1 corrective pass) — never two
        independently observable add_message() calls a future change
        could insert an `await` between. A generation that times out,
        fails, or is cancelled must never reach this method at all (see
        app/tutor.py's route_text_request()), so no orphaned user-only
        turn is ever recorded.

        Both appends happen while holding `self._lock` (Stage 7A-1 second
        corrective pass) — the SAME lock get_history() takes its snapshot
        under — so this is a genuine atomic exchange to a concurrent OS-
        thread observer, not merely "nothing awaited in between" (which
        only rules out interleaving from other asyncio TASKS on the same
        event loop, not a different OS thread). See
        tests/test_stage5b_tutor_session.py's deterministic cross-thread
        proof.
        """
        from config import MAX_HISTORY_LENGTH

        with self._lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = []

            self.sessions[user_id].append({"role": "user", "content": user_text})
            self.sessions[user_id].append({"role": "assistant", "content": assistant_text})

            if len(self.sessions[user_id]) > MAX_HISTORY_LENGTH * 2:
                self.sessions[user_id] = self.sessions[user_id][-MAX_HISTORY_LENGTH * 2:]

    def clear_history(self, user_id: uuid.UUID):
        """Clear conversation history for a user."""
        with self._lock:
            if user_id in self.sessions:
                del self.sessions[user_id]

    async def get_mode(self, user_id: uuid.UUID) -> str:
        """Get the user's EFFECTIVE mode (durable — PostgreSQL): the stored
        canonical mode, else the configured default (BOT_MODE) — for no row,
        a NULL mode and a legacy/noncanonical mode alike. Never writes; the
        resolution is app.preferences.resolve_effective_mode(), the same one
        GET /api/settings uses. Offloaded via
        utils.helpers.submit_worker()/await_worker() — see db/engine.py's
        module docstring (Stage 7A-3 unified-runtime corrective pass) for
        why this is no longer a plain asyncio.to_thread()."""
        mode, _voice = await await_worker(submit_worker(db_preferences.get_preferences_sync, user_id))
        return app_preferences.resolve_effective_mode(mode)

    async def set_mode(self, user_id: uuid.UUID, mode: str):
        """Set mode for a user (durable — PostgreSQL). An explicit user
        selection — nothing else ever persists a mode (not even /start)."""
        await await_worker(submit_worker(db_preferences.set_mode_sync, user_id, mode))

    async def get_voice(self, user_id: uuid.UUID) -> str:
        """Get the user's EFFECTIVE voice (durable — PostgreSQL): the stored
        canonical voice, else the configured DEFAULT_VOICE. Never writes."""
        _mode, voice = await await_worker(submit_worker(db_preferences.get_preferences_sync, user_id))
        return app_preferences.resolve_effective_voice(voice)

    async def set_voice(self, user_id: uuid.UUID, voice: str):
        """Set voice for a user (durable — PostgreSQL)."""
        await await_worker(submit_worker(db_preferences.set_voice_sync, user_id, voice))

    def set_pending_image(self, user_id: uuid.UUID, image_data_url: str):
        """Сохранить base64 data URL изображения в ожидании вопроса от пользователя."""
        self.sessions[f"{user_id}_pending_image"] = image_data_url

    def get_pending_image(self, user_id: uuid.UUID) -> Optional[str]:
        """Получить base64 data URL изображения, ожидающего вопрос (или None)."""
        return self.sessions.get(f"{user_id}_pending_image")

    def clear_pending_image(self, user_id: uuid.UUID):
        """Сбросить ожидающее изображение."""
        if f"{user_id}_pending_image" in self.sessions:
            del self.sessions[f"{user_id}_pending_image"]


# Global session manager instance
user_sessions = UserSession()
