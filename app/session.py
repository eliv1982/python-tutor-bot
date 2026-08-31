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
"""

import asyncio
import uuid
from typing import Optional

import db.preferences as db_preferences


class UserSession:
    """Simple user session manager to store conversation history."""

    def __init__(self):
        self.sessions = {}

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
        """
        return list(self.sessions.get(user_id, []))

    def add_message(self, user_id: uuid.UUID, role: str, content: str):
        """Add a message to user's conversation history."""
        if user_id not in self.sessions:
            self.sessions[user_id] = []

        self.sessions[user_id].append({
            "role": role,
            "content": content
        })

        # Limit history length
        from config import MAX_HISTORY_LENGTH
        if len(self.sessions[user_id]) > MAX_HISTORY_LENGTH * 2:
            self.sessions[user_id] = self.sessions[user_id][-MAX_HISTORY_LENGTH * 2:]

    def clear_history(self, user_id: uuid.UUID):
        """Clear conversation history for a user."""
        if user_id in self.sessions:
            del self.sessions[user_id]

    async def get_mode(self, user_id: uuid.UUID) -> str:
        """Get current mode for a user (durable — PostgreSQL)."""
        mode, _voice = await asyncio.to_thread(db_preferences.get_preferences_sync, user_id)
        return mode if mode is not None else "text"

    async def set_mode(self, user_id: uuid.UUID, mode: str):
        """Set mode for a user (durable — PostgreSQL)."""
        await asyncio.to_thread(db_preferences.set_mode_sync, user_id, mode)

    async def get_voice(self, user_id: uuid.UUID) -> str:
        """Get current voice setting for a user (durable — PostgreSQL)."""
        from config import DEFAULT_VOICE
        _mode, voice = await asyncio.to_thread(db_preferences.get_preferences_sync, user_id)
        return voice if voice is not None else DEFAULT_VOICE

    async def set_voice(self, user_id: uuid.UUID, voice: str):
        """Set voice for a user (durable — PostgreSQL)."""
        await asyncio.to_thread(db_preferences.set_voice_sync, user_id, voice)

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
