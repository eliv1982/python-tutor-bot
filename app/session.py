"""
Application conversation-state ownership (Stage 5B).

Moved out of utils/helpers.py (a generic, Telegram-oriented utility
module) into its own application-layer module: conversation history,
tutoring/RAG mode, voice selection, and pending-image state are
application concerns, not Telegram-transport utilities.

Current canonical user identity remains the existing positive integer
Telegram user id (Stage 5C's internal-UUID identity migration is out of
scope here) — this module does not know or care that the id happens to
come from Telegram; it is handed an int by whichever adapter calls it.
"""

from typing import Optional


class UserSession:
    """Simple user session manager to store conversation history."""

    def __init__(self):
        self.sessions = {}

    def get_history(self, user_id: int) -> list:
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

    def add_message(self, user_id: int, role: str, content: str):
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

    def clear_history(self, user_id: int):
        """Clear conversation history for a user."""
        if user_id in self.sessions:
            del self.sessions[user_id]

    def get_mode(self, user_id: int) -> str:
        """Get current mode for a user."""
        return self.sessions.get(f"{user_id}_mode", "text")

    def set_mode(self, user_id: int, mode: str):
        """Set mode for a user."""
        self.sessions[f"{user_id}_mode"] = mode

    def get_voice(self, user_id: int) -> str:
        """Get current voice setting for a user."""
        from config import DEFAULT_VOICE
        return self.sessions.get(f"{user_id}_voice", DEFAULT_VOICE)

    def set_voice(self, user_id: int, voice: str):
        """Set voice for a user."""
        self.sessions[f"{user_id}_voice"] = voice

    def set_pending_image(self, user_id: int, image_data_url: str):
        """Сохранить base64 data URL изображения в ожидании вопроса от пользователя."""
        self.sessions[f"{user_id}_pending_image"] = image_data_url

    def get_pending_image(self, user_id: int) -> Optional[str]:
        """Получить base64 data URL изображения, ожидающего вопрос (или None)."""
        return self.sessions.get(f"{user_id}_pending_image")

    def clear_pending_image(self, user_id: int):
        """Сбросить ожидающее изображение."""
        if f"{user_id}_pending_image" in self.sessions:
            del self.sessions[f"{user_id}_pending_image"]


# Global session manager instance
user_sessions = UserSession()
