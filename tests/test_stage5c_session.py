"""
Stage 5C regression tests: app/session.py's ephemeral state (history,
pending-image) remains correctly isolated by canonical internal UUID —
the identity type changed from Telegram int to uuid.UUID, but the
isolation guarantee itself (Stage 5A/5B) must not weaken.

Mode/voice persistence-across-recreation is covered separately in
tests/test_stage5c_preferences.py (needs a real PostgreSQL to prove
durability — see that module for why). This module covers only the
ephemeral, in-memory pieces, so it needs no database at all — the
autouse `_default_fake_preferences`/`_default_fake_documents_catalog`
fixtures (tests/conftest.py) are active by default and unused here.
"""

import uuid

import pytest

from app.session import UserSession


def test_history_isolated_by_uuid():
    session = UserSession()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()

    session.add_message(user_a, "user", "user A's message")
    session.add_message(user_b, "user", "user B's message")

    assert session.get_history(user_a) == [{"role": "user", "content": "user A's message"}]
    assert session.get_history(user_b) == [{"role": "user", "content": "user B's message"}]


def test_history_snapshot_is_a_copy_not_the_live_list():
    """Stage 5A/5B guarantee preserved through the UUID re-keying: a
    caller must not see a turn added after the snapshot was taken."""
    session = UserSession()
    user = uuid.uuid4()

    session.add_message(user, "user", "first turn")
    snapshot = session.get_history(user)
    session.add_message(user, "assistant", "second turn (added after snapshot)")

    assert snapshot == [{"role": "user", "content": "first turn"}]


def test_clear_history_is_per_user():
    session = UserSession()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    session.add_message(user_a, "user", "hello")
    session.add_message(user_b, "user", "hello")

    session.clear_history(user_a)

    assert session.get_history(user_a) == []
    assert session.get_history(user_b) == [{"role": "user", "content": "hello"}]


def test_pending_image_isolated_by_uuid():
    session = UserSession()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()

    session.set_pending_image(user_a, "data:image/png;base64,AAA")

    assert session.get_pending_image(user_a) == "data:image/png;base64,AAA"
    assert session.get_pending_image(user_b) is None


def test_clear_pending_image_is_per_user():
    session = UserSession()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    session.set_pending_image(user_a, "img-a")
    session.set_pending_image(user_b, "img-b")

    session.clear_pending_image(user_a)

    assert session.get_pending_image(user_a) is None
    assert session.get_pending_image(user_b) == "img-b"


def test_history_length_cap_still_enforced_after_uuid_migration():
    from config import MAX_HISTORY_LENGTH

    session = UserSession()
    user = uuid.uuid4()
    for i in range(MAX_HISTORY_LENGTH * 2 + 5):
        session.add_message(user, "user", f"message {i}")

    assert len(session.get_history(user)) == MAX_HISTORY_LENGTH * 2


@pytest.mark.asyncio
async def test_mode_and_voice_are_now_async_and_isolated_by_uuid():
    """Mode/voice are durable (db.preferences) as of Stage 5C, but this
    module only needs to prove the async interface + per-user isolation
    against the offline in-memory fake (see conftest.py's
    _default_fake_preferences) — genuine cross-process durability is
    tests/test_stage5c_preferences.py's job, against a real PostgreSQL."""
    from config import BotMode

    session = UserSession()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()

    await session.set_mode(user_a, BotMode.RAG)
    await session.set_voice(user_a, "nova")
    await session.set_mode(user_b, BotMode.VOICE)

    assert await session.get_mode(user_a) == BotMode.RAG
    assert await session.get_voice(user_a) == "nova"
    assert await session.get_mode(user_b) == BotMode.VOICE
    # user_b never set a voice — falls back to the configured default.
    from config import DEFAULT_VOICE
    assert await session.get_voice(user_b) == DEFAULT_VOICE


@pytest.mark.asyncio
async def test_get_mode_and_voice_default_for_unknown_user():
    from config import BotMode, DEFAULT_VOICE

    session = UserSession()
    unknown_user = uuid.uuid4()

    assert await session.get_mode(unknown_user) == BotMode.TEXT
    assert await session.get_voice(unknown_user) == DEFAULT_VOICE
