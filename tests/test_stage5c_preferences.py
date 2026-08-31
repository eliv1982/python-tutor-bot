"""
Stage 5C regression tests: mode/voice preferences (db.preferences) durably
survive a fresh UserSession() instance (a process-restart proxy — the
in-memory UserSession object is entirely rebuilt, but PostgreSQL is not),
against a REAL disposable PostgreSQL container. Isolated per user, never
one user's write affecting another's.

See tests/conftest.py's postgres_container()/postgres_db() fixtures for
the disposable-container mechanics.
"""

import random
import uuid

import pytest

import db.identity as db_identity
import db.preferences as db_preferences
from app.session import UserSession
from config import BotMode


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    exercises the REAL db.preferences functions against postgres_db."""
    yield


def _real_user() -> uuid.UUID:
    """A genuine `users` row via db.identity's real resolver —
    user_preferences.user_id is a foreign key into users.id, so these
    tests need actual owning rows, never an arbitrary unresolved uuid4()."""
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


@pytest.mark.asyncio
async def test_mode_and_voice_survive_a_fresh_usersession_instance(postgres_db):
    user = _real_user()
    session1 = UserSession()
    await session1.set_mode(user, BotMode.RAG)
    await session1.set_voice(user, "nova")

    # A brand-new UserSession() — no shared in-memory state with session1
    # at all — still sees the SAME mode/voice, because they live in
    # PostgreSQL, not in either object's own __init__-created dict. This
    # is the durability proof process-restart itself would otherwise
    # require simulating.
    session2 = UserSession()
    assert await session2.get_mode(user) == BotMode.RAG
    assert await session2.get_voice(user) == "nova"


@pytest.mark.asyncio
async def test_preferences_isolated_per_user(postgres_db):
    user_a, user_b = _real_user(), _real_user()
    session = UserSession()

    await session.set_mode(user_a, BotMode.RAG)
    await session.set_voice(user_a, "nova")
    await session.set_mode(user_b, BotMode.VOICE)

    assert await session.get_mode(user_a) == BotMode.RAG
    assert await session.get_voice(user_a) == "nova"
    assert await session.get_mode(user_b) == BotMode.VOICE
    from config import DEFAULT_VOICE
    assert await session.get_voice(user_b) == DEFAULT_VOICE  # never set for B


@pytest.mark.asyncio
async def test_setting_mode_does_not_disturb_a_previously_set_voice(postgres_db):
    """set_mode()/set_voice() are independent single-column upserts — one
    must never clobber the other's already-persisted value."""
    user = _real_user()
    session = UserSession()

    await session.set_voice(user, "shimmer")
    await session.set_mode(user, BotMode.VISION)

    assert await session.get_voice(user) == "shimmer"
    assert await session.get_mode(user) == BotMode.VISION


def test_get_preferences_returns_none_none_for_an_unseen_user(postgres_db):
    unseen = uuid.uuid4()
    mode, voice = db_preferences.get_preferences_sync(unseen)
    assert mode is None
    assert voice is None


def test_set_mode_upsert_creates_row_on_first_write_and_updates_on_second(postgres_db):
    user = _real_user()
    assert db_preferences.get_preferences_sync(user) == (None, None)

    db_preferences.set_mode_sync(user, BotMode.TEXT)
    assert db_preferences.get_preferences_sync(user) == (BotMode.TEXT, None)

    db_preferences.set_mode_sync(user, BotMode.RAG)
    assert db_preferences.get_preferences_sync(user) == (BotMode.RAG, None)
