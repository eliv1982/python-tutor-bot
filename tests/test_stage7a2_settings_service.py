"""
Stage 7A-2 regression tests: app.preferences — the adapter-independent
mode-preference service behind GET/PATCH /api/settings.

Sections A-B run against conftest.py's in-memory db.preferences fake
(plus local spies); Section C against a real disposable PostgreSQL
container. Section D is a structural import-boundary check.
"""

import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

import app.preferences as preferences
import db.preferences as db_preferences
from config import BotMode

SENTINEL = "SENTINEL-7a2-preference-value-9b1f"


@pytest.fixture
def spy_db(monkeypatch):
    state = {"rows": {}, "reads": [], "writes": [], "threads": []}

    def fake_get(user_id):
        state["reads"].append(user_id)
        state["threads"].append(threading.get_ident())
        return state["rows"].get(user_id, (None, None))

    def fake_set_mode(user_id, mode):
        state["writes"].append((user_id, mode))
        state["threads"].append(threading.get_ident())
        _mode, voice = state["rows"].get(user_id, (None, None))
        state["rows"][user_id] = (mode, voice)

    monkeypatch.setattr(db_preferences, "get_preferences_sync", fake_get)
    monkeypatch.setattr(db_preferences, "set_mode_sync", fake_set_mode)
    return state


# ============================================================================
# A. Reads.
# ============================================================================


async def test_no_row_returns_effective_default_and_writes_nothing(spy_db):
    user_id = uuid.uuid4()
    assert await preferences.get_effective_mode(user_id) == BotMode.TEXT
    assert spy_db["reads"] == [user_id]
    assert spy_db["writes"] == []


@pytest.mark.parametrize("mode", BotMode.ALL)
async def test_canonical_stored_mode_is_returned(spy_db, mode):
    user_id = uuid.uuid4()
    spy_db["rows"][user_id] = (mode, "nova")
    assert await preferences.get_effective_mode(user_id) == mode


class _StrSubclass(str):
    pass


@pytest.mark.parametrize(
    "stored",
    ["legacy-chat", "", "TEXT", " text", 1, _StrSubclass("text")],
    ids=["unknown", "empty", "wrong-case", "padded", "non-str", "str-subclass"],
)
async def test_noncanonical_stored_mode_reads_as_default_and_is_not_rewritten(spy_db, stored):
    user_id = uuid.uuid4()
    spy_db["rows"][user_id] = (stored, None)
    result = await preferences.get_effective_mode(user_id)
    assert result == BotMode.TEXT and type(result) is str
    assert spy_db["writes"] == []
    assert spy_db["rows"][user_id] == (stored, None)


async def test_db_calls_run_off_the_event_loop_thread(spy_db):
    loop_thread = threading.get_ident()
    user_id = uuid.uuid4()
    await preferences.get_effective_mode(user_id)
    await preferences.set_mode(user_id, BotMode.RAG)
    assert len(spy_db["threads"]) == 2
    assert all(t != loop_thread for t in spy_db["threads"])


# ============================================================================
# B. Writes and validation.
# ============================================================================


@pytest.mark.parametrize("mode", BotMode.ALL)
async def test_set_valid_mode_writes_exactly_once_for_exact_uuid(spy_db, mode):
    user_id = uuid.uuid4()
    assert await preferences.set_mode(user_id, mode) == mode
    assert spy_db["writes"] == [(user_id, mode)]
    assert await preferences.get_effective_mode(user_id) == mode


@pytest.mark.parametrize(
    "mode",
    [SENTINEL, "", "TEXT", " rag", None, 1, _StrSubclass("text"), ["text"]],
    ids=["unknown", "empty", "wrong-case", "padded", "none", "int", "str-subclass", "list"],
)
async def test_invalid_mode_raises_and_writes_nothing(spy_db, mode):
    with pytest.raises(preferences.PreferenceValidationError) as info:
        await preferences.set_mode(uuid.uuid4(), mode)
    assert SENTINEL not in str(info.value)
    assert spy_db["writes"] == []


@pytest.mark.parametrize("user_id", [str(uuid.uuid4()), uuid.uuid4().int, None], ids=["str", "int", "none"])
async def test_non_uuid_user_id_is_rejected_before_any_db_access(spy_db, user_id):
    with pytest.raises(preferences.PreferenceValidationError):
        await preferences.get_effective_mode(user_id)
    with pytest.raises(preferences.PreferenceValidationError):
        await preferences.set_mode(user_id, BotMode.TEXT)
    assert spy_db["reads"] == [] and spy_db["writes"] == []


class _SpoofableUUID(uuid.UUID):
    """A uuid.UUID subclass that overrides __eq__/__hash__ to collide with
    any other UUID regardless of its own value — proves the service
    rejects subclasses by exact type (not isinstance, not value equality)."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


async def test_uuid_subclass_is_rejected_before_any_db_access(spy_db):
    genuine = uuid.uuid4()
    spoofed = _SpoofableUUID(bytes=uuid.uuid4().bytes)
    assert spoofed.bytes != genuine.bytes  # a genuinely different UUID value...
    assert spoofed == genuine  # ...that the hostile __eq__ claims is equal

    with pytest.raises(preferences.PreferenceValidationError):
        await preferences.get_effective_mode(spoofed)
    with pytest.raises(preferences.PreferenceValidationError):
        await preferences.set_mode(spoofed, BotMode.TEXT)
    assert spy_db["reads"] == [] and spy_db["writes"] == []


async def test_exact_uuid_is_still_accepted_after_subclass_hardening(spy_db):
    user_id = uuid.uuid4()
    assert await preferences.get_effective_mode(user_id) == BotMode.TEXT
    assert await preferences.set_mode(user_id, BotMode.TEXT) == BotMode.TEXT
    assert spy_db["reads"] == [user_id]
    assert spy_db["writes"] == [(user_id, BotMode.TEXT)]


# ============================================================================
# C. Real PostgreSQL.
# ============================================================================


class TestRealPostgres:
    @pytest.fixture(autouse=True)
    def _default_fake_preferences(self, postgres_db):
        """Shadows conftest.py's autouse fake — exercises real db.preferences."""
        yield

    @staticmethod
    def _real_user():
        import random

        import db.identity as db_identity

        return db_identity.resolve_or_create_user_by_telegram_id_sync(random.randint(10 ** 11, 10 ** 12 - 1))

    async def test_round_trip_and_no_row_created_by_read(self):
        user_id = self._real_user()
        assert await preferences.get_effective_mode(user_id) == BotMode.TEXT
        assert db_preferences.get_preferences_sync(user_id) == (None, None)

        assert await preferences.set_mode(user_id, BotMode.VISION) == BotMode.VISION
        assert await preferences.get_effective_mode(user_id) == BotMode.VISION
        assert db_preferences.get_preferences_sync(user_id) == (BotMode.VISION, None)

    async def test_existing_voice_is_preserved_by_mode_write(self):
        user_id = self._real_user()
        db_preferences.set_voice_sync(user_id, "nova")
        await preferences.set_mode(user_id, BotMode.RAG)
        assert db_preferences.get_preferences_sync(user_id) == (BotMode.RAG, "nova")

    async def test_legacy_stored_mode_reads_as_default_and_is_untouched(self):
        user_id = self._real_user()
        db_preferences.set_mode_sync(user_id, "legacy-chat")
        assert await preferences.get_effective_mode(user_id) == BotMode.TEXT
        assert db_preferences.get_preferences_sync(user_id) == ("legacy-chat", None)

    async def test_shares_the_durable_row_telegram_reads(self):
        from app.session import UserSession

        user_id = self._real_user()
        await preferences.set_mode(user_id, BotMode.VOICE)
        assert await UserSession().get_mode(user_id) == BotMode.VOICE


# ============================================================================
# D. Structural boundary.
# ============================================================================


def test_service_never_imports_or_uses_telegram_session_state():
    assert not hasattr(preferences, "user_sessions")
    project_root = Path(__file__).resolve().parents[1]
    script = (
        "import sys, app.preferences; "
        "loaded = [m for m in sys.modules if m in ('app.session', 'app.tutor', 'telebot') "
        "or m.startswith('handlers')]; "
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, cwd=str(project_root)
    )
    assert result.returncode == 0, result.stderr[-2000:]
