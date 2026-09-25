"""
Stage 7B-3P regression tests: user_preferences / Telegram-linking
compatibility (final policy).

Root cause being pinned: PATCH /api/settings creates a user_preferences row
for the GitHub/web canonical user, and db.telegram_link.redeem_attempt_sync()
used to treat ANY such row as REJECTED_AMBIGUOUS_MERGE — consuming the
one-use link attempt on a deterministic rejection, with no public recovery.
Telegram's /start also used to write DEFAULT_MODE up front, manufacturing a
target row before redemption ever ran.

Final policy under test:

  * BOT_MODE (config.DEFAULT_MODE) and DEFAULT_VOICE are EFFECTIVE defaults.
    Defaults are never persisted: /start creates/updates no user_preferences
    row. `user_preferences` holds persisted customization only.
  * One shared resolver (app/preferences.py) turns a stored row — or its
    absence — into the effective mode/voice for Telegram and GET
    /api/settings alike.
  * For merging, each side's preference state is  Ø (no row),  D
    (default-equivalent row) or  M (material row); a row is MATERIAL iff
    `voice IS NOT NULL OR mode NOT IN (NULL, current DEFAULT_MODE)` and
    `updated_at` never participates.

        source | target | result
        -------+--------+---------------------------------------------
          Ø    |   Ø    | MERGED; no row
          Ø    |   D    | MERGED; the target's D row is removed
          Ø    |   M    | MERGED; the target's M row is untouched
          D    |   Ø    | MERGED; the source's D row is discarded
          D    |   D    | MERGED; both D rows are discarded
          D    |   M    | MERGED; the target's M row is untouched
          M    |   Ø    | MERGED; the complete source row moves over
          M    |   D    | MERGED; the target's D row is removed, then
               |        | the complete source row moves over
          M    |   M    | REJECTED_AMBIGUOUS_MERGE; both rows unchanged

  * Documents stay a hard blocker, decided BEFORE any preference work.

Real disposable PostgreSQL (tests/conftest.py's postgres_db), real threads
for the concurrency section, real registered Telegram dispatch for the
/start flows, the real FastAPI app for the settings/link flows, and fresh
interpreters (real environment -> config.py loading) for the configuration
sections.
"""

import hashlib
import json
import os
import random
import secrets
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.testclient import TestClient
from telebot import types

import app.auth_session as auth_session
import app.preferences as app_preferences
import config
import db.auth_sessions as db_auth_sessions
import db.documents as db_documents
import db.github_identity as db_github_identity
import db.identity as db_identity
import db.preferences as db_preferences
import db.settings as db_settings
import db.telegram_link as db_telegram_link
import handlers.start as start_handler
import telegram_link_config
import web.routes as web_routes
import web_config
from app.session import user_sessions
from bot import bot as shared_bot
from concurrency_helpers import (
    assert_all_terminated,
    assert_no_exceptions,
    capture,
    run_workers,
    wait_until_blocked_on,
)
from config import BotMode, VoiceType
from db.engine import get_sync_engine
from db.models import Document, GithubAccount, TelegramLinkAttempt, User, UserPreference
from db.preferences import PreferenceState
from secrecy_helpers import assert_no_secret_leak
from web.app import create_app
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME

Outcome = db_telegram_link.RedemptionOutcome
ABSENT, DEFAULT_EQUIVALENT, MATERIAL = (
    PreferenceState.ABSENT,
    PreferenceState.DEFAULT_EQUIVALENT,
    PreferenceState.MATERIAL,
)


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fake — this module needs
    REAL users/telegram_accounts/github_accounts/user_preferences rows."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's in-memory documents fake — the scope-guard
    tests below need REAL `documents` rows for the merge gate to see."""
    yield


@pytest.fixture(autouse=True)
def _real_db(postgres_db, monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    monkeypatch.setattr(telegram_link_config, "TELEGRAM_BOT_USERNAME", "my_tutor_bot")


@pytest.fixture(autouse=True)
def _pinned_defaults(monkeypatch):
    """The configured defaults are read from `config` at CALL time by every
    reader (app/preferences.py, db.telegram_link), so pinning the module
    attributes pins the whole application. Tests needing another default
    call _configure_defaults(); the real environment -> config.py path is
    covered separately by the fresh-interpreter sections at the bottom."""
    monkeypatch.setattr(config, "DEFAULT_MODE", BotMode.TEXT)
    monkeypatch.setattr(config, "DEFAULT_VOICE", VoiceType.ALLOY)


def _configure_defaults(monkeypatch, *, mode=None, voice=None) -> None:
    if mode is not None:
        monkeypatch.setattr(config, "DEFAULT_MODE", mode)
    if voice is not None:
        monkeypatch.setattr(config, "DEFAULT_VOICE", voice)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T_SOURCE = datetime(2024, 1, 2, 3, 4, 5)
_T_TARGET = datetime(2024, 6, 7, 8, 9, 10)
_T_DEFAULT = datetime(2023, 5, 5, 5, 5, 5)


def _hash(raw: str) -> bytes:
    return hashlib.sha256(raw.encode()).digest()


def _fresh_telegram_id() -> int:
    return random.randint(10 ** 11, 10 ** 12 - 1)


def _github_only_user() -> uuid.UUID:
    return db_github_identity.resolve_or_create_user_by_github_id_sync(random.randint(10 ** 8, 10 ** 9 - 1))


def _telegram_user():
    telegram_id = _fresh_telegram_id()
    return telegram_id, db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _create_attempt_raw(web_user_id: uuid.UUID) -> str:
    raw_secret = secrets.token_urlsafe(32)
    outcome = db_telegram_link.create_attempt_sync(
        web_user_id=web_user_id,
        link_secret_hash=_hash(raw_secret),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    assert outcome == db_telegram_link.CreateAttemptOutcome.CREATED
    return raw_secret


def _redeem_raw(raw_secret: str, telegram_id: int, **kwargs):
    return db_telegram_link.redeem_attempt_sync(
        link_secret_hash=_hash(raw_secret), telegram_user_id=telegram_id, **kwargs
    )


def _insert_row(user_id: uuid.UUID, mode, voice, updated_at=None) -> None:
    """Writes a preference row DIRECTLY (bypassing the writers), so a test
    controls mode, voice AND updated_at exactly — including noncanonical
    values and NULL columns no writer would ever produce together."""
    with Session(get_sync_engine()) as session:
        values = {"user_id": user_id, "mode": mode, "voice": voice}
        if updated_at is not None:
            values["updated_at"] = updated_at
        session.add(UserPreference(**values))
        session.commit()


def _preference_snapshot(user_id: uuid.UUID):
    """The COMPLETE stored row (mode, voice, updated_at) or None."""
    with Session(get_sync_engine()) as session:
        row = session.get(UserPreference, user_id)
        return None if row is None else (row.mode, row.voice, row.updated_at)


def _preference_count() -> int:
    with Session(get_sync_engine()) as session:
        return session.execute(select(func.count()).select_from(UserPreference)).scalar_one()


def _github_mapping_owner(github_user_id: int):
    with Session(get_sync_engine()) as session:
        return session.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_user_id)
        ).scalar_one_or_none()


def _github_id_of(user_id: uuid.UUID) -> int:
    with Session(get_sync_engine()) as session:
        return session.execute(
            select(GithubAccount.github_user_id).where(GithubAccount.user_id == user_id)
        ).scalar_one()


def _user_exists(user_id: uuid.UUID) -> bool:
    with Session(get_sync_engine()) as session:
        return session.get(User, user_id) is not None


def _attempt_exists(web_user_id: uuid.UUID) -> bool:
    with Session(get_sync_engine()) as session:
        return session.get(TelegramLinkAttempt, web_user_id) is not None


def _add_document(owner_user_id: uuid.UUID) -> uuid.UUID:
    document_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=document_id,
        owner_user_id=owner_user_id,
        stored_name=f"{document_id.hex}.txt",
        display_name="compat-test.txt",
        content_sha256=hashlib.sha256(b"compat").hexdigest(),
    )
    return document_id


def _document_owner(document_id: uuid.UUID):
    with Session(get_sync_engine()) as session:
        return session.execute(select(Document.owner_user_id).where(Document.id == document_id)).scalar_one_or_none()


def _new_message() -> types.Message:
    return types.Message.__new__(types.Message)


def _text_message(user_id: int, text_: str) -> types.Message:
    message = _new_message()
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.text = text_
    message.content_type = "text"
    return message


async def _dispatch(monkeypatch, telegram_id: int, text_: str) -> str:
    """Runs one message through the REAL registered handler chain
    (bot.process_new_messages) and returns the single reply text."""
    send_message_mock = AsyncMock()
    monkeypatch.setattr(shared_bot, "send_message", send_message_mock)
    await shared_bot.process_new_messages([_text_message(telegram_id, text_)])
    send_message_mock.assert_awaited_once()
    return send_message_mock.await_args.args[1]


async def _web_client(user_id: uuid.UUID):
    """A TestClient authenticated as `user_id` through a real session, plus
    that session's CSRF header."""
    issued = await auth_session.create_session(user_id, issued_secure=False)
    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    return client, {CSRF_HEADER_NAME: derive_csrf_token(issued.raw_token)}


def _link_payload(client: TestClient, csrf: dict) -> str:
    started = client.post("/api/link/telegram/start", headers=csrf)
    assert started.status_code == 200
    payload = started.json()["deep_link"].split("?start=", 1)[1]
    assert payload.startswith("link_")
    return payload


# ---- Ø / D / M row builders ------------------------------------------------

# Two spellings of a default-equivalent row under the pinned default (text).
_D_FORMS = {"null-mode": (None, None), "default-mode": (BotMode.TEXT, None)}
_SOURCE_MATERIAL = (BotMode.VOICE, "nova", _T_SOURCE)
_TARGET_MATERIAL = (BotMode.RAG, "echo", _T_TARGET)


def _seed(user_id: uuid.UUID, state: str, material_row, d_form: str) -> None:
    if state == "absent":
        return
    if state == "default":
        mode, voice = _D_FORMS[d_form]
        _insert_row(user_id, mode, voice, _T_DEFAULT)
    else:
        _insert_row(user_id, *material_row)


# ============================================================================
# A. Canonical configuration lists and the pure effective resolver.
# ============================================================================


def _class_constants(cls) -> set:
    return {value for name, value in vars(cls).items() if name.isupper() and name != "ALL" and isinstance(value, str)}


def test_canonical_allowlists_cover_every_declared_value():
    """BotMode.ALL / VoiceType.ALL are the single canonical sources config
    validation, the resolver and the handlers share — never a second list
    that can drift from the class's own members."""
    assert set(BotMode.ALL) == _class_constants(BotMode)
    assert set(VoiceType.ALL) == _class_constants(VoiceType)
    assert len(BotMode.ALL) == len(set(BotMode.ALL)) and len(VoiceType.ALL) == len(set(VoiceType.ALL))


@pytest.mark.parametrize("default", list(BotMode.ALL))
def test_resolve_effective_mode_table(monkeypatch, default):
    _configure_defaults(monkeypatch, mode=default)
    resolve = app_preferences.resolve_effective_mode
    assert resolve(None) == default  # no row / NULL mode
    for canonical in BotMode.ALL:
        assert resolve(canonical) == canonical  # a stored canonical mode wins, default or not
    for noncanonical in ["legacy-chat", "", "TEXT", " text", 1, object()]:
        assert resolve(noncanonical) == default  # reads as the default, never raw


@pytest.mark.parametrize("default", list(VoiceType.ALL))
def test_resolve_effective_voice_table(monkeypatch, default):
    _configure_defaults(monkeypatch, voice=default)
    resolve = app_preferences.resolve_effective_voice
    assert resolve(None) == default
    for canonical in VoiceType.ALL:
        assert resolve(canonical) == canonical
    for noncanonical in ["robot", "", "ALLOY", " nova", 1]:
        assert resolve(noncanonical) == default


def test_resolvers_read_the_configured_default_at_call_time(monkeypatch):
    assert app_preferences.resolve_effective_mode(None) == BotMode.TEXT
    _configure_defaults(monkeypatch, mode=BotMode.RAG, voice=VoiceType.ONYX)
    assert app_preferences.default_mode() == BotMode.RAG
    assert app_preferences.resolve_effective_mode(None) == BotMode.RAG
    assert app_preferences.default_voice() == VoiceType.ONYX
    assert app_preferences.resolve_effective_voice(None) == VoiceType.ONYX


# ============================================================================
# B. Ø / D / M classification, tested directly (not only through the matrix).
# ============================================================================

_CLASSIFICATION_CASES = [
    # (configured default mode, stored mode, stored voice, expected)
    (BotMode.TEXT, None, None, DEFAULT_EQUIVALENT),
    (BotMode.TEXT, BotMode.TEXT, None, DEFAULT_EQUIVALENT),
    (BotMode.TEXT, BotMode.RAG, None, MATERIAL),
    (BotMode.TEXT, BotMode.VOICE, None, MATERIAL),
    (BotMode.TEXT, BotMode.VISION, None, MATERIAL),
    (BotMode.TEXT, None, VoiceType.ALLOY, MATERIAL),  # voice == DEFAULT_VOICE is still a stored voice
    (BotMode.TEXT, BotMode.TEXT, VoiceType.ALLOY, MATERIAL),
    (BotMode.TEXT, None, VoiceType.NOVA, MATERIAL),
    (BotMode.TEXT, BotMode.TEXT, VoiceType.NOVA, MATERIAL),
    (BotMode.TEXT, BotMode.RAG, VoiceType.NOVA, MATERIAL),
    (BotMode.TEXT, "legacy-chat", None, MATERIAL),  # noncanonical mode: reads as default, is never D
    (BotMode.TEXT, "", None, MATERIAL),
    (BotMode.TEXT, "TEXT", None, MATERIAL),
    (BotMode.TEXT, " text", None, MATERIAL),
    (BotMode.VOICE, BotMode.VOICE, None, DEFAULT_EQUIVALENT),
    (BotMode.VOICE, None, None, DEFAULT_EQUIVALENT),
    (BotMode.VOICE, BotMode.TEXT, None, MATERIAL),  # an OLD default row after BOT_MODE changed
    (BotMode.VOICE, BotMode.VOICE, VoiceType.ALLOY, MATERIAL),
    (BotMode.VOICE, "legacy-chat", None, MATERIAL),
    (BotMode.RAG, BotMode.RAG, None, DEFAULT_EQUIVALENT),
    (BotMode.VISION, BotMode.VISION, None, DEFAULT_EQUIVALENT),
    (BotMode.VISION, BotMode.TEXT, None, MATERIAL),
]


@pytest.mark.parametrize(
    "default_mode,mode,voice,expected",
    _CLASSIFICATION_CASES,
    ids=[f"default={c[0]}-mode={c[1]!r}-voice={c[2]!r}" for c in _CLASSIFICATION_CASES],
)
def test_row_classification(default_mode, mode, voice, expected):
    _telegram_id, user = _telegram_user()
    _insert_row(user, mode, voice, _T_DEFAULT)
    before = _preference_snapshot(user)

    assert db_preferences.classify_preference_sync(user, default_mode) is expected
    assert _preference_snapshot(user) == before  # classification never rewrites what it reads


def test_no_row_is_absent_not_default_equivalent():
    _telegram_id, user = _telegram_user()
    assert db_preferences.classify_preference_sync(user, BotMode.TEXT) is ABSENT
    assert _preference_count() == 0


@pytest.mark.parametrize("updated_at", [datetime(2000, 1, 1), datetime(2024, 1, 1), datetime(2099, 12, 31)])
@pytest.mark.parametrize(
    "mode,voice,expected",
    [(None, None, DEFAULT_EQUIVALENT), (BotMode.TEXT, None, DEFAULT_EQUIVALENT), (BotMode.RAG, None, MATERIAL), (None, "nova", MATERIAL)],
)
def test_updated_at_never_decides_materiality(updated_at, mode, voice, expected):
    _telegram_id, user = _telegram_user()
    _insert_row(user, mode, voice, updated_at)
    assert db_preferences.classify_preference_sync(user, BotMode.TEXT) is expected


def test_classification_follows_the_default_passed_in_not_a_baked_in_text():
    _telegram_id, user = _telegram_user()
    _insert_row(user, BotMode.TEXT, None)
    assert db_preferences.classify_preference_sync(user, BotMode.TEXT) is DEFAULT_EQUIVALENT
    assert db_preferences.classify_preference_sync(user, BotMode.VOICE) is MATERIAL


# ============================================================================
# C. The nine-cell Ø / D / M linking matrix (real PostgreSQL).
# ============================================================================

_MATRIX = [
    # (source, target, MERGED?, which row survives on the target)
    ("absent", "absent", True, "none"),
    ("absent", "default", True, "none"),
    ("absent", "material", True, "target"),
    ("default", "absent", True, "none"),
    ("default", "default", True, "none"),
    ("default", "material", True, "target"),
    ("material", "absent", True, "source"),
    ("material", "default", True, "source"),
    ("material", "material", False, "both"),
]


@pytest.mark.parametrize("d_form", list(_D_FORMS))
@pytest.mark.parametrize(
    "source_state,target_state,merged,survivor", _MATRIX, ids=[f"{m[0]}-{m[1]}" for m in _MATRIX]
)
async def test_preference_matrix(source_state, target_state, merged, survivor, d_form):
    source = _github_only_user()
    github_id = _github_id_of(source)
    telegram_id, target = _telegram_user()
    _seed(source, source_state, _SOURCE_MATERIAL, d_form)
    _seed(target, target_state, _TARGET_MATERIAL, d_form)
    source_before, target_before = _preference_snapshot(source), _preference_snapshot(target)
    raw_secret = _create_attempt_raw(source)

    result = _redeem_raw(raw_secret, telegram_id)

    assert not _attempt_exists(source)  # the claim is consumed for every deterministic outcome
    if not merged:
        assert result.outcome == Outcome.REJECTED_AMBIGUOUS_MERGE
        assert result.target_user_id is None
        assert _preference_snapshot(source) == source_before  # neither side overwritten,
        assert _preference_snapshot(target) == target_before  # merged or otherwise touched
        assert _preference_count() == 2
        assert _github_mapping_owner(github_id) == source  # mapping did not move
        assert _user_exists(source) and _user_exists(target)
        assert db_identity.lookup_user_by_telegram_id_sync(telegram_id) == target
        assert _redeem_raw(raw_secret, telegram_id).outcome == Outcome.INVALID_OR_EXPIRED  # replay is dead
        return

    assert result.outcome == Outcome.MERGED
    assert result.target_user_id == target
    assert _github_mapping_owner(github_id) == target
    assert not _user_exists(source) and _user_exists(target)
    assert _preference_snapshot(source) is None
    expected_row = {"none": None, "target": target_before, "source": source_before}[survivor]
    assert _preference_snapshot(target) == expected_row  # incl. exact mode, voice AND updated_at
    assert _preference_count() == (0 if survivor == "none" else 1)

    # Whatever survived (or its absence), the surviving user's EFFECTIVE mode
    # and voice are exactly what the shared resolver derives from it.
    stored_mode, stored_voice = (None, None) if expected_row is None else expected_row[:2]
    assert await user_sessions.get_mode(target) == app_preferences.resolve_effective_mode(stored_mode)
    assert await user_sessions.get_voice(target) == app_preferences.resolve_effective_voice(stored_voice)


def test_matrix_never_compares_values_to_pick_a_winner():
    """M/M with values that would 'obviously' resolve one way (a newer
    timestamp, a non-default mode on one side only, the same mode on both)
    still rejects — no winner, no field merge."""
    for source_row, target_row in [
        ((BotMode.RAG, None, datetime(2099, 1, 1)), (BotMode.RAG, None, datetime(2000, 1, 1))),
        ((None, "nova", datetime(2000, 1, 1)), (None, "nova", datetime(2099, 1, 1))),
        ((BotMode.VOICE, "nova", _T_SOURCE), (None, "echo", _T_TARGET)),
    ]:
        source = _github_only_user()
        telegram_id, target = _telegram_user()
        _insert_row(source, *source_row)
        _insert_row(target, *target_row)
        source_before, target_before = _preference_snapshot(source), _preference_snapshot(target)

        assert _redeem_raw(_create_attempt_raw(source), telegram_id).outcome == Outcome.REJECTED_AMBIGUOUS_MERGE
        assert _preference_snapshot(source) == source_before
        assert _preference_snapshot(target) == target_before


def test_partial_source_row_transfers_with_its_null_column_intact():
    """A row that only ever had `voice` set (mode NULL) must arrive as
    exactly that — never a row with a default mode filled in."""
    source = _github_only_user()
    db_preferences.set_voice_sync(source, "onyx")
    before = _preference_snapshot(source)
    assert before[0] is None

    telegram_id, target = _telegram_user()
    result = _redeem_raw(_create_attempt_raw(source), telegram_id)

    assert result.outcome == Outcome.MERGED
    assert _preference_snapshot(target) == before
    assert _preference_snapshot(target)[0] is None
    assert _preference_count() == 1


def test_a_stored_voice_equal_to_the_default_voice_is_material():
    """The voice is never classified by comparison with DEFAULT_VOICE: a
    source whose only stored value is voice=DEFAULT_VOICE transfers over a D
    target and still conflicts with an M target."""
    source = _github_only_user()
    _insert_row(source, None, VoiceType.ALLOY, _T_SOURCE)
    source_before = _preference_snapshot(source)
    telegram_id, default_target = _telegram_user()
    _insert_row(default_target, BotMode.TEXT, None, _T_DEFAULT)

    assert _redeem_raw(_create_attempt_raw(source), telegram_id).outcome == Outcome.MERGED
    assert _preference_snapshot(default_target) == source_before

    other_source = _github_only_user()
    _insert_row(other_source, None, VoiceType.ALLOY, _T_SOURCE)
    other_id, material_target = _telegram_user()
    _insert_row(material_target, BotMode.RAG, None, _T_TARGET)
    assert _redeem_raw(_create_attempt_raw(other_source), other_id).outcome == Outcome.REJECTED_AMBIGUOUS_MERGE


@pytest.mark.parametrize("side", ["source", "target"])
def test_noncanonical_stored_mode_is_material_even_though_it_reads_as_the_default(side):
    """Reads fall back to the default, merges never discard it: the legacy
    value survives verbatim (no repair), or blocks an M counterpart."""
    source = _github_only_user()
    telegram_id, target = _telegram_user()
    legacy_owner = source if side == "source" else target
    _insert_row(legacy_owner, "legacy-chat", None, _T_DEFAULT)
    legacy_before = _preference_snapshot(legacy_owner)

    result = _redeem_raw(_create_attempt_raw(source), telegram_id)

    assert result.outcome == Outcome.MERGED  # the other side is Ø
    assert _preference_snapshot(target) == legacy_before  # never repaired/normalized
    assert _preference_count() == 1


def test_noncanonical_stored_mode_conflicts_with_a_material_counterpart():
    source = _github_only_user()
    telegram_id, target = _telegram_user()
    _insert_row(source, "legacy-chat", None, _T_DEFAULT)
    _insert_row(target, BotMode.RAG, None, _T_TARGET)
    before = (_preference_snapshot(source), _preference_snapshot(target))

    assert _redeem_raw(_create_attempt_raw(source), telegram_id).outcome == Outcome.REJECTED_AMBIGUOUS_MERGE
    assert (_preference_snapshot(source), _preference_snapshot(target)) == before


def test_a_noncanonical_source_over_a_default_equivalent_target_moves_verbatim():
    source = _github_only_user()
    telegram_id, target = _telegram_user()
    _insert_row(source, "legacy-chat", "nova", _T_SOURCE)
    _insert_row(target, None, None, _T_DEFAULT)
    source_before = _preference_snapshot(source)

    assert _redeem_raw(_create_attempt_raw(source), telegram_id).outcome == Outcome.MERGED
    assert _preference_snapshot(target) == source_before


# ============================================================================
# D. Documents remain a hard blocker, decided BEFORE any preference work.
# ============================================================================


@pytest.mark.parametrize("source_state", ["absent", "default", "material"])
@pytest.mark.parametrize("target_state", ["absent", "default", "material"])
def test_source_documents_reject_before_any_preference_is_touched(source_state, target_state):
    source = _github_only_user()
    github_id = _github_id_of(source)
    document_id = _add_document(source)
    telegram_id, target = _telegram_user()
    _seed(source, source_state, _SOURCE_MATERIAL, "null-mode")
    _seed(target, target_state, _TARGET_MATERIAL, "default-mode")
    before = (_preference_snapshot(source), _preference_snapshot(target))

    result = _redeem_raw(_create_attempt_raw(source), telegram_id)

    assert result.outcome == Outcome.REJECTED_AMBIGUOUS_MERGE
    assert _document_owner(document_id) == source
    # No D row was normalized away and no M row was copied: the document gate
    # rejects first, so the preference state is exactly as it was.
    assert (_preference_snapshot(source), _preference_snapshot(target)) == before
    assert _github_mapping_owner(github_id) == source
    assert _user_exists(source)


def test_target_documents_are_left_alone_when_preferences_transfer():
    """The preference exception must not widen into any document
    migration: a target's own document stays owned by the target."""
    source = _github_only_user()
    db_preferences.set_mode_sync(source, BotMode.RAG)
    telegram_id, target = _telegram_user()
    document_id = _add_document(target)

    result = _redeem_raw(_create_attempt_raw(source), telegram_id)

    assert result.outcome == Outcome.MERGED
    assert _document_owner(document_id) == target
    assert _preference_snapshot(target)[0] == BotMode.RAG


def test_other_rejections_never_normalize_preferences():
    """REJECTED_TARGET_ALREADY_LINKED_ELSEWHERE is decided before the
    preference step: a default-equivalent target row is not normalized away."""
    first_source = _github_only_user()
    telegram_id, target = _telegram_user()
    assert _redeem_raw(_create_attempt_raw(first_source), telegram_id).outcome == Outcome.MERGED
    _insert_row(target, BotMode.TEXT, None, _T_DEFAULT)
    second_source = _github_only_user()
    _insert_row(second_source, *_SOURCE_MATERIAL)
    before = (_preference_snapshot(second_source), _preference_snapshot(target))

    result = _redeem_raw(_create_attempt_raw(second_source), telegram_id)

    assert result.outcome == Outcome.REJECTED_TARGET_ALREADY_LINKED_ELSEWHERE
    assert (_preference_snapshot(second_source), _preference_snapshot(target)) == before


# ============================================================================
# E. /start persists nothing — for every shape, including a valid link.
# ============================================================================

_START_SHAPES = [
    "plain",
    "arbitrary-payload",
    "malformed-link",
    "unknown-link",
    "superseded-link",
    "valid-link-without-source-preferences",
    "rejected-link-source-owns-a-document",
]


def _start_text_for(shape: str) -> str:
    """Returns the /start text for one shape, creating whatever linking
    state (a web source with its attempt) that shape needs."""
    if shape == "plain":
        return "/start"
    if shape == "arbitrary-payload":
        return "/start promo123"
    if shape == "malformed-link":
        return "/start link_not-a-valid-secret"
    if shape == "unknown-link":
        return f"/start link_{secrets.token_urlsafe(32)}"
    source = _github_only_user()
    if shape == "superseded-link":
        superseded = _create_attempt_raw(source)
        _create_attempt_raw(source)
        return f"/start link_{superseded}"
    if shape == "valid-link-without-source-preferences":
        return f"/start link_{_create_attempt_raw(source)}"
    if shape == "rejected-link-source-owns-a-document":
        _add_document(source)
        return f"/start link_{_create_attempt_raw(source)}"
    raise AssertionError(shape)


@pytest.mark.parametrize("configured_mode", list(BotMode.ALL))
@pytest.mark.parametrize("shape", _START_SHAPES)
async def test_start_never_creates_a_preference_row(monkeypatch, configured_mode, shape):
    """Mutation guard for `/start` persisting DEFAULT_MODE (or anything):
    a fresh sender ends every /start shape with NO user_preferences row, for
    every configured default, and still READS the configured default."""
    _configure_defaults(monkeypatch, mode=configured_mode)
    telegram_id = _fresh_telegram_id()
    start_text = _start_text_for(shape)

    await _dispatch(monkeypatch, telegram_id, start_text)

    user = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert user is not None
    assert _preference_snapshot(user) is None
    assert _preference_count() == 0  # not even for the (merged) source
    assert await user_sessions.get_mode(user) == configured_mode
    assert await user_sessions.get_voice(user) == VoiceType.ALLOY


async def test_start_never_creates_a_row_for_the_both_material_rejection(monkeypatch):
    source = _github_only_user()
    _insert_row(source, *_SOURCE_MATERIAL)
    telegram_id, target = _telegram_user()
    _insert_row(target, *_TARGET_MATERIAL)

    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start link_{_create_attempt_raw(source)}")

    assert sent_text == start_handler._LINK_REJECTED_TEXT
    assert _preference_count() == 2  # exactly the two rows the test seeded


@pytest.mark.parametrize(
    "row",
    [
        (BotMode.RAG, "onyx"),  # a returning user's real customization
        (BotMode.TEXT, None),  # a historical default-equivalent row: /start does not clean it up
        (None, "onyx"),  # a partial legacy row: /start does not fill the mode in
        ("legacy-chat", None),  # a noncanonical row: /start does not repair it
        (None, None),
    ],
    ids=["customized", "historical-default", "voice-only", "noncanonical", "all-null"],
)
@pytest.mark.parametrize("start_text", ["/start", "/start promo123", "/start link_not-a-valid-secret"])
async def test_start_never_touches_an_existing_row(monkeypatch, row, start_text):
    """No write at all — not even a same-value one (updated_at is untouched)
    — even though the configured default differs from what is stored."""
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)
    telegram_id, user = _telegram_user()
    _insert_row(user, row[0], row[1], _T_TARGET)
    before = _preference_snapshot(user)

    await _dispatch(monkeypatch, telegram_id, start_text)

    assert _preference_snapshot(user) == before
    assert _preference_count() == 1


async def test_start_never_initializes_a_voice(monkeypatch):
    _configure_defaults(monkeypatch, voice=VoiceType.NOVA)
    telegram_id = _fresh_telegram_id()

    await _dispatch(monkeypatch, telegram_id, "/start")

    user = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert db_preferences.get_preferences_sync(user) == (None, None)
    assert await user_sessions.get_voice(user) == VoiceType.NOVA  # the effective default, unstored


def test_the_start_handler_no_longer_carries_a_default_mode_writer():
    assert not hasattr(user_sessions, "initialize_mode_if_absent")
    assert not hasattr(db_preferences, "initialize_mode_if_absent_sync")
    assert not hasattr(start_handler, "DEFAULT_MODE")


# ============================================================================
# F. Telegram and GET /api/settings read the SAME effective mode/voice.
# ============================================================================

_READ_ROWS = {
    "absent": None,
    "null-mode": (None, "onyx"),
    "canonical-text": (BotMode.TEXT, None),
    "canonical-voice": (BotMode.VOICE, None),
    "canonical-vision": (BotMode.VISION, "nova"),
    "canonical-rag": (BotMode.RAG, None),
    "noncanonical": ("legacy-chat", "nova"),
}


@pytest.mark.parametrize("default_mode", [BotMode.TEXT, BotMode.VOICE, BotMode.RAG])
@pytest.mark.parametrize("row_name", list(_READ_ROWS))
async def test_telegram_and_web_settings_agree_on_the_effective_mode(monkeypatch, default_mode, row_name):
    _configure_defaults(monkeypatch, mode=default_mode)
    _telegram_id, user = _telegram_user()
    row = _READ_ROWS[row_name]
    if row is not None:
        _insert_row(user, row[0], row[1], _T_TARGET)
    before, count_before = _preference_snapshot(user), _preference_count()
    stored_mode = None if row is None else row[0]
    expected = stored_mode if stored_mode in BotMode.ALL else default_mode
    client, _csrf = await _web_client(user)

    telegram_mode = await user_sessions.get_mode(user)
    service_mode = await app_preferences.get_effective_mode(user)
    response = client.get("/api/settings")

    assert response.status_code == 200
    assert response.json() == {"mode": expected}
    assert telegram_mode == service_mode == expected
    assert _preference_snapshot(user) == before  # reads never write / repair
    assert _preference_count() == count_before


@pytest.mark.parametrize("default_voice", [VoiceType.ALLOY, VoiceType.NOVA])
@pytest.mark.parametrize(
    "row,expected_source",
    [
        (None, "default"),  # no row
        ((BotMode.RAG, None), "default"),  # row exists, voice NULL
        ((None, VoiceType.ONYX), "stored"),
        ((None, VoiceType.ALLOY), "stored"),  # stored value equal to the default is still just a stored value
        ((None, VoiceType.NOVA), "stored"),
        ((None, "robot"), "default"),  # noncanonical voice: the same fallback services/tts.py applies
    ],
    ids=["no-row", "null-voice", "stored-onyx", "stored-alloy", "stored-nova", "noncanonical"],
)
async def test_effective_voice(monkeypatch, default_voice, row, expected_source):
    _configure_defaults(monkeypatch, voice=default_voice)
    _telegram_id, user = _telegram_user()
    if row is not None:
        _insert_row(user, row[0], row[1], _T_TARGET)
    before, count_before = _preference_snapshot(user), _preference_count()
    expected = default_voice if expected_source == "default" else row[1]

    assert await user_sessions.get_voice(user) == expected
    assert _preference_snapshot(user) == before
    assert _preference_count() == count_before


# ============================================================================
# G. End-to-end flows through the real handler and the real web app.
# ============================================================================


async def test_flow1_settings_saved_before_link_survive_the_real_web_to_telegram_flow(monkeypatch):
    """Fresh GitHub/web identity saves settings; a fresh Telegram identity's
    FIRST meaningful action is the valid link redemption. The source's
    material row transfers and no artificial target row exists."""
    source = _github_only_user()
    github_id = _github_id_of(source)
    client, csrf = await _web_client(source)
    assert client.patch("/api/settings", json={"mode": BotMode.RAG}, headers=csrf).status_code == 200
    source_before = _preference_snapshot(source)
    assert _preference_count() == 1
    payload = _link_payload(client, csrf)

    telegram_id = _fresh_telegram_id()
    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start {payload}")

    assert sent_text == start_handler._LINK_MERGED_TEXT
    assert_no_secret_leak(payload[len("link_"):], sent_text)
    target = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert target is not None and target != source
    assert _github_mapping_owner(github_id) == target
    assert not _user_exists(source) and not _attempt_exists(source)
    assert _preference_snapshot(target) == source_before  # the source's own row, not a synthesized one
    assert _preference_count() == 1
    assert await user_sessions.get_mode(target) == BotMode.RAG
    assert await app_preferences.get_effective_mode(target) == BotMode.RAG


async def test_flow2_an_ordinary_historical_telegram_start_row_does_not_block_a_material_web_source(monkeypatch):
    """The Telegram target owns the row an OLD /start wrote — (mode=current
    default, voice=NULL), i.e. D. It is normalized away and the web source's
    material preferences survive on the target."""
    source = _github_only_user()
    github_id = _github_id_of(source)
    client, csrf = await _web_client(source)
    assert client.patch("/api/settings", json={"mode": BotMode.RAG}, headers=csrf).status_code == 200
    db_preferences.set_voice_sync(source, "nova")
    source_before = _preference_snapshot(source)
    payload = _link_payload(client, csrf)
    telegram_id, target = _telegram_user()
    _insert_row(target, config.DEFAULT_MODE, None, _T_DEFAULT)  # what the historical /start wrote
    assert db_preferences.classify_preference_sync(target, config.DEFAULT_MODE) is DEFAULT_EQUIVALENT

    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start {payload}")

    assert sent_text == start_handler._LINK_MERGED_TEXT
    assert _github_mapping_owner(github_id) == target
    assert _preference_snapshot(target) == source_before
    assert _preference_count() == 1
    assert await user_sessions.get_mode(target) == BotMode.RAG
    assert await user_sessions.get_voice(target) == "nova"


async def test_flow3_both_meaningfully_customized_is_the_generic_rejection_and_changes_nothing(monkeypatch):
    source = _github_only_user()
    github_id = _github_id_of(source)
    _insert_row(source, *_SOURCE_MATERIAL)
    telegram_id, target = _telegram_user()
    _insert_row(target, *_TARGET_MATERIAL)
    before = (_preference_snapshot(source), _preference_snapshot(target))
    raw_secret = _create_attempt_raw(source)

    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start link_{raw_secret}")

    assert sent_text == start_handler._LINK_REJECTED_TEXT  # identical to every other REJECTED_*
    assert "настро" not in sent_text.lower() and "prefer" not in sent_text.lower()
    assert (_preference_snapshot(source), _preference_snapshot(target)) == before
    assert _github_mapping_owner(github_id) == source
    assert _user_exists(source) and _user_exists(target)
    assert not _attempt_exists(source)  # deterministic attempt consumption is unchanged
    assert await _dispatch(monkeypatch, telegram_id, f"/start link_{raw_secret}") == (
        start_handler._LINK_INVALID_OR_EXPIRED_TEXT
    )


@pytest.mark.parametrize("source_state", ["absent", "default"])
async def test_flow4_a_customized_target_survives_an_empty_or_default_source(monkeypatch, source_state):
    source = _github_only_user()
    _seed(source, source_state, _SOURCE_MATERIAL, "default-mode")
    telegram_id, target = _telegram_user()
    db_preferences.set_mode_sync(target, BotMode.RAG)
    db_preferences.set_voice_sync(target, "echo")
    before = _preference_snapshot(target)

    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start link_{_create_attempt_raw(source)}")

    assert sent_text == start_handler._LINK_MERGED_TEXT
    assert _preference_snapshot(target) == before
    assert _preference_count() == 1
    assert not _user_exists(source)


async def test_flow5_both_default_equivalent_leaves_no_row_and_the_configured_defaults(monkeypatch):
    """The web side's D row is what PATCH /api/settings writes when the user
    explicitly selects the current default; the Telegram side's is the old
    /start row. Both are discarded — behavior is identical without them."""
    source = _github_only_user()
    client, csrf = await _web_client(source)
    assert client.patch("/api/settings", json={"mode": config.DEFAULT_MODE}, headers=csrf).status_code == 200
    payload = _link_payload(client, csrf)
    telegram_id, target = _telegram_user()
    _insert_row(target, config.DEFAULT_MODE, None, _T_DEFAULT)
    assert _preference_count() == 2

    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start {payload}")

    assert sent_text == start_handler._LINK_MERGED_TEXT
    assert _preference_count() == 0
    assert not _user_exists(source) and _user_exists(target)
    assert await user_sessions.get_mode(target) == config.DEFAULT_MODE
    assert await user_sessions.get_voice(target) == config.DEFAULT_VOICE
    assert await app_preferences.get_effective_mode(target) == config.DEFAULT_MODE


@pytest.mark.parametrize("start_first", [False, True], ids=["link-is-first-message", "plain-start-then-link"])
@pytest.mark.parametrize("configured_mode", [m for m in BotMode.ALL if m != BotMode.TEXT])
async def test_flow6_non_text_default_without_customization(monkeypatch, configured_mode, start_first):
    _configure_defaults(monkeypatch, mode=configured_mode)
    source = _github_only_user()
    client, csrf = await _web_client(source)

    # The web read: the configured default, and GET writes nothing.
    response = client.get("/api/settings")
    assert response.status_code == 200 and response.json() == {"mode": configured_mode}
    assert _preference_count() == 0
    payload = _link_payload(client, csrf)

    telegram_id = _fresh_telegram_id()
    if start_first:
        await _dispatch(monkeypatch, telegram_id, "/start")
        user = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
        assert _preference_count() == 0  # /start did not create a row
        assert await user_sessions.get_mode(user) == configured_mode  # Telegram reads it

    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start {payload}")

    assert sent_text == start_handler._LINK_MERGED_TEXT  # no manufactured preference conflict
    target = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert _preference_count() == 0
    assert await user_sessions.get_mode(target) == configured_mode
    assert await app_preferences.get_effective_mode(target) == configured_mode


async def test_flow6_a_web_setting_saved_under_a_non_text_default_still_transfers(monkeypatch):
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)
    source = _github_only_user()
    client, csrf = await _web_client(source)
    assert client.patch("/api/settings", json={"mode": BotMode.RAG}, headers=csrf).status_code == 200
    payload = _link_payload(client, csrf)
    telegram_id = _fresh_telegram_id()
    await _dispatch(monkeypatch, telegram_id, "/start")  # ordinary first contact: no row

    sent_text = await _dispatch(monkeypatch, telegram_id, f"/start {payload}")

    assert sent_text == start_handler._LINK_MERGED_TEXT
    target = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert (await user_sessions.get_mode(target), _preference_count()) == (BotMode.RAG, 1)


async def test_flow7_a_historical_row_for_the_old_default_becomes_material_after_bot_mode_changes(monkeypatch):
    """Historically /start wrote (mode=text, voice=NULL). BOT_MODE later
    changes to voice: that stored row no longer equals the default, so it is
    M — never silently deleted, still the effective mode, and it conflicts
    with a material source. The schema cannot tell it from an explicit
    selection, and this stage adds no provenance."""
    telegram_id, target = _telegram_user()
    _insert_row(target, BotMode.TEXT, None, _T_DEFAULT)
    assert db_preferences.classify_preference_sync(target, BotMode.TEXT) is DEFAULT_EQUIVALENT
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)

    assert db_preferences.classify_preference_sync(target, config.DEFAULT_MODE) is MATERIAL
    assert await user_sessions.get_mode(target) == BotMode.TEXT  # stored canonical mode wins

    # ...it survives a link whose source has nothing to offer...
    empty_source = _github_only_user()
    before = _preference_snapshot(target)
    assert await _dispatch(monkeypatch, telegram_id, f"/start link_{_create_attempt_raw(empty_source)}") == (
        start_handler._LINK_MERGED_TEXT
    )
    assert _preference_snapshot(target) == before  # not normalized away
    assert await user_sessions.get_mode(target) == BotMode.TEXT

    # ...and against a material source it is a genuine conflict.
    other_id, other_target = _telegram_user()
    _insert_row(other_target, BotMode.TEXT, None, _T_DEFAULT)
    material_source = _github_only_user()
    _insert_row(material_source, *_SOURCE_MATERIAL)
    assert await _dispatch(monkeypatch, other_id, f"/start link_{_create_attempt_raw(material_source)}") == (
        start_handler._LINK_REJECTED_TEXT
    )
    assert _preference_snapshot(other_target)[:2] == (BotMode.TEXT, None)


async def test_flow7_no_row_resolves_to_the_new_default_after_bot_mode_changes(monkeypatch):
    _telegram_id, user = _telegram_user()
    assert await user_sessions.get_mode(user) == BotMode.TEXT
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)
    assert await user_sessions.get_mode(user) == BotMode.VOICE
    assert await app_preferences.get_effective_mode(user) == BotMode.VOICE
    assert _preference_count() == 0


# ============================================================================
# H. Settings API under a non-text default (GET/PATCH shape unchanged).
# ============================================================================


async def test_settings_get_reports_the_configured_default_and_writes_nothing(monkeypatch):
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)
    _telegram_id, user = _telegram_user()
    client, _csrf = await _web_client(user)

    response = client.get("/api/settings")

    assert response.status_code == 200
    assert response.json() == {"mode": BotMode.VOICE}
    assert set(response.json()) == {"mode"}  # the public shape did not change
    assert _preference_count() == 0


async def test_settings_patch_persists_the_exact_mode_and_preserves_voice_under_a_non_text_default(monkeypatch):
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)
    _telegram_id, user = _telegram_user()
    db_preferences.set_voice_sync(user, "onyx")
    client, csrf = await _web_client(user)

    response = client.patch("/api/settings", json={"mode": BotMode.RAG}, headers=csrf)

    assert response.status_code == 200 and response.json() == {"mode": BotMode.RAG}
    assert db_preferences.get_preferences_sync(user) == (BotMode.RAG, "onyx")
    assert client.get("/api/settings").json() == {"mode": BotMode.RAG}


async def test_settings_patch_of_the_current_default_is_still_an_explicit_persisted_selection(monkeypatch):
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)
    _telegram_id, user = _telegram_user()
    client, csrf = await _web_client(user)

    assert client.patch("/api/settings", json={"mode": BotMode.VOICE}, headers=csrf).status_code == 200

    assert db_preferences.get_preferences_sync(user) == (BotMode.VOICE, None)  # persisted (a D row for merging)
    assert db_preferences.classify_preference_sync(user, config.DEFAULT_MODE) is DEFAULT_EQUIVALENT


@pytest.mark.parametrize("stored", [None, "legacy-chat"])
async def test_settings_get_maps_null_and_noncanonical_stored_modes_to_the_default(monkeypatch, stored):
    _configure_defaults(monkeypatch, mode=BotMode.VISION)
    _telegram_id, user = _telegram_user()
    _insert_row(user, stored, "nova", _T_TARGET)
    before = _preference_snapshot(user)
    client, _csrf = await _web_client(user)

    assert client.get("/api/settings").json() == {"mode": BotMode.VISION}
    assert await user_sessions.get_mode(user) == BotMode.VISION
    assert _preference_snapshot(user) == before


async def test_web_chat_stays_fixed_to_text_whatever_the_default_or_saved_mode(monkeypatch):
    """The settings mode is not consumed by web chat in this stage."""
    _configure_defaults(monkeypatch, mode=BotMode.VOICE)
    _telegram_id, user = _telegram_user()
    db_preferences.set_mode_sync(user, BotMode.RAG)
    client, csrf = await _web_client(user)
    captured = {}

    async def _fake_run_text_chat(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="ok")

    monkeypatch.setattr(web_routes.text_chat, "run_text_chat", _fake_run_text_chat)

    response = client.post("/api/chat", json={"message": "hi", "history": []}, headers=csrf)

    assert response.status_code == 200
    assert captured["mode"] == BotMode.TEXT


# ============================================================================
# I. Atomicity: an unexpected failure at ANY point of the resolution rolls the
# whole merge back — claim included — and the secret stays redeemable.
# ============================================================================


def _fail_on_nth_statement(engine, prefix: str, nth: int):
    """Installs a before_cursor_execute listener raising on the `nth`
    statement (1-based) starting with `prefix`; returns the remover."""
    seen = {"count": 0}

    def _listener(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(prefix):
            seen["count"] += 1
            if seen["count"] == nth:
                raise RuntimeError("simulated unexpected persistence failure")

    event.listen(engine, "before_cursor_execute", _listener)
    return lambda: event.remove(engine, "before_cursor_execute", _listener)


_FAULTS = [
    # (source, target, statement prefix, nth statement of that prefix, what it is)
    ("material", "absent", "INSERT INTO USER_PREFERENCES", 1),  # the copy onto the target
    ("material", "absent", "DELETE FROM USER_PREFERENCES", 1),  # AFTER the copy: both users would own a row
    ("material", "absent", "DELETE FROM USERS", 1),  # the very last mutation
    ("material", "default", "DELETE FROM USER_PREFERENCES", 1),  # removing the target's D row
    ("material", "default", "INSERT INTO USER_PREFERENCES", 1),  # D row gone, copy fails
    ("material", "default", "DELETE FROM USER_PREFERENCES", 2),  # D row gone AND copied, source row remains
    ("default", "default", "DELETE FROM USER_PREFERENCES", 1),
    ("default", "default", "DELETE FROM USER_PREFERENCES", 2),
    ("default", "default", "DELETE FROM USERS", 1),
    ("default", "absent", "DELETE FROM USER_PREFERENCES", 1),
    ("default", "material", "DELETE FROM USER_PREFERENCES", 1),
    ("absent", "default", "DELETE FROM USER_PREFERENCES", 1),
]


@pytest.mark.parametrize(
    "source_state,target_state,fault_prefix,nth",
    _FAULTS,
    ids=[f"{f[0]}-{f[1]}:{f[2].split()[0]}#{f[3]}:{f[2].split()[-1].lower()}" for f in _FAULTS],
)
def test_unexpected_failure_rolls_everything_back_and_the_retry_merges(
    source_state, target_state, fault_prefix, nth
):
    source = _github_only_user()
    github_id = _github_id_of(source)
    telegram_id, target = _telegram_user()
    _seed(source, source_state, _SOURCE_MATERIAL, "null-mode")
    _seed(target, target_state, _TARGET_MATERIAL, "default-mode")
    source_before, target_before = _preference_snapshot(source), _preference_snapshot(target)
    count_before = _preference_count()
    raw_secret = _create_attempt_raw(source)

    remove_listener = _fail_on_nth_statement(get_sync_engine(), fault_prefix, nth)
    try:
        with pytest.raises(RuntimeError, match="simulated unexpected persistence failure"):
            _redeem_raw(raw_secret, telegram_id)
    finally:
        remove_listener()

    # Nothing moved, nothing half-moved, and the claim itself was rolled back.
    assert _preference_snapshot(source) == source_before
    assert _preference_snapshot(target) == target_before
    assert _preference_count() == count_before
    assert _github_mapping_owner(github_id) == source
    assert _user_exists(source) and _user_exists(target)
    assert _attempt_exists(source)  # unexpected failure => retryable, unchanged semantics

    retry = _redeem_raw(raw_secret, telegram_id)
    assert retry.outcome == Outcome.MERGED
    expected_row = source_before if source_state == "material" else (target_before if target_state == "material" else None)
    assert _preference_snapshot(target) == expected_row
    assert _preference_snapshot(source) is None
    assert _preference_count() == (0 if expected_row is None else 1)


def test_material_material_rejection_issues_no_preference_write():
    """A deterministic M/M rejection commits (consuming the claim) with NO
    INSERT/UPDATE/DELETE against user_preferences, so it cannot carry a
    partial change."""
    source = _github_only_user()
    telegram_id, target = _telegram_user()
    _insert_row(source, *_SOURCE_MATERIAL)
    _insert_row(target, *_TARGET_MATERIAL)
    raw_secret = _create_attempt_raw(source)
    engine = get_sync_engine()
    writes = []

    def _listener(conn, cursor, statement, parameters, context, executemany):
        upper = statement.lstrip().upper()
        if upper.startswith(("INSERT INTO USER_PREFERENCES", "UPDATE USER_PREFERENCES", "DELETE FROM USER_PREFERENCES")):
            writes.append(upper[:40])

    event.listen(engine, "before_cursor_execute", _listener)
    try:
        result = _redeem_raw(raw_secret, telegram_id)
    finally:
        event.remove(engine, "before_cursor_execute", _listener)

    assert result.outcome == Outcome.REJECTED_AMBIGUOUS_MERGE
    assert writes == []
    assert not _attempt_exists(source)


# ============================================================================
# J. Lock ordering and concurrency.
# ============================================================================


def _observe_statement_tags(fn):
    """Runs `fn()` while recording, IN CALL ORDER, one tag per SQL statement
    the real production code issues (never a reimplementation of its logic)."""
    engine = get_sync_engine()
    observed = []

    def _listener(conn, cursor, statement, parameters, context, executemany):
        upper = statement.upper()
        if "PG_ADVISORY_XACT_LOCK" in upper:
            observed.append("advisory")
        elif "GITHUB_ACCOUNTS" in upper and "FOR UPDATE" in upper:
            observed.append("provider_lock")
        elif "FROM USERS" in upper and "FOR UPDATE" in upper:
            observed.append("user_lock")
        elif "FROM USERS" in upper and "FOR KEY SHARE" in upper:
            observed.append("user_share_lock")
        elif upper.lstrip().startswith("DELETE FROM TELEGRAM_LINK_ATTEMPTS") and "RETURNING" in upper:
            observed.append("attempt_claim")
        elif "USER_PREFERENCES" in upper:
            observed.append("pref_touch")

    event.listen(engine, "before_cursor_execute", _listener)
    try:
        result = fn()
    finally:
        event.remove(engine, "before_cursor_execute", _listener)
    return result, observed


@pytest.mark.parametrize("target_state", ["absent", "default", "material"])
def test_redemption_touches_preferences_only_after_advisory_provider_user_and_claim(target_state):
    """Deterministic statement-order proof (no thread scheduling involved):
    classification, D normalization and the copy add NO new lock step ahead
    of the established advisory -> github_accounts -> users -> attempt-claim
    order; they run strictly afterwards, under locks already held."""
    source = _github_only_user()
    _seed(source, "material", _SOURCE_MATERIAL, "null-mode")
    telegram_id, target = _telegram_user()
    _seed(target, target_state, _TARGET_MATERIAL, "default-mode")
    raw_secret = _create_attempt_raw(source)

    result, observed = _observe_statement_tags(lambda: _redeem_raw(raw_secret, telegram_id))

    assert result.outcome == (Outcome.REJECTED_AMBIGUOUS_MERGE if target_state == "material" else Outcome.MERGED)
    assert observed.count("advisory") == 1
    assert observed.count("provider_lock") == 1
    assert observed.count("user_lock") == 1
    assert observed.count("attempt_claim") == 1
    assert observed.count("pref_touch") >= 2  # both sides are classified
    assert "user_share_lock" not in observed  # redemption takes no writer-style lock of its own
    assert (
        observed.index("advisory")
        < observed.index("provider_lock")
        < observed.index("user_lock")
        < observed.index("attempt_claim")
        < observed.index("pref_touch")
    ), f"unexpected statement order: {observed}"


def test_preference_writers_lock_the_users_row_before_touching_user_preferences():
    """The writer half of the discipline (see db/preferences.py): the
    owning `users` row is locked BEFORE any user_preferences statement, the
    same users -> user_preferences order redemption uses."""
    _telegram_id, user = _telegram_user()

    _, observed = _observe_statement_tags(lambda: db_preferences.set_mode_sync(user, BotMode.RAG))
    assert observed[:2] == ["user_share_lock", "pref_touch"], observed

    _, observed = _observe_statement_tags(lambda: db_preferences.set_voice_sync(user, "echo"))
    assert observed[:2] == ["user_share_lock", "pref_touch"], observed


def _pause_redemption_after_user_lock(raw_secret: str, telegram_id: int):
    """Starts a redemption in a thread that pauses holding its users-row
    locks; returns (release_event, result_holder, thread)."""
    locks_held = threading.Event()
    release_redemption = threading.Event()

    def _pause_after_user_lock():
        locks_held.set()
        assert release_redemption.wait(timeout=10), "test never released redemption"

    holder = {}

    def _run_redemption():
        holder["record"] = capture(
            lambda: _redeem_raw(raw_secret, telegram_id, _test_hook_after_user_lock=_pause_after_user_lock)
        )

    thread = threading.Thread(target=_run_redemption)
    thread.start()
    assert locks_held.wait(timeout=10), "redemption never reached its user locks"
    return release_redemption, holder, thread


def _run_blocked_writer(write_fn):
    holder = {}

    def _run_writer():
        holder["record"] = capture(write_fn)

    thread = threading.Thread(target=_run_writer)
    thread.start()
    assert wait_until_blocked_on(table_substring="FOR KEY SHARE"), (
        "the preference writer never showed up as genuinely blocked on the users-row lock"
    )
    return holder, thread


def test_target_preference_write_waits_for_redemption_then_lands_on_the_transferred_row():
    """A preference write for the TARGET arriving while redemption holds
    its user locks must WAIT (never deadlock against the transfer, never
    abort it). Once redemption commits, the write lands on the transferred
    row: the source's voice survives, the writer's mode wins."""
    source = _github_only_user()
    db_preferences.set_mode_sync(source, BotMode.VOICE)
    db_preferences.set_voice_sync(source, "nova")
    telegram_id, target = _telegram_user()
    raw_secret = _create_attempt_raw(source)

    release, redemption, redemption_thread = _pause_redemption_after_user_lock(raw_secret, telegram_id)
    writer, writer_thread = _run_blocked_writer(lambda: db_preferences.set_mode_sync(target, BotMode.RAG))
    # Blocked BEFORE it inserted anything: no uncommitted target row exists
    # for redemption's transfer to wait on (that wait is the deadlock).
    assert _preference_snapshot(target) is None

    release.set()
    redemption_thread.join(timeout=15)
    writer_thread.join(timeout=15)
    assert not redemption_thread.is_alive() and not writer_thread.is_alive()

    assert redemption["record"].exception is None
    assert writer["record"].exception is None
    assert redemption["record"].result.outcome == Outcome.MERGED
    final = _preference_snapshot(target)
    assert final[0] == BotMode.RAG and final[1] == "nova"
    assert _preference_count() == 1


def test_default_equivalent_target_row_is_removed_before_a_waiting_writer_lands():
    """The target owns a D row; a writer for it queues behind redemption. The
    D row is removed and the source's row copied over; when the writer runs
    it upserts onto the COPIED row (source mode kept, the writer's voice)."""
    source = _github_only_user()
    _insert_row(source, BotMode.VOICE, "nova", _T_SOURCE)
    telegram_id, target = _telegram_user()
    _insert_row(target, BotMode.TEXT, None, _T_DEFAULT)
    raw_secret = _create_attempt_raw(source)

    release, redemption, redemption_thread = _pause_redemption_after_user_lock(raw_secret, telegram_id)
    writer, writer_thread = _run_blocked_writer(lambda: db_preferences.set_voice_sync(target, "echo"))
    assert _preference_snapshot(target)[:2] == (BotMode.TEXT, None)  # still the untouched D row

    release.set()
    redemption_thread.join(timeout=15)
    writer_thread.join(timeout=15)
    assert not redemption_thread.is_alive() and not writer_thread.is_alive()

    assert redemption["record"].exception is None and writer["record"].exception is None
    assert redemption["record"].result.outcome == Outcome.MERGED
    assert _preference_snapshot(target)[:2] == (BotMode.VOICE, "echo")
    assert _preference_count() == 1


def test_target_write_in_flight_before_redemption_turns_a_default_row_material_before_it_is_classified():
    """Ordering where the writer wins: a target-side voice write holding the
    users-row share lock (uncommitted) makes redemption wait at its user
    lock; once the write commits, redemption classifies the row it
    ACTUALLY finds — now material — and rejects rather than deleting it."""
    source = _github_only_user()
    _insert_row(source, *_SOURCE_MATERIAL)
    telegram_id, target = _telegram_user()
    _insert_row(target, BotMode.TEXT, None, _T_DEFAULT)  # D when redemption started waiting
    raw_secret = _create_attempt_raw(source)
    source_before = _preference_snapshot(source)

    engine = get_sync_engine()
    conn = engine.connect()
    tx = conn.begin()
    conn.execute(text("SELECT id FROM users WHERE id = :u FOR KEY SHARE"), {"u": target})
    conn.execute(text("UPDATE user_preferences SET voice = 'echo' WHERE user_id = :u"), {"u": target})

    redemption = {}

    def _run_redemption():
        redemption["record"] = capture(lambda: _redeem_raw(raw_secret, telegram_id))

    redemption_thread = threading.Thread(target=_run_redemption)
    redemption_thread.start()
    try:
        assert wait_until_blocked_on(table_substring="FROM users"), (
            "redemption never showed up as genuinely blocked on the users-row lock"
        )
    finally:
        tx.commit()
        conn.close()
    redemption_thread.join(timeout=15)
    assert not redemption_thread.is_alive()

    assert redemption["record"].exception is None
    assert redemption["record"].result.outcome == Outcome.REJECTED_AMBIGUOUS_MERGE
    assert _preference_snapshot(target)[:2] == (BotMode.TEXT, "echo")  # the committed write survived
    assert _preference_snapshot(source) == source_before


def test_source_preference_write_in_flight_before_redemption_is_seen_and_transferred():
    """Ordering where the preference write wins: a source-side write that
    holds the users-row share lock (and has an UNCOMMITTED row) makes
    redemption wait at its user lock; once that write commits, redemption's
    later read sees the row and transfers it — nothing is missed."""
    source = _github_only_user()
    telegram_id, target = _telegram_user()
    raw_secret = _create_attempt_raw(source)

    engine = get_sync_engine()
    conn = engine.connect()
    tx = conn.begin()
    conn.execute(text("SELECT id FROM users WHERE id = :u FOR KEY SHARE"), {"u": source})
    conn.execute(text("INSERT INTO user_preferences (user_id, mode) VALUES (:u, 'voice')"), {"u": source})

    redemption = {}

    def _run_redemption():
        redemption["record"] = capture(lambda: _redeem_raw(raw_secret, telegram_id))

    redemption_thread = threading.Thread(target=_run_redemption)
    redemption_thread.start()
    try:
        assert wait_until_blocked_on(table_substring="FROM users"), (
            "redemption never showed up as genuinely blocked on the users-row lock"
        )
    finally:
        tx.commit()
        conn.close()
    redemption_thread.join(timeout=15)
    assert not redemption_thread.is_alive()

    assert redemption["record"].exception is None
    assert redemption["record"].result.outcome == Outcome.MERGED
    assert _preference_snapshot(target)[0] == BotMode.VOICE
    assert _preference_snapshot(source) is None
    assert _preference_count() == 1


def test_source_preference_write_racing_a_merge_never_leaves_an_orphan_row():
    """The write that loses to the merge (its own user row is deleted by it)
    fails loudly with the same FK error it always did — it is never
    silently swallowed and never leaves a preference row for a deleted user."""
    source = _github_only_user()
    telegram_id, target = _telegram_user()
    raw_secret = _create_attempt_raw(source)

    release, redemption, redemption_thread = _pause_redemption_after_user_lock(raw_secret, telegram_id)
    writer, writer_thread = _run_blocked_writer(lambda: db_preferences.set_mode_sync(source, BotMode.VOICE))

    release.set()
    redemption_thread.join(timeout=15)
    writer_thread.join(timeout=15)
    assert not redemption_thread.is_alive() and not writer_thread.is_alive()

    assert redemption["record"].exception is None
    assert redemption["record"].result.outcome == Outcome.MERGED
    assert isinstance(writer["record"].exception, IntegrityError)
    assert _preference_count() == 0
    assert not _user_exists(source)


@pytest.mark.parametrize("target_state", ["absent", "default"])
def test_concurrent_target_writers_and_redemption_settle_consistently_without_deadlock(target_state):
    """Stress: one redemption races several target-side preference writers,
    all released together. Whichever side gets there first, the invariants
    hold — no deadlock, no worker exception, and exactly one of the two
    consistent end states (the source's row transferred and the writers'
    mode applied on top, or the writers' row kept and — being material — the
    merge rejected with the source's row untouched)."""
    for _round in range(3):
        source = _github_only_user()
        github_id = _github_id_of(source)
        _insert_row(source, BotMode.VOICE, "nova", _T_SOURCE)
        source_before = _preference_snapshot(source)
        telegram_id, target = _telegram_user()
        if target_state == "default":
            _insert_row(target, BotMode.TEXT, None, _T_DEFAULT)
        raw_secret = _create_attempt_raw(source)

        worker_ids = ["redeem"] + [f"write-{i}" for i in range(6)]

        def _body(worker_id):
            if worker_id == "redeem":
                return _redeem_raw(raw_secret, telegram_id).outcome
            db_preferences.set_mode_sync(target, BotMode.RAG)
            return None

        records, threads = run_workers(worker_ids, _body, timeout=30.0)

        assert_all_terminated(threads)
        assert set(records) == set(worker_ids)
        assert_no_exceptions(records)

        outcome = records["redeem"].result
        target_row = _preference_snapshot(target)
        assert target_row is not None and target_row[0] == BotMode.RAG  # a writer always lands
        if outcome == Outcome.MERGED:
            assert _preference_snapshot(source) is None
            assert target_row[1] == "nova"  # transferred voice survived the writers
            assert not _user_exists(source)
            assert _github_mapping_owner(github_id) == target
        else:
            assert outcome == Outcome.REJECTED_AMBIGUOUS_MERGE
            assert _preference_snapshot(source) == source_before
            assert target_row[1] is None
            assert _user_exists(source)
            assert _github_mapping_owner(github_id) == source
        assert not _attempt_exists(source)  # the claim is consumed either way


# ============================================================================
# K. Real environment -> config.py loading (fresh interpreters). config.py
# reads BOT_MODE / DEFAULT_VOICE from the environment once, at import, so each
# case runs in a NEW interpreter started with those variables in its
# environment — never a post-import patch, never an import-order assumption —
# against this test's real disposable database.
# ============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_REAL_CONFIG_SCRIPT = """
import asyncio
import json
import os
from types import SimpleNamespace

from starlette.testclient import TestClient
from telebot import types

import config
import web_config
from app.session import user_sessions
from bot import bot
import db.identity as db_identity
import handlers.start  # noqa: F401 — registers the REAL /start handler on the shared bot
from web.app import create_app


async def _main():
    out = {"configured_mode": config.DEFAULT_MODE, "configured_voice": config.DEFAULT_VOICE, "replies": []}

    async def _capture_send(chat_id, text, *args, **kwargs):
        out["replies"].append(text)

    bot.send_message = _capture_send
    telegram_id = int(os.environ["TEST_TELEGRAM_ID"])
    start_text = os.environ.get("TEST_START_TEXT")
    if start_text:
        message = types.Message.__new__(types.Message)
        message.from_user = SimpleNamespace(id=telegram_id, first_name="Test")
        message.chat = SimpleNamespace(id=telegram_id)
        message.text = start_text
        message.content_type = "text"
        await bot.process_new_messages([message])
    user = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    out["telegram_mode"] = None if user is None else await user_sessions.get_mode(user)
    out["telegram_voice"] = None if user is None else await user_sessions.get_voice(user)
    token = os.environ.get("TEST_SESSION_TOKEN")
    if token:
        client = TestClient(create_app())
        client.cookies.set(web_config.session_cookie_name(), token)
        response = client.get("/api/settings")
        out["web_status"] = response.status_code
        out["web_body"] = response.json()
    print("RESULT:" + json.dumps(out))


asyncio.run(_main())
"""


def _run_real_config(
    *, bot_mode=None, default_voice=None, telegram_id: int, start_text=None, session_token=None
) -> dict:
    """One fresh interpreter: real environment -> config.py, optionally one
    real /start dispatch, then Telegram's effective mode/voice and (with a
    session token) the real GET /api/settings for the same configuration."""
    env = dict(os.environ)
    env.update(
        DATABASE_URL=db_settings.DATABASE_URL,  # the container postgres_db redirected this test to
        TELEGRAM_ALLOWED_USER_IDS=str(telegram_id),
        TEST_TELEGRAM_ID=str(telegram_id),
        WEB_ENV="development",
        WEB_COOKIE_SECURE="false",
    )
    for name, value in (("BOT_MODE", bot_mode), ("DEFAULT_VOICE", default_voice)):
        if value is not None:
            env[name] = value
    if start_text is not None:
        env["TEST_START_TEXT"] = start_text
    if session_token is not None:
        env["TEST_SESSION_TOKEN"] = session_token
    completed = subprocess.run(
        [sys.executable, "-c", _REAL_CONFIG_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(_PROJECT_ROOT),
        env=env,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    result_lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT:")]
    assert len(result_lines) == 1, (completed.stdout[-2000:], completed.stderr[-2000:])
    result = json.loads(result_lines[0][len("RESULT:"):])
    if bot_mode is not None:
        assert result["configured_mode"] == bot_mode  # the environment value really reached config.py
    if default_voice is not None:
        assert result["configured_voice"] == default_voice
    return result


@pytest.mark.parametrize("bot_mode", list(BotMode.ALL))
async def test_real_config_default_mode_reaches_telegram_and_the_settings_api_without_any_row(bot_mode):
    """BOT_MODE=<mode>, no stored preference: a real /start creates no row,
    Telegram reads the configured mode, and GET /api/settings reports the
    same one (and also writes nothing)."""
    telegram_id = _fresh_telegram_id()
    web_user = _github_only_user()
    token = (await auth_session.create_session(web_user, issued_secure=False)).raw_token

    result = _run_real_config(
        bot_mode=bot_mode, telegram_id=telegram_id, start_text="/start", session_token=token
    )

    assert len(result["replies"]) == 1
    assert result["telegram_mode"] == bot_mode
    assert result["web_status"] == 200 and result["web_body"] == {"mode": bot_mode}
    assert _preference_count() == 0
    assert db_identity.lookup_user_by_telegram_id_sync(telegram_id) is not None


async def test_real_config_noncanonical_stored_mode_reads_as_the_configured_default_on_both_paths():
    telegram_id, user = _telegram_user()
    _insert_row(user, "legacy-chat", "nova", _T_TARGET)
    before = _preference_snapshot(user)
    token = (await auth_session.create_session(user, issued_secure=False)).raw_token

    result = _run_real_config(bot_mode=BotMode.VOICE, telegram_id=telegram_id, session_token=token)

    assert result["telegram_mode"] == BotMode.VOICE
    assert result["web_body"] == {"mode": BotMode.VOICE}
    assert result["telegram_voice"] == "nova"  # a canonical stored voice still wins
    assert _preference_snapshot(user) == before


async def test_real_config_default_voice_is_the_effective_voice_and_is_never_stored():
    telegram_id = _fresh_telegram_id()

    result = _run_real_config(default_voice=VoiceType.NOVA, telegram_id=telegram_id, start_text="/start")

    assert result["telegram_voice"] == VoiceType.NOVA
    assert _preference_count() == 0


async def test_real_config_link_start_under_a_non_text_default_merges_without_a_manufactured_conflict():
    source = _github_only_user()
    github_id = _github_id_of(source)
    raw_secret = _create_attempt_raw(source)
    telegram_id = _fresh_telegram_id()

    result = _run_real_config(
        bot_mode=BotMode.VOICE, telegram_id=telegram_id, start_text=f"/start link_{raw_secret}"
    )

    assert result["replies"] == [start_handler._LINK_MERGED_TEXT]
    target = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert _github_mapping_owner(github_id) == target
    assert not _user_exists(source)
    assert _preference_count() == 0
    assert result["telegram_mode"] == BotMode.VOICE  # the merged identity still reads the default


async def test_real_config_web_setting_transfers_and_beats_a_different_configured_default():
    """The real web -> Telegram flow end to end: PATCH /api/settings saves a
    mode on the web identity, the link start is issued, and the fresh
    Telegram identity's FIRST message (in a fresh interpreter configured with
    a DIFFERENT non-text BOT_MODE) redeems it. The saved setting arrives
    intact; BOT_MODE neither pre-empts nor replaces it."""
    source = _github_only_user()
    client, csrf = await _web_client(source)
    assert client.patch("/api/settings", json={"mode": BotMode.RAG}, headers=csrf).status_code == 200
    source_before = _preference_snapshot(source)
    payload = _link_payload(client, csrf)
    telegram_id = _fresh_telegram_id()

    result = _run_real_config(bot_mode=BotMode.VOICE, telegram_id=telegram_id, start_text=f"/start {payload}")

    assert result["replies"] == [start_handler._LINK_MERGED_TEXT]
    target = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert not _user_exists(source)
    assert _preference_snapshot(target) == source_before
    assert result["telegram_mode"] == BotMode.RAG
    assert _preference_count() == 1


# ---- configuration validation ---------------------------------------------


def _load_config_in_fresh_interpreter(**environment) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", "import config; print('LOADED:' + config.DEFAULT_MODE + ':' + config.DEFAULT_VOICE)"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_PROJECT_ROOT),
        env=env,
    )


@pytest.mark.parametrize("bot_mode", list(BotMode.ALL))
def test_every_supported_default_mode_is_accepted(bot_mode):
    completed = _load_config_in_fresh_interpreter(BOT_MODE=bot_mode, DEFAULT_VOICE=VoiceType.ALLOY)
    assert completed.returncode == 0, completed.stderr[-1500:]
    assert f"LOADED:{bot_mode}:{VoiceType.ALLOY}" in completed.stdout


@pytest.mark.parametrize("bad_mode", ["bogus", "", "TEXT", " text", "text ", "chat", "voice,text"])
def test_an_unsupported_default_mode_fails_configuration_loading(bad_mode):
    completed = _load_config_in_fresh_interpreter(BOT_MODE=bad_mode, DEFAULT_VOICE=VoiceType.ALLOY)
    assert completed.returncode != 0
    assert "ValueError" in completed.stderr and "BOT_MODE must be one of" in completed.stderr
    assert "LOADED:" not in completed.stdout


@pytest.mark.parametrize("voice", list(VoiceType.ALL))
def test_every_supported_default_voice_is_accepted(voice):
    completed = _load_config_in_fresh_interpreter(BOT_MODE=BotMode.TEXT, DEFAULT_VOICE=voice)
    assert completed.returncode == 0, completed.stderr[-1500:]
    assert f"LOADED:{BotMode.TEXT}:{voice}" in completed.stdout


@pytest.mark.parametrize("bad_voice", ["bogus", "", "ALLOY", " alloy", "alloy ", "coral", "nova,echo"])
def test_an_unsupported_default_voice_fails_configuration_loading(bad_voice):
    completed = _load_config_in_fresh_interpreter(BOT_MODE=BotMode.TEXT, DEFAULT_VOICE=bad_voice)
    assert completed.returncode != 0
    assert "ValueError" in completed.stderr and "DEFAULT_VOICE must be one of" in completed.stderr
    assert "LOADED:" not in completed.stdout
