"""
Stage 6C regression tests: `/start link_<secret>` redemption (Section I/J)
— every deterministic outcome, deterministic-vs-rollback commit behavior,
replay/expiry/malformed-secret handling, real registered Telegram
dispatch, the unallowlisted-sender gate, and normal /start being left
unchanged. Real disposable PostgreSQL via tests/conftest.py's postgres_db.
"""

import random
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telebot import types

import app.telegram_link as telegram_link
import db.github_identity as db_github_identity
import db.identity as db_identity
import db.documents as db_documents
import db.telegram_link as db_telegram_link
import handlers.start as start_handler
import utils.access_control as access_control
from bot import bot as shared_bot
from db.engine import get_sync_engine
from secrecy_helpers import assert_no_secret_leak
from db.models import GithubAccount, TelegramAccount, TelegramLinkAttempt, User, WebSession
from sqlalchemy import select
from sqlalchemy.orm import Session


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    needs REAL users/telegram_accounts/github_accounts rows throughout."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's in-memory documents fake — the ambiguous-merge
    gate below must see a REAL `documents` row."""
    yield


def _fresh_telegram_id() -> int:
    return random.randint(10 ** 11, 10 ** 12 - 1)


def _fresh_github_id() -> int:
    return random.randint(10 ** 8, 10 ** 9 - 1)


def _github_only_user() -> uuid.UUID:
    return db_github_identity.resolve_or_create_user_by_github_id_sync(_fresh_github_id())


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _create_attempt_raw(web_user_id) -> str:
    raw_secret = secrets.token_urlsafe(32)
    import hashlib

    outcome = db_telegram_link.create_attempt_sync(
        web_user_id=web_user_id,
        link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    assert outcome == db_telegram_link.CreateAttemptOutcome.CREATED
    return raw_secret


def _redeem_raw(raw_secret: str, telegram_id: int):
    return db_telegram_link.redeem_attempt_sync(
        link_secret_hash=__import__("hashlib").sha256(raw_secret.encode()).digest(),
        telegram_user_id=telegram_id,
    )


# ---------------------------------------------------------------------------
# A. Each deterministic outcome
# ---------------------------------------------------------------------------


def test_merged_moves_github_row_deletes_source_user_and_sessions(postgres_db):
    source = _github_only_user()
    telegram_id = _fresh_telegram_id()
    target = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    # A source session exists before redemption — must be gone after merge.
    import app.auth_session as auth_session

    _run(auth_session.create_session(source, issued_secure=True))

    raw_secret = _create_attempt_raw(source)
    result = _redeem_raw(raw_secret, telegram_id)

    assert result.outcome == db_telegram_link.RedemptionOutcome.MERGED
    assert result.target_user_id == target

    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, source) is None
        assert session.get(User, target) is not None
        github_row = session.execute(select(GithubAccount).where(GithubAccount.user_id == target)).scalar_one()
        assert github_row is not None
        assert session.execute(select(WebSession).where(WebSession.user_id == source)).first() is None
        assert session.get(TelegramLinkAttempt, source) is None


def test_already_linked_when_source_equals_target(postgres_db):
    source = _github_only_user()
    telegram_id = _fresh_telegram_id()
    target = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    raw1 = _create_attempt_raw(source)
    first = _redeem_raw(raw1, telegram_id)
    assert first.outcome == db_telegram_link.RedemptionOutcome.MERGED

    # target now has a github mapping (moved from source) — it can start a
    # SECOND attempt for itself and redeem with its own telegram id.
    raw2 = _create_attempt_raw(target)
    second = _redeem_raw(raw2, telegram_id)
    assert second.outcome == db_telegram_link.RedemptionOutcome.ALREADY_LINKED
    assert second.target_user_id == target

    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, target) is not None
        assert session.get(TelegramLinkAttempt, target) is None  # claim still consumed


def test_rejected_source_already_linked_elsewhere(postgres_db):
    source = _github_only_user()
    telegram_1 = _fresh_telegram_id()
    target_1 = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_1)
    raw1 = _create_attempt_raw(source)
    assert _redeem_raw(raw1, telegram_1).outcome == db_telegram_link.RedemptionOutcome.MERGED

    # target_1 (now github-mapped AND telegram-mapped) tries to link a
    # SECOND, different Telegram identity.
    telegram_2 = _fresh_telegram_id()
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_2)
    raw2 = _create_attempt_raw(target_1)
    result = _redeem_raw(raw2, telegram_2)

    assert result.outcome == db_telegram_link.RedemptionOutcome.REJECTED_SOURCE_ALREADY_LINKED_ELSEWHERE
    # Nothing was mutated: target_1's github mapping is untouched, no user deleted.
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, target_1) is not None
        assert session.execute(
            select(GithubAccount).where(GithubAccount.user_id == target_1)
        ).scalar_one_or_none() is not None


def test_rejected_target_already_linked_elsewhere(postgres_db):
    source_1 = _github_only_user()
    telegram_id = _fresh_telegram_id()
    target = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    raw1 = _create_attempt_raw(source_1)
    assert _redeem_raw(raw1, telegram_id).outcome == db_telegram_link.RedemptionOutcome.MERGED

    source_2 = _github_only_user()
    raw2 = _create_attempt_raw(source_2)
    result = _redeem_raw(raw2, telegram_id)

    assert result.outcome == db_telegram_link.RedemptionOutcome.REJECTED_TARGET_ALREADY_LINKED_ELSEWHERE
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, source_2) is not None  # source_2 untouched, never deleted
        assert session.execute(
            select(GithubAccount).where(GithubAccount.user_id == source_2)
        ).scalar_one_or_none() is not None


def test_rejected_ambiguous_merge_when_source_has_domain_data(postgres_db):
    """Documents are the domain data that still makes a merge ambiguous.
    (Stage 7B-3P: a source's `user_preferences` row no longer does by
    itself — see tests/test_stage7b3p_preference_link_compat.py for the
    Ø/D/M preference matrix; only material-on-both-sides rejects.)"""
    source = _github_only_user()
    document_id = uuid.uuid4()
    db_documents.create_pending_sync(  # gives source an owned document row
        document_id=document_id,
        owner_user_id=source,
        stored_name=f"{document_id.hex}.txt",
        display_name="ambiguous-merge.txt",
        content_sha256="0" * 64,
    )
    telegram_id = _fresh_telegram_id()
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    raw_secret = _create_attempt_raw(source)
    result = _redeem_raw(raw_secret, telegram_id)

    assert result.outcome == db_telegram_link.RedemptionOutcome.REJECTED_AMBIGUOUS_MERGE
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, source) is not None
        assert session.execute(
            select(GithubAccount).where(GithubAccount.user_id == source)
        ).scalar_one_or_none() is not None


def test_rejected_source_mapping_gone(postgres_db):
    """Simulates a GitHub mapping vanishing between attempt creation and
    redemption WITHOUT going through the app's own unlink path (which
    would also delete the outstanding attempt itself, making
    INVALID_OR_EXPIRED — not this branch — the observed outcome) — proves
    the redemption-side defensive gate itself, independent of how such a
    state could arise."""
    source = _github_only_user()
    telegram_id = _fresh_telegram_id()
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    raw_secret = _create_attempt_raw(source)

    from sqlalchemy import text as sql_text

    engine = get_sync_engine()
    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM github_accounts WHERE user_id = :uid"), {"uid": source})

    result = _redeem_raw(raw_secret, telegram_id)
    assert result.outcome == db_telegram_link.RedemptionOutcome.REJECTED_SOURCE_MAPPING_GONE
    with Session(engine) as session:
        assert session.get(User, source) is not None  # nothing further deleted


def test_invalid_or_expired_for_unknown_secret(postgres_db):
    telegram_id = _fresh_telegram_id()
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    result = _redeem_raw(secrets.token_urlsafe(32), telegram_id)
    assert result.outcome == db_telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED


def test_invalid_or_expired_for_a_genuinely_expired_attempt(postgres_db):
    import hashlib

    source = _github_only_user()
    telegram_id = _fresh_telegram_id()
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    raw_secret = secrets.token_urlsafe(32)
    outcome = db_telegram_link.create_attempt_sync(
        web_user_id=source,
        link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(),
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),  # already expired
    )
    assert outcome == db_telegram_link.CreateAttemptOutcome.CREATED

    result = _redeem_raw(raw_secret, telegram_id)
    assert result.outcome == db_telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED


def test_replay_of_an_already_redeemed_secret_is_invalid(postgres_db):
    source = _github_only_user()
    telegram_id = _fresh_telegram_id()
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    raw_secret = _create_attempt_raw(source)

    first = _redeem_raw(raw_secret, telegram_id)
    assert first.outcome == db_telegram_link.RedemptionOutcome.MERGED

    replay = _redeem_raw(raw_secret, telegram_id)
    assert replay.outcome == db_telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED


def test_malformed_secret_never_reaches_the_database(postgres_db, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("db.telegram_link.redeem_attempt_sync was reached for a malformed secret")

    monkeypatch.setattr(db_telegram_link, "redeem_attempt_sync", _fail_if_called)

    outcome = _run(telegram_link.redeem_link(telegram_user_id=12345, raw_secret="not-a-real-secret"))
    assert outcome == telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED


# ---------------------------------------------------------------------------
# B. Deterministic consumption vs rollback (Section J)
# ---------------------------------------------------------------------------


def test_transient_failure_rolls_back_the_claim_and_all_partial_mutations(postgres_db):
    """A genuinely unexpected exception raised immediately AFTER the atomic
    claim DELETE (via redeem_attempt_sync()'s own _test_hook_after_claim
    seam — a stable, named checkpoint, rather than counting raw
    Session.execute() calls, which is brittle against internal
    refactoring) must roll back the WHOLE transaction, claim included —
    the secret must remain redeemable afterward."""
    source = _github_only_user()
    telegram_id = _fresh_telegram_id()
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    raw_secret = _create_attempt_raw(source)

    import hashlib

    def _boom():
        raise RuntimeError("simulated transient failure")

    with pytest.raises(RuntimeError):
        db_telegram_link.redeem_attempt_sync(
            link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(),
            telegram_user_id=telegram_id,
            _test_hook_after_claim=_boom,
        )

    # The claim was rolled back — the secret is still redeemable.
    result = _redeem_raw(raw_secret, telegram_id)
    assert result.outcome == db_telegram_link.RedemptionOutcome.MERGED

    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, source) is None  # the SECOND (successful) attempt did complete the merge


# ---------------------------------------------------------------------------
# C. Real registered Telegram dispatch + allowlist gate + normal /start
# ---------------------------------------------------------------------------


def _new_message() -> types.Message:
    return types.Message.__new__(types.Message)


def _text_message(user_id: int, text_: str) -> types.Message:
    message = _new_message()
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.text = text_
    message.content_type = "text"
    return message


@pytest.mark.asyncio
async def test_real_dispatch_redeems_a_valid_link_via_the_registered_handler(monkeypatch, postgres_db):
    """Uses the ACTUAL registered `/start` handler (handlers.start.cmd_start,
    wrapped by @require_authorized, registered on the shared `bot` via
    @bot.message_handler(commands=['start'])) through pyTelegramBotAPI's
    real async dispatcher (bot.process_new_messages) — not a direct call to
    cmd_start (Section O: "real registered Telegram dispatch, not only
    direct handler invocation")."""
    source = _github_only_user()
    raw_secret = _create_attempt_raw(source)

    send_message_mock = AsyncMock()
    monkeypatch.setattr(shared_bot, "send_message", send_message_mock)

    telegram_id = _fresh_telegram_id()
    message = _text_message(telegram_id, f"/start link_{raw_secret}")

    await shared_bot.process_new_messages([message])

    send_message_mock.assert_awaited_once()
    sent_text = send_message_mock.await_args.args[1]
    assert "✅" in sent_text
    # Never a bare `assert raw_secret not in sent_text` (Stage 6C
    # corrective pass, independent-audit MINOR 2) — that would print BOTH
    # the raw secret and the full sent text on failure via pytest's own
    # assertion-rewriting introspection. Routed through the shared helper
    # instead, which fails with one fixed, secret-free message.
    assert_no_secret_leak(raw_secret, sent_text, clear_containers=[send_message_mock.call_args_list])

    target = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, source) is None
        assert session.execute(
            select(GithubAccount).where(GithubAccount.user_id == target)
        ).scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_unallowlisted_sender_cannot_redeem(monkeypatch, postgres_db):
    ALLOWED_ID = 555555555
    UNALLOWED_ID = 666666666
    monkeypatch.setattr(access_control, "is_authorized", lambda uid: uid == ALLOWED_ID)

    source = _github_only_user()
    raw_secret = _create_attempt_raw(source)

    send_message_mock = AsyncMock()
    monkeypatch.setattr(shared_bot, "send_message", send_message_mock)
    deny_mock = AsyncMock()
    monkeypatch.setattr(access_control, "_deny", deny_mock)

    message = _text_message(UNALLOWED_ID, f"/start link_{raw_secret}")
    await shared_bot.process_new_messages([message])

    deny_mock.assert_awaited_once()
    send_message_mock.assert_not_called()

    # The attempt must still be outstanding/redeemable — an unauthorized
    # sender must never consume it.
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(TelegramLinkAttempt, source) is not None


@pytest.mark.asyncio
async def test_normal_start_without_a_payload_is_unchanged(monkeypatch, postgres_db):
    send_message_mock = AsyncMock()
    monkeypatch.setattr(shared_bot, "send_message", send_message_mock)

    telegram_id = _fresh_telegram_id()
    message = _text_message(telegram_id, "/start")
    await shared_bot.process_new_messages([message])

    send_message_mock.assert_awaited_once()
    sent_text = send_message_mock.await_args.args[1]
    assert "Привет" in sent_text


@pytest.mark.asyncio
async def test_first_ever_allowlisted_start_link_creates_telegram_uuid_before_merge(monkeypatch, postgres_db):
    """A brand-new Telegram sender (no telegram_accounts row yet) redeeming
    a valid link must have their canonical Telegram UUID created FIRST
    (via the existing, unchanged resolve_user_uuid() call) and used as the
    merge target — not fail merely because it didn't exist yet."""
    source = _github_only_user()
    raw_secret = _create_attempt_raw(source)

    send_message_mock = AsyncMock()
    monkeypatch.setattr(shared_bot, "send_message", send_message_mock)

    telegram_id = _fresh_telegram_id()
    assert db_identity.lookup_user_by_telegram_id_sync(telegram_id) is None  # genuinely brand new

    message = _text_message(telegram_id, f"/start link_{raw_secret}")
    await shared_bot.process_new_messages([message])

    target = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    assert target is not None
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.execute(
            select(GithubAccount).where(GithubAccount.user_id == target)
        ).scalar_one_or_none() is not None
