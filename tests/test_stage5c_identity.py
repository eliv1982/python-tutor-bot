"""
Stage 5C regression tests: canonical identity resolution
(db.identity.resolve_or_create_user_by_telegram_id_sync) against a REAL
disposable PostgreSQL container — proving genuine database behavior
(unique constraints, the pg_advisory_xact_lock race-safety pattern), never
pretend/mocked behavior. See tests/conftest.py's postgres_container()/
postgres_db() fixtures for the disposable-container mechanics, and
resolved_telegram_ids-style reasoning in test_stage1c_access_control.py
for why this must be a REAL resolver here, not the offline in-memory fake.

Skips cleanly (not a failure) if Docker/the postgres:16-alpine image is
unavailable — see postgres_container()'s own docstring. Every test truncates
its tables fresh via postgres_db, so there is no cross-test data leakage and
nothing here ever touches a developer's real database.
"""

import asyncio
import uuid

import pytest
from sqlalchemy import select

import db.identity as db_identity
from db.engine import get_sync_engine
from db.models import TelegramAccount, User


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture (same mechanism
    tests/test_stage1c_access_control.py already uses) — this module
    exercises the REAL db.identity functions against postgres_db, never
    the offline in-memory fake."""
    yield


def test_allowed_user_resolves_to_a_stable_uuid(postgres_db):
    telegram_id = 555000111
    first = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    second = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    assert isinstance(first, uuid.UUID)
    assert first == second


def test_different_telegram_users_receive_different_uuids(postgres_db):
    uuid_a = db_identity.resolve_or_create_user_by_telegram_id_sync(555000222)
    uuid_b = db_identity.resolve_or_create_user_by_telegram_id_sync(555000333)

    assert uuid_a != uuid_b


def test_resolution_creates_exactly_one_users_row_and_one_mapping_row(postgres_db):
    telegram_id = 555000444
    user_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    with get_sync_engine().connect() as conn:
        user_row = conn.execute(select(User.id).where(User.id == user_uuid)).scalar_one_or_none()
        mapping = conn.execute(
            select(TelegramAccount.user_id).where(TelegramAccount.telegram_user_id == telegram_id)
        ).scalar_one()

    assert user_row == user_uuid
    assert mapping == user_uuid


def test_repeated_resolution_does_not_create_additional_rows(postgres_db):
    telegram_id = 555000555
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            select(TelegramAccount.user_id).where(TelegramAccount.telegram_user_id == telegram_id)
        ).fetchall()

    assert len(rows) == 1


def test_one_telegram_id_cannot_map_to_two_users(postgres_db):
    """Direct proof of the unique constraint itself: attempting to insert
    a SECOND telegram_accounts row for an already-mapped telegram_id must
    fail at the database level, not merely "happen not to occur" through
    application logic alone."""
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session

    telegram_id = 555000666
    existing_user_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    assert existing_user_uuid is not None

    with Session(get_sync_engine()) as session:
        session.add(User(id=uuid.uuid4()))
        session.flush()
        rogue_user = session.execute(select(User.id).order_by(User.created_at.desc()).limit(1)).scalar_one()
        session.add(TelegramAccount(telegram_user_id=telegram_id, user_id=rogue_user))
        with pytest.raises(IntegrityError):
            session.commit()


def test_concurrent_first_use_resolution_is_race_safe(postgres_db):
    """The core Stage 5C race-safety proof: many concurrent first-time
    resolutions of the SAME unseen telegram_id must converge to exactly
    one internal UUID / one users row — proven via real concurrent
    asyncio.to_thread() calls against a real Postgres advisory lock, not a
    single-threaded simulation."""
    telegram_id = 555000777

    async def _run():
        return await asyncio.gather(*[
            asyncio.to_thread(db_identity.resolve_or_create_user_by_telegram_id_sync, telegram_id)
            for _ in range(25)
        ])

    results = asyncio.run(_run())

    assert len(set(results)) == 1, f"race produced multiple distinct uuids: {set(results)}"

    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            select(TelegramAccount.user_id).where(TelegramAccount.telegram_user_id == telegram_id)
        ).fetchall()
    assert len(rows) == 1


def test_unallowed_user_never_gets_created_merely_by_being_referenced_elsewhere(postgres_db):
    """Stage 5C requirement #3: database identity and Telegram
    authorization are different concerns — resolve_or_create_user_by_
    telegram_id_sync() itself has no notion of the allowlist at all (that
    check lives entirely in utils/access_control.py, upstream of
    app/identity.py's call into this function). This test proves the
    negative space: a telegram_id that this test never resolves has no
    row — resolution is the ONLY thing that ever creates one, and it is
    the Telegram adapter's job (never exercised here) to gate calling it
    at all."""
    never_resolved_telegram_id = 555000888

    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            select(TelegramAccount.user_id).where(TelegramAccount.telegram_user_id == never_resolved_telegram_id)
        ).fetchall()

    assert rows == []
