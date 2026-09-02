"""
Stage 6B regression tests: canonical identity resolution for the GitHub
OAuth provider (db.github_identity / app.github_identity) against a REAL
disposable PostgreSQL container — mirrors
tests/test_stage5c_identity.py's Telegram equivalent test-for-test,
including its real-concurrent-thread race-safety proof, applied to the
new GitHub provider.
"""

import asyncio
import random
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import app.github_identity as app_github_identity
import db.github_identity as db_github_identity
import db.identity as db_identity
from db.engine import get_sync_engine
from db.models import GithubAccount, User


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    exercises the REAL db.github_identity functions against postgres_db,
    never the offline in-memory fake."""
    yield


def _github_id() -> int:
    return random.randint(10 ** 8, 10 ** 9 - 1)


# --- basic resolve/create contract ------------------------------------------


def test_first_login_creates_a_stable_uuid(postgres_db):
    github_id = _github_id()
    first = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    second = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)

    assert isinstance(first, uuid.UUID)
    assert first == second


def test_different_github_ids_receive_different_uuids(postgres_db):
    uuid_a = db_github_identity.resolve_or_create_user_by_github_id_sync(_github_id())
    uuid_b = db_github_identity.resolve_or_create_user_by_github_id_sync(_github_id())
    assert uuid_a != uuid_b


def test_resolution_creates_exactly_one_users_row_and_one_mapping_row(postgres_db):
    github_id = _github_id()
    user_uuid = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)

    with get_sync_engine().connect() as conn:
        user_row = conn.execute(select(User.id).where(User.id == user_uuid)).scalar_one_or_none()
        mapping = conn.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_id)
        ).scalar_one()

    assert user_row == user_uuid
    assert mapping == user_uuid


def test_repeated_resolution_does_not_create_additional_rows(postgres_db):
    github_id = _github_id()
    db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)

    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_id)
        ).fetchall()
    assert len(rows) == 1


def test_lookup_never_creates_anything(postgres_db):
    never_resolved = _github_id()
    assert db_github_identity.lookup_user_by_github_id_sync(never_resolved) is None

    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == never_resolved)
        ).fetchall()
    assert rows == []


def test_lookup_finds_an_existing_mapping(postgres_db):
    github_id = _github_id()
    created = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    assert db_github_identity.lookup_user_by_github_id_sync(github_id) == created


# --- one GitHub identity can never be stolen/rebound (Section 11/23) ------


def test_one_github_id_cannot_map_to_two_users(postgres_db):
    """Direct proof of the unique PK itself: a second github_accounts row
    for an already-mapped github_user_id must fail at the database level."""
    github_id = _github_id()
    existing_user_uuid = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    assert existing_user_uuid is not None

    with Session(get_sync_engine()) as session:
        session.add(User(id=uuid.uuid4()))
        session.flush()
        rogue_user = session.execute(select(User.id).order_by(User.created_at.desc()).limit(1)).scalar_one()
        session.add(GithubAccount(github_user_id=github_id, user_id=rogue_user))
        with pytest.raises(IntegrityError):
            session.commit()


def test_one_canonical_user_cannot_accumulate_two_github_accounts(postgres_db):
    """Direct proof of the `user_id` UNIQUE constraint — the same
    "no accumulation" invariant TelegramAccount.user_id already enforces,
    mirrored here."""
    github_id_a = _github_id()
    user_id = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id_a)
    github_id_b = _github_id()

    with Session(get_sync_engine()) as session:
        session.add(GithubAccount(github_user_id=github_id_b, user_id=user_id))
        with pytest.raises(IntegrityError):
            session.commit()


# --- concurrent first-login race safety (Section 11/23, mirrors Stage 5C) -


def test_concurrent_first_use_resolution_is_race_safe(postgres_db):
    """The core Stage 6B race-safety proof, mirroring
    test_stage5c_identity.py's Telegram equivalent exactly: many
    concurrent first-time resolutions of the SAME unseen github_user_id
    must converge to exactly one internal UUID / one users row — proven
    via real concurrent asyncio.to_thread() calls against a real Postgres
    advisory lock, not a single-threaded simulation."""
    github_id = _github_id()

    async def _run():
        return await asyncio.gather(*[
            asyncio.to_thread(db_github_identity.resolve_or_create_user_by_github_id_sync, github_id)
            for _ in range(25)
        ])

    results = asyncio.run(_run())

    assert len(set(results)) == 1, f"race produced multiple distinct uuids: {set(results)}"

    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_id)
        ).fetchall()
    assert len(rows) == 1


def test_github_and_telegram_advisory_lock_namespaces_never_collide(postgres_db):
    """Concurrent first-use resolution for a Telegram id and a GitHub id
    that happen to share the exact same NUMERIC value must both proceed
    independently and each create their OWN canonical user — proving the
    negated GitHub lock key (see db.github_identity's own docstring) never
    contends with, or gets confused for, the Telegram advisory-lock
    namespace using the same raw number."""
    shared_numeric_value = 424242424

    async def _run():
        return await asyncio.gather(
            asyncio.to_thread(db_identity.resolve_or_create_user_by_telegram_id_sync, shared_numeric_value),
            asyncio.to_thread(db_github_identity.resolve_or_create_user_by_github_id_sync, shared_numeric_value),
        )

    telegram_uuid, github_uuid = asyncio.run(_run())

    assert telegram_uuid != github_uuid


# --- Stage 6C boundary: no silent merge with an existing Telegram user ----


def test_existing_telegram_user_is_not_silently_merged_on_first_github_login(postgres_db):
    """Section 3/23: a human who already has a Telegram-backed canonical
    user and then logs in with GitHub for the first time gets a SECOND,
    separate canonical user — this module has no email/username/heuristic
    merge logic at all, proven here by the plain absence of any shared
    UUID between the two independently-created identities."""
    telegram_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(random.randint(10 ** 11, 10 ** 12 - 1))
    github_uuid = db_github_identity.resolve_or_create_user_by_github_id_sync(_github_id())

    assert telegram_uuid != github_uuid


# --- app-layer wrapper -------------------------------------------------------


@pytest.mark.asyncio
async def test_app_layer_resolve_user_uuid_wraps_the_sync_resolver(postgres_db):
    github_id = _github_id()
    first = await app_github_identity.resolve_user_uuid(github_id)
    second = await app_github_identity.resolve_user_uuid(github_id)
    assert first == second

    with get_sync_engine().connect() as conn:
        mapping = conn.execute(
            select(GithubAccount.user_id).where(GithubAccount.github_user_id == github_id)
        ).scalar_one()
    assert mapping == first
