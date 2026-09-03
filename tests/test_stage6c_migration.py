"""
Stage 6C regression tests: the 0004_telegram_link_attempts migration <->
db.models.TelegramLinkAttempt ORM contract — real disposable PostgreSQL,
mirroring tests/test_stage5c_alembic_migration.py's own conventions
exactly (never SQLite, never Base.metadata.create_all()).
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from pathlib import Path

_FUTURE = lambda: datetime.now(timezone.utc) + timedelta(minutes=10)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_TABLES = {
    "users", "telegram_accounts", "user_preferences", "documents", "web_sessions",
    "github_accounts", "github_oauth_transactions", "telegram_link_attempts",
    "github_unlink_tombstones",
}


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    exercises the REAL PostgreSQL schema/migration path directly."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    yield


def _alembic_config(dsn: str) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.attributes["sqlalchemy_url"] = dsn
    return cfg


def _table_names(dsn: str) -> set:
    from sqlalchemy import create_engine
    engine = create_engine(dsn)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_migration_upgrades_cleanly_and_creates_telegram_link_attempts(postgres_container):
    cfg = _alembic_config(postgres_container)
    command.downgrade(cfg, "base")
    assert _table_names(postgres_container) & _EXPECTED_TABLES == set()

    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(postgres_container)


def test_downgrade_then_reupgrade_reproduces_a_working_schema(postgres_container):
    cfg = _alembic_config(postgres_container)
    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(postgres_container)

    command.downgrade(cfg, "base")
    assert _table_names(postgres_container) & _EXPECTED_TABLES == set()

    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(postgres_container)

    from sqlalchemy import create_engine
    from db.models import TelegramLinkAttempt, User

    engine = create_engine(postgres_container)
    try:
        with Session(engine) as session:
            user_id = uuid.uuid4()
            session.add(User(id=user_id))
            session.flush()
            session.add(
                TelegramLinkAttempt(
                    web_user_id=user_id,
                    link_secret_hash=b"\x01" * 32,
                    expires_at=_FUTURE(),
                )
            )
            session.execute(text("SELECT 1"))
            session.commit()
    finally:
        engine.dispose()


def test_downgrade_only_drops_telegram_link_attempts_and_generation_protocol(postgres_db):
    """0004 must be reversible in isolation — downgrading to 0003 removes
    telegram_link_attempts AND the generation/tombstone protocol additions
    (unlink_generation, auth_generation, github_unlink_tombstones) this
    same migration was later amended to add (Stage 6C corrective pass,
    independent-audit MAJOR 1), leaving every earlier table/column
    intact."""
    from sqlalchemy import create_engine

    dsn = postgres_db
    cfg = _alembic_config(dsn)
    command.downgrade(cfg, "0003")
    tables = _table_names(dsn)
    assert "telegram_link_attempts" not in tables
    assert "github_unlink_tombstones" not in tables
    assert {"users", "github_accounts", "web_sessions"} <= tables

    def _columns(table_name: str) -> set:
        engine = create_engine(dsn)
        try:
            return {c["name"] for c in inspect(engine).get_columns(table_name)}
        finally:
            engine.dispose()

    assert "unlink_generation" not in _columns("github_oauth_admission")
    assert "auth_generation" not in _columns("github_oauth_transactions")

    command.upgrade(cfg, "head")
    tables = _table_names(dsn)
    assert "telegram_link_attempts" in tables
    assert "github_unlink_tombstones" in tables
    assert "unlink_generation" in _columns("github_oauth_admission")
    assert "auth_generation" in _columns("github_oauth_transactions")


# ---------------------------------------------------------------------------
# D. OAuth generation/tombstone protocol schema (Stage 6C corrective pass,
# independent-audit MAJOR 1)
# ---------------------------------------------------------------------------


def test_admission_unlink_generation_defaults_to_zero_and_rejects_negative(postgres_db):
    from sqlalchemy import select

    from db.engine import get_sync_engine
    from db.models import GITHUB_OAUTH_ADMISSION_ID, GithubOAuthAdmission

    engine = get_sync_engine()
    with Session(engine) as session:
        value = session.execute(
            select(GithubOAuthAdmission.unlink_generation).where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
        ).scalar_one()
    assert value == 0

    with Session(engine) as session:
        with pytest.raises(IntegrityError):
            session.execute(
                text("UPDATE github_oauth_admission SET unlink_generation = -1 WHERE id = :id"),
                {"id": GITHUB_OAUTH_ADMISSION_ID},
            )
            session.commit()


def test_oauth_transaction_auth_generation_defaults_to_zero_and_rejects_negative(postgres_db):
    from sqlalchemy import select

    from db.engine import get_sync_engine
    from db.models import GithubOAuthTransaction

    engine = get_sync_engine()
    with Session(engine) as session:
        session.add(GithubOAuthTransaction(state_hash=b"\x09" * 32, code_verifier="v", expires_at=_FUTURE()))
        session.commit()
        value = session.execute(
            select(GithubOAuthTransaction.auth_generation).where(GithubOAuthTransaction.state_hash == b"\x09" * 32)
        ).scalar_one()
    assert value == 0

    with Session(engine) as session:
        with pytest.raises(IntegrityError):
            session.add(
                GithubOAuthTransaction(
                    state_hash=b"\x0a" * 32, code_verifier="v2", expires_at=_FUTURE(), auth_generation=-1
                )
            )
            session.commit()


def test_github_unlink_tombstones_shape_and_no_foreign_key(postgres_db):
    from db.engine import get_sync_engine
    from db.models import GithubUnlinkTombstone

    engine = get_sync_engine()
    with engine.connect() as conn:
        pk_cols = inspect(conn).get_pk_constraint("github_unlink_tombstones")["constrained_columns"]
        assert pk_cols == ["github_user_id"]
        fks = inspect(conn).get_foreign_keys("github_unlink_tombstones")
        assert fks == []  # deliberately no FK to github_accounts — see db/models.py's docstring

    with Session(engine) as session:
        session.add(GithubUnlinkTombstone(github_user_id=424242, unlink_generation=1))
        session.commit()
        row = session.get(GithubUnlinkTombstone, 424242)
        assert row.unlink_generation == 1
        assert row.unlinked_at is not None

    with Session(engine) as session:
        with pytest.raises(IntegrityError):
            session.add(GithubUnlinkTombstone(github_user_id=515151, unlink_generation=0))
            session.commit()


def test_no_orm_alembic_schema_drift(postgres_db):
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    import db.engine as db_engine
    from db.base import Base
    import db.models  # noqa: F401

    engine = db_engine.get_sync_engine()
    with engine.connect() as conn:
        migration_context = MigrationContext.configure(conn)
        diff = compare_metadata(migration_context, Base.metadata)

    assert diff == [], f"ORM models and Alembic migrations have drifted: {diff!r}"


def test_web_user_id_is_primary_key_and_fk_to_users_with_restrict(postgres_db):
    from db.engine import get_sync_engine

    engine = get_sync_engine()
    with engine.connect() as conn:
        pk_cols = inspect(conn).get_pk_constraint("telegram_link_attempts")["constrained_columns"]
        assert pk_cols == ["web_user_id"]

        fks = inspect(conn).get_foreign_keys("telegram_link_attempts")
        assert len(fks) == 1
        assert fks[0]["referred_table"] == "users"
        assert fks[0]["constrained_columns"] == ["web_user_id"]
        assert fks[0]["options"].get("ondelete", "").upper() == "RESTRICT"


def test_link_secret_hash_is_unique_and_check_constrained_to_32_bytes(postgres_db):
    from db.engine import get_sync_engine
    from db.models import TelegramLinkAttempt, User

    engine = get_sync_engine()
    with Session(engine) as session:
        user_a, user_b = uuid.uuid4(), uuid.uuid4()
        session.add_all([User(id=user_a), User(id=user_b)])
        session.flush()
        digest = b"\x02" * 32
        session.add(TelegramLinkAttempt(web_user_id=user_a, link_secret_hash=digest, expires_at=_FUTURE()))
        session.commit()

    with Session(engine) as session:
        session.add(TelegramLinkAttempt(web_user_id=user_b, link_secret_hash=digest, expires_at=_FUTURE()))
        with pytest.raises(IntegrityError):
            session.commit()

    with Session(engine) as session:
        user_c = uuid.uuid4()
        session.add(User(id=user_c))
        session.flush()
        session.add(TelegramLinkAttempt(web_user_id=user_c, link_secret_hash=b"short", expires_at=_FUTURE()))
        with pytest.raises(IntegrityError):
            session.commit()


def test_users_row_with_an_outstanding_attempt_cannot_be_deleted(postgres_db):
    """Direct proof of the ON DELETE RESTRICT FK — a `users` row must never
    be deletable while it still holds an outstanding link attempt (Section
    E). Every code path that legitimately deletes a users row (merge,
    GitHub-only unlink) is required to delete the attempt row FIRST, in
    the same transaction — this constraint is the database-level backstop
    for that invariant."""
    from db.engine import get_sync_engine
    from db.models import TelegramLinkAttempt, User

    engine = get_sync_engine()
    with Session(engine) as session:
        user_id = uuid.uuid4()
        session.add(User(id=user_id))
        session.flush()
        session.add(
            TelegramLinkAttempt(web_user_id=user_id, link_secret_hash=b"\x03" * 32, expires_at=_FUTURE())
        )
        session.commit()

    with Session(engine) as session:
        # PostgreSQL checks a plain (non-deferrable) RESTRICT constraint
        # immediately, at statement-execution time — not deferred to
        # COMMIT — so the IntegrityError raises here, not at commit().
        with pytest.raises(IntegrityError):
            session.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
        session.rollback()


def test_expires_at_has_an_index(postgres_db):
    from db.engine import get_sync_engine

    engine = get_sync_engine()
    with engine.connect() as conn:
        indexes = inspect(conn).get_indexes("telegram_link_attempts")
    assert any(ix["column_names"] == ["expires_at"] for ix in indexes)
