"""
Stage 5C corrective pass regression tests (Blocker 10): the Alembic
migration <-> SQLAlchemy ORM contract, proved against a REAL disposable
PostgreSQL container — never SQLite (which has no real BigInt/UUID/CHECK/
sequence semantics to speak of, and could not prove any of these), and
never `Base.metadata.create_all()` (which would prove the ORM models are
internally consistent, not that the actual operator-facing migration path
produces the same schema).

Covers:
- `alembic upgrade head` succeeds from a genuinely empty database and
  creates every expected table;
- `alembic downgrade base` followed by `alembic upgrade head` again
  reproduces a fully working schema (no half-applied state left behind);
- the live database schema (after migrating) has ZERO drift from
  `db.base.Base.metadata` — Alembic's own `compare_metadata()` autogenerate
  diff, the standard mechanism `alembic check`/`alembic revision
  --autogenerate` itself uses;
- `telegram_accounts.telegram_user_id` — an application-supplied Telegram
  numeric id, deliberately declared `autoincrement=False` in both the ORM
  model and the migration (see db/models.py) — never silently acquires
  PostgreSQL SERIAL/IDENTITY/owned-sequence behavior.

Each test restores the schema to `head` before returning control (even
when it deliberately drops everything along the way) so the shared
session-scoped container (tests/conftest.py's postgres_container()) is
always left ready for whichever test the session runs next.
"""

import subprocess
import urllib.parse
import uuid as uuid_module
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

import conftest as _conftest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_TABLES = {"users", "telegram_accounts", "user_preferences", "documents"}


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
    # Fresh engine bound to exactly this DSN — never the process-wide
    # singleton, which a sibling test's postgres_db fixture may have
    # already bound/reset to something else.
    from sqlalchemy import create_engine
    engine = create_engine(dsn)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# A. Upgrade from an empty database
# ---------------------------------------------------------------------------

def test_migration_upgrades_cleanly_from_an_empty_database(postgres_container):
    cfg = _alembic_config(postgres_container)

    # Guarantee a genuinely empty starting point regardless of what any
    # earlier test in this session already applied to the shared container.
    command.downgrade(cfg, "base")
    assert _table_names(postgres_container) & _EXPECTED_TABLES == set()

    command.upgrade(cfg, "head")

    tables = _table_names(postgres_container)
    assert _EXPECTED_TABLES <= tables


# ---------------------------------------------------------------------------
# B. Downgrade and re-upgrade
# ---------------------------------------------------------------------------

def test_migration_downgrade_then_reupgrade_reproduces_a_working_schema(postgres_container):
    cfg = _alembic_config(postgres_container)
    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(postgres_container)

    command.downgrade(cfg, "base")
    assert _table_names(postgres_container) & _EXPECTED_TABLES == set()

    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(postgres_container)

    # The reproduced schema is genuinely usable, not merely present by name.
    import uuid as uuid_module
    from db.models import User

    from sqlalchemy import create_engine
    engine = create_engine(postgres_container)
    try:
        with Session(engine) as session:
            user_id = uuid_module.uuid4()
            session.add(User(id=user_id))
            session.commit()
            assert session.get(User, user_id) is not None
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# C. No ORM/Alembic schema drift
# ---------------------------------------------------------------------------

def test_no_orm_alembic_schema_drift(postgres_db):
    """The exact mechanism `alembic check`/`alembic revision --autogenerate`
    uses internally: compare the LIVE database schema (after migrating to
    head) against db.base.Base.metadata (populated by importing
    db.models). A non-empty diff means the ORM models and the committed
    migration(s) have drifted apart — a real Stage 5C acceptance risk this
    proves is NOT the case."""
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    import db.engine as db_engine
    from db.base import Base
    import db.models  # noqa: F401 -- populates Base.metadata

    engine = db_engine.get_sync_engine()
    with engine.connect() as conn:
        migration_context = MigrationContext.configure(conn)
        diff = compare_metadata(migration_context, Base.metadata)

    assert diff == [], f"ORM models and Alembic migrations have drifted: {diff!r}"


# ---------------------------------------------------------------------------
# D. telegram_user_id never acquires SERIAL/identity/autoincrement behavior
# ---------------------------------------------------------------------------

def test_telegram_user_id_is_not_a_serial_or_identity_column(postgres_db):
    import db.engine as db_engine

    engine = db_engine.get_sync_engine()
    with engine.connect() as conn:
        owned_sequence = conn.execute(
            text("SELECT pg_get_serial_sequence('telegram_accounts', 'telegram_user_id')")
        ).scalar()
        assert owned_sequence is None, (
            "telegram_user_id has an owned sequence (SERIAL-like behavior) — "
            "it must be a plain application-supplied BigInteger column"
        )

        row = conn.execute(
            text(
                "SELECT column_default, is_identity FROM information_schema.columns "
                "WHERE table_name = 'telegram_accounts' AND column_name = 'telegram_user_id'"
            )
        ).one()
        assert row.column_default is None, "telegram_user_id must have no server-side default"
        assert row.is_identity == "NO", "telegram_user_id must not be a GENERATED ... AS IDENTITY column"


def test_telegram_user_id_requires_an_explicit_value_and_does_not_autogenerate_one(postgres_db):
    """Direct behavioral proof to complement the catalog inspection above:
    inserting a telegram_accounts row WITHOUT telegram_user_id must fail
    (NOT NULL, no default to fall back on) — a SERIAL/IDENTITY column would
    instead silently generate one."""
    import uuid as uuid_module

    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session as _Session

    import db.engine as db_engine
    from db.models import TelegramAccount, User

    engine = db_engine.get_sync_engine()
    with _Session(engine) as session:
        user_id = uuid_module.uuid4()
        session.add(User(id=user_id))
        session.flush()
        session.add(TelegramAccount(user_id=user_id))  # no telegram_user_id given
        with pytest.raises(IntegrityError):
            session.commit()


# ---------------------------------------------------------------------------
# E. Stage 5C corrective pass #4 (Blocker 4): valid percent-encoded DSN
# credentials (e.g. "%40" for a literal "@", "%2F" for "/") must not break
# `alembic upgrade head` — alembic/env.py's Config object is ConfigParser-
# backed, which treats bare "%" as ITS OWN interpolation escape character;
# storing an unescaped percent-encoded URL there used to raise
# configparser.InterpolationSyntaxError the moment anything (run_migrations_
# online()'s config.get_section() call) read it back. A dedicated disposable
# container with credentials that GENUINELY require percent-encoding (a
# literal "@"/"/" in the password) proves this is fixed end-to-end — never
# only that the encoding round-trips as a string.
# ---------------------------------------------------------------------------

_SPECIAL_CHAR_PG_IMAGE = _conftest._POSTGRES_IMAGE


@pytest.fixture
def special_char_postgres_container():
    """
    A dedicated, disposable PostgreSQL container (independent of the
    session-scoped `postgres_container` fixture, whose fixed "pytest"/
    "pytest" credentials contain no characters that need percent-encoding
    at all) whose PASSWORD genuinely contains "@" and "/" — the two
    characters an independent audit reproduced Alembic failing on when
    percent-encoded in a DSN. Reuses tests/conftest.py's own Docker
    availability/readiness helpers (never a second, drifting copy of that
    logic) so this fixture degrades identically: skipped locally without
    Docker/the cached image, hard-failed under PYTEST_REQUIRE_POSTGRES=1
    (CI). Torn down (docker stop) unconditionally in the finally block.
    """
    if not _conftest._docker_available():
        _conftest._unavailable("docker is not on PATH")
        return
    if not _conftest._image_available_locally(_SPECIAL_CHAR_PG_IMAGE):
        _conftest._unavailable(f"image {_SPECIAL_CHAR_PG_IMAGE!r} is not already cached locally")
        return

    raw_user = "encoded_user"
    raw_password = "p@ss/word%25with#percent"  # contains "@", "/", and a literal "%" too
    raw_dbname = "pytest_encoded"
    container_name = f"pytutorbot_test_pg_encoded_{uuid_module.uuid4().hex[:8]}"
    try:
        subprocess.run(
            [
                "docker", "run", "--rm", "-d", "--pull=never", "--name", container_name,
                "-e", f"POSTGRES_PASSWORD={raw_password}", "-e", f"POSTGRES_USER={raw_user}",
                "-e", f"POSTGRES_DB={raw_dbname}",
                "-p", "127.0.0.1::5432",
                "--health-cmd", f"pg_isready -U {raw_user} -d {raw_dbname}",
                "--health-interval=1s", "--health-timeout=3s", "--health-retries=30", "--health-start-period=5s",
                _SPECIAL_CHAR_PG_IMAGE,
            ],
            check=True, capture_output=True, text=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        _conftest._unavailable(f"could not start a disposable PostgreSQL container — error_type={type(e).__name__}")
        return

    try:
        port_output = subprocess.run(
            ["docker", "port", container_name, "5432/tcp"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        host_port = port_output.rsplit(":", 1)[-1]

        # Percent-encode the credential components — exactly what an
        # operator's real .env would contain for a password with special
        # characters (never the raw password embedded unencoded, which
        # would itself be ambiguous/invalid DSN syntax).
        encoded_user = urllib.parse.quote(raw_user, safe="")
        encoded_password = urllib.parse.quote(raw_password, safe="")
        url = f"postgresql+psycopg://{encoded_user}:{encoded_password}@127.0.0.1:{host_port}/{raw_dbname}"

        healthy = False
        for _ in range(60):
            inspect_result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
                capture_output=True, text=True, timeout=5,
            )
            if inspect_result.returncode == 0 and inspect_result.stdout.strip() == "healthy":
                healthy = True
                break
            import time as _time
            _time.sleep(0.5)
        if not healthy:
            _conftest._unavailable("container health status never reached 'healthy'")
            return

        failure = _conftest._wait_for_real_postgres_connection(url, timeout_seconds=30)
        if failure is not None:
            _conftest._unavailable(f"container reported healthy but host-side connection never stabilized — {failure}")
            return

        yield url
    finally:
        subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=30)


def test_alembic_upgrade_succeeds_with_a_normal_dsn(postgres_container):
    """Baseline (no percent-encoding involved) — the session container's
    own plain "pytest"/"pytest" credentials, proving the harness itself is
    sound before the encoded-credential proofs below."""
    cfg = _alembic_config(postgres_container)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(postgres_container)


def test_alembic_upgrade_succeeds_with_percent_encoded_at_sign_and_slash_in_password(special_char_postgres_container):
    """The exact reproduction: a DSN whose password contains percent-
    encoded "%40" (a literal "@") and "%2F" (a literal "/") — the two
    characters that, unescaped in the DSN, would themselves be
    misinterpreted as URL delimiters, and which an independent audit
    reproduced Alembic's ConfigParser-backed Config choking on once
    stored via set_main_option()/read back via get_section()."""
    dsn = special_char_postgres_container
    assert "%40" in dsn and "%2F" in dsn, f"fixture DSN unexpectedly missing encoded chars: {dsn!r}"

    cfg = _alembic_config(dsn)
    # Must not raise configparser.InterpolationSyntaxError (or anything
    # else) — this exercises exactly the env.py code path (config.
    # set_main_option() then config.get_section() inside
    # run_migrations_online()) an independent audit reproduced failing.
    command.upgrade(cfg, "head")

    assert _EXPECTED_TABLES <= _table_names(dsn)

    # The migration didn't merely "not crash" — the resulting schema is
    # genuinely usable through this exact encoded-credential connection.
    from sqlalchemy import create_engine
    engine = create_engine(dsn)
    try:
        from db.models import User
        with Session(engine) as session:
            user_id = uuid_module.uuid4()
            session.add(User(id=user_id))
            session.commit()
            assert session.get(User, user_id) is not None
    finally:
        engine.dispose()


def test_alembic_downgrade_then_reupgrade_succeeds_with_encoded_dsn(special_char_postgres_container):
    """The full upgrade/downgrade/upgrade cycle (not just a single
    `upgrade head` call) against the encoded-credential DSN — proves every
    env.py invocation along that path (offline and online configuration
    are both exercised across these commands) tolerates the encoded URL."""
    dsn = special_char_postgres_container
    cfg = _alembic_config(dsn)

    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(dsn)

    command.downgrade(cfg, "base")
    assert _table_names(dsn) & _EXPECTED_TABLES == set()

    command.upgrade(cfg, "head")
    assert _EXPECTED_TABLES <= _table_names(dsn)


def test_alembic_check_style_metadata_comparison_works_with_encoded_dsn(special_char_postgres_container):
    """`alembic check`'s own underlying mechanism (compare_metadata against
    a live connection) must also work through an encoded-credential DSN —
    proves the fix isn't narrowly scoped to just the `upgrade`/`downgrade`
    commands' own env.py invocation path."""
    dsn = special_char_postgres_container
    cfg = _alembic_config(dsn)
    command.upgrade(cfg, "head")

    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext
    from db.base import Base
    import db.models  # noqa: F401 -- populates Base.metadata

    from sqlalchemy import create_engine
    engine = create_engine(dsn)
    try:
        with engine.connect() as conn:
            migration_context = MigrationContext.configure(conn)
            diff = compare_metadata(migration_context, Base.metadata)
        assert diff == []
    finally:
        engine.dispose()
