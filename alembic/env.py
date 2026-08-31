"""
Alembic environment (Stage 5C).

sqlalchemy.url is resolved from db.settings.DATABASE_URL at RUNTIME,
never hardcoded in alembic.ini — this lets the Docker-Postgres test
fixture call alembic.command.upgrade(cfg, "head") programmatically
against a dynamically assigned container port (by pre-setting
config.attributes["sqlalchemy_url"] on the Config object before invoking
the command), while a plain operator `alembic upgrade head` picks up
db.settings.DATABASE_URL the normal way. Sync-mode migrations only
(psycopg's sync driver) — no async-migration machinery needed here.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import db.settings as db_settings
from db.base import Base
import db.models  # noqa: F401 -- populates Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Attribute injection (used by the test fixture) takes priority over the
# ini file's own [alembic] section, which in turn takes priority over
# db.settings.DATABASE_URL's own env/default resolution.
_url = config.attributes.get("sqlalchemy_url") or config.get_main_option("sqlalchemy.url") or db_settings.DATABASE_URL

# Stage 5C corrective pass #4 (Blocker 4): this Config object's ini-backed
# storage is a plain configparser.ConfigParser, which treats "%" as its own
# interpolation escape character. A percent-encoded DSN credential (e.g.
# "%40" for a literal "@", "%2F" for "/") stored verbatim via
# set_main_option() would make a LATER config.get_main_option("sqlalchemy.
# url")/config.get_section(...) call (run_migrations_online() below calls
# the latter) raise configparser.InterpolationSyntaxError, since a bare
# "%40" is not valid "%%" or "%(name)s" interpolation syntax — this is
# exactly the failure an independent audit reproduced for valid encoded
# PostgreSQL DSNs. Doubling every "%" here is ConfigParser's own documented
# escape for a literal percent sign: interpolation collapses "%%" back to
# exactly one "%" on retrieval, so get_main_option()/get_section() continue
# to round-trip the exact original URL unchanged (including any encoded
# "%40"/"%2F"/"%25" sequences it contains) rather than ever raising. Never
# weakens secret handling — this only changes how the value survives this
# one config object's own interpolation, never what credential it decodes
# to or whether it gets logged.
config.set_main_option("sqlalchemy.url", _url.replace("%", "%%"))


def run_migrations_offline() -> None:
    context.configure(
        url=_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
