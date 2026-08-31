"""
Side-effect-free-ish database configuration (Stage 5C).

Mirrors rag/constants.py's contract: importable with ZERO credentials —
no client construction, no schema creation, no raise-on-missing-value (a
missing/invalid DATABASE_URL surfaces naturally the first time db/engine.py
actually constructs an engine, not merely by importing this module). This
is what lets config.py, app/*.py, and db/*.py stay importable without
TELEGRAM_BOT_TOKEN (see telegram_config.py for the credential this module
deliberately does NOT require).

Calls load_dotenv() itself (idempotent — python-dotenv's default
override=False never clobbers an already-set os.environ value) rather than
relying on config.py having already done so: this module must be
independently importable by Alembic's env.py or scripts/migrate_sidecars_v2_to_v3.py
in a process where config.py never runs at all.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Local-development default matches this repo's docker-based disposable
# Postgres test convention (see tests/conftest.py's Docker-Postgres
# fixture) — never a production credential. Real deployments must set
# DATABASE_URL explicitly via .env.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://python_tutor:python_tutor@localhost:5432/python_tutor_bot"
)
