"""
PostgreSQL persistence layer (Stage 5C).

Concrete persistence operations, not a generic repository framework: each
submodule owns one entity family (users/telegram_accounts, preferences,
documents) and exposes plain functions, mirroring rag/index.py's existing
"lazy singleton + explicit init/close" convention for the shared Qdrant
client rather than introducing a new DI/ORM abstraction style.

db/settings.py is importable with zero credentials of any kind (same
contract as rag/constants.py) — this is what lets config.py/app/*.py be
importable without TELEGRAM_BOT_TOKEN (see telegram_config.py).
"""
