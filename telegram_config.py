"""
Telegram-specific credential validation (Stage 5C).

Split out of config.py: the shared application/persistence layer
(config.py, db/*.py, app/*.py) must be importable without
TELEGRAM_BOT_TOKEN — a future FastAPI adapter, and anything that merely
needs OpenAI/Anthropic/database configuration, has no reason to require a
Telegram credential merely to import. Only the Telegram adapter itself
(bot.py, and transitively handlers/*.py, main.py) imports this module, and
it still fails fast (raises at import time) exactly like config.py's other
credential checks — starting the Telegram adapter with a missing/invalid
token is still a hard, immediate failure, not silently deferred.
"""

import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env file")
