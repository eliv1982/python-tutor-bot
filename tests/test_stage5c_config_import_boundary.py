"""
Stage 5C regression test: config.py/db/*.py/app/*.py must be importable
and usable WITHOUT TELEGRAM_BOT_TOKEN — the Telegram credential was moved
out of config.py entirely, into telegram_config.py (imported only by
bot.py and, transitively, handlers/*.py/main.py). Constructing the actual
Telegram bot instance (bot.py) still fails fast without it.

Mirrors tests/test_stage5b_import_boundary.py's subprocess-based
methodology exactly, for the same reason: a fresh subprocess is needed to
prove the import graph itself, independent of whatever this test session's
other modules have already pulled into sys.modules.

Neither subprocess touches real runtime state: no configure_logging(), no
vector index construction, no bot.log/data/qdrant/managed-uploads writes.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_IMPORT_WITHOUT_TELEGRAM_TOKEN_SCRIPT = """
import sys

import config
import db.settings
import db.base
import db.models
import db.engine
import db.identity
import db.preferences
import db.documents
import app.tutor
import app.session
import app.documents
import app.identity

blocked = [m for m in sys.modules if m == "telebot" or m.startswith("telebot.") or m == "bot" or m == "telegram_config"]
assert not blocked, f"importing config/db/app pulled in Telegram modules: {blocked}"
print("CONFIG_DB_APP_IMPORT_WITHOUT_TELEGRAM_TOKEN_OK")
"""

_BOT_CONSTRUCTION_REQUIRES_TELEGRAM_TOKEN_SCRIPT = """
import sys

try:
    import bot
    print("UNEXPECTED_BOT_IMPORT_SUCCEEDED")
except ValueError as e:
    assert "TELEGRAM_BOT_TOKEN" in str(e)
    print("BOT_CONSTRUCTION_FAILED_FAST_AS_EXPECTED")
"""


def _clean_env(*, with_telegram_token: bool) -> dict:
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "sk-test-dummy-key"
    env["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-key"
    env["LLM_PROVIDER"] = "openai"
    if with_telegram_token:
        env["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
    else:
        # Deliberately an explicit EMPTY string, not env.pop(...): the
        # subprocess's cwd is the real repository root, which has a real
        # .env file defining a real TELEGRAM_BOT_TOKEN. python-dotenv's
        # load_dotenv() (override=False, the default) never overrides a
        # variable already PRESENT in os.environ — merely popping the key
        # would leave it absent, and telegram_config.py's own
        # load_dotenv() call would then load the real .env's value right
        # back in, making this "missing token" test silently exercise the
        # "token present" path instead. An explicit empty string is
        # present (so load_dotenv() leaves it alone) and falsy (so
        # telegram_config.py's own `if not TELEGRAM_BOT_TOKEN:` still
        # raises) — the same technique tests/conftest.py's own module
        # docstring documents for this exact class of test.
        env["TELEGRAM_BOT_TOKEN"] = ""
    for proxy_var in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy", "OPENAI_PROXY",
    ):
        env.pop(proxy_var, None)
    return env


def test_config_db_app_layer_importable_without_telegram_bot_token():
    """Stage 5C acceptance invariant #9: importing shared application/
    persistence code must not require Telegram credentials. Real .env
    files elsewhere on this machine are irrelevant here — config.py and
    db/*.py simply no longer read/validate TELEGRAM_BOT_TOKEN at all, so
    this holds regardless of what any .env file happens to define."""
    env = _clean_env(with_telegram_token=False)

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_TELEGRAM_TOKEN_SCRIPT],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"config/db/app import without TELEGRAM_BOT_TOKEN failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "CONFIG_DB_APP_IMPORT_WITHOUT_TELEGRAM_TOKEN_OK" in result.stdout


def test_bot_construction_still_fails_fast_without_telegram_bot_token():
    """The Telegram adapter itself (bot.py, via telegram_config.py) must
    still fail fast — starting the Telegram adapter with a missing/invalid
    token remains a hard, immediate failure, exactly as before Stage 5C."""
    env = _clean_env(with_telegram_token=False)

    result = subprocess.run(
        [sys.executable, "-c", _BOT_CONSTRUCTION_REQUIRES_TELEGRAM_TOKEN_SCRIPT],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"bot construction fail-fast check errored:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "BOT_CONSTRUCTION_FAILED_FAST_AS_EXPECTED" in result.stdout
    assert "UNEXPECTED_BOT_IMPORT_SUCCEEDED" not in result.stdout


def test_bot_constructs_successfully_with_a_valid_telegram_bot_token():
    """Sanity counterpart: with a valid token present, bot.py imports
    (and constructs the AsyncTeleBot instance) without error — proves the
    fail-fast check above is actually exercising the token path, not
    failing for some unrelated reason."""
    env = _clean_env(with_telegram_token=True)

    result = subprocess.run(
        [sys.executable, "-c", "import bot; print('BOT_IMPORT_OK')"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"bot import with a valid token unexpectedly failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "BOT_IMPORT_OK" in result.stdout
