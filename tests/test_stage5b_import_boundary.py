"""
Stage 5B regression test: the application layer (app.tutor, app.session,
app.documents) must be importable and usable without importing telebot or
constructing a Telegram bot instance — the whole point of extracting it
so a future FastAPI adapter can reuse it directly.

This must run in a FRESH subprocess: by the time most of this test
session's other modules run, handlers/*.py (and therefore telebot/bot.py)
have already been imported into THIS process's sys.modules, so an
in-process check could never distinguish "app.* doesn't import telebot"
from "something else already did, earlier in the session". A subprocess
gives a clean module cache with no such ordering dependency.

The subprocess only imports modules — it never calls configure_logging(),
never constructs the vector index, and never imports handlers/bot.py, so
it never touches the real bot.log/data/qdrant/managed-uploads state
(matching this suite's "no real runtime data mutated" requirement); the
one side effect of the transitively-imported config.py — DATA_DIR/
DOCUMENTS_DIR .mkdir(exist_ok=True) — is a no-op given those directories
already exist in this repository.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CHECK_SCRIPT = """
import sys

import app.tutor
import app.session
import app.documents

blocked = [m for m in sys.modules if m == "telebot" or m.startswith("telebot.") or m == "bot"]
assert not blocked, f"importing the application layer pulled in Telegram modules: {blocked}"
print("APPLICATION_LAYER_IMPORT_OK")
"""


def test_application_layer_importable_without_telebot_or_bot_instance():
    env = dict(os.environ)
    # Same dummy credentials tests/conftest.py assigns for the main
    # pytest process — config.py still validates these at import time
    # (Stage 5B did not remove that; see the final report's
    # "Configuration/import impact" section), but no real Telegram/OpenAI/
    # Anthropic call is ever made merely by importing these modules.
    env["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
    env["OPENAI_API_KEY"] = "sk-test-dummy-key"
    env["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-key"
    env["LLM_PROVIDER"] = "openai"
    for proxy_var in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy", "OPENAI_PROXY",
    ):
        env.pop(proxy_var, None)

    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"application-layer-only import failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "APPLICATION_LAYER_IMPORT_OK" in result.stdout
