"""
Pytest configuration for offline regression tests.

Credentials are assigned deterministically (plain assignment, not
setdefault) BEFORE any project module is imported. This guarantees tests
always run with dummy values and can never inherit a real token/key from
the ambient environment or a developer's local .env file: config.py's
load_dotenv() never overrides a variable that is already present in
os.environ.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"


@pytest.fixture(autouse=True)
def _default_test_access_allowed(monkeypatch):
    """
    Stage 1C added a fail-closed Telegram access gate (utils.access_control)
    in front of every handler: with no TELEGRAM_ALLOWED_USER_IDS configured
    (the default in this test environment), every user_id is denied.

    Tests written before/independent of that feature call handlers with
    arbitrary user_ids and don't expect to be denied, so default every test
    to "authorized" here. tests/test_stage1c_access_control.py — which
    exercises the gate itself — defines a same-named fixture that shadows
    this one for that module, leaving the real is_authorized() in place so
    it can monkeypatch TELEGRAM_ALLOWED_USER_IDS and assert on real
    allow/deny behavior.
    """
    import utils.access_control as access_control
    monkeypatch.setattr(access_control, "is_authorized", lambda user_id: True)


def pytest_configure(config):
    """
    Test-only isolation, run once before any test module is collected.

    Two application modules bind filesystem paths to module-level names at
    *their own* import time via `from config import SOME_PATH` (a copy,
    not a live reference): `rag/index.py` (its `vector_index = VectorIndex()`
    singleton persists to `DATA_DIR / "chroma_db"`) and `utils/logging.py`
    (its `logger` singleton opens a `FileHandler` on `LOG_FILE`). Once
    either module has been imported once, patching config.py's attributes
    afterward has no effect on their already-bound names.

    So: temporarily redirect config.py's DATA_DIR / LOG_FILE to fresh temp
    directories, proactively trigger each module's *first* import while
    the redirect is active (binding their singletons to the temp paths),
    then restore the real config values immediately. Every later import
    anywhere in the test session (module-cached by Python) reuses those
    already-isolated singletons, so the real, gitignored data/chroma_db
    and bot.log are never read, created, or modified by running tests.

    This changes no production code and no production behavior.
    """
    import config as app_config

    real_log_file = app_config.LOG_FILE
    tmp_log_file = Path(tempfile.mkdtemp(prefix="pytest_bot_log_")) / "bot.log"
    app_config.LOG_FILE = tmp_log_file
    try:
        import utils.logging  # noqa: F401  (binds its FileHandler to tmp_log_file)
    finally:
        app_config.LOG_FILE = real_log_file

    real_data_dir = app_config.DATA_DIR
    tmp_data_dir = Path(tempfile.mkdtemp(prefix="pytest_rag_chroma_"))
    app_config.DATA_DIR = tmp_data_dir
    try:
        import rag.index  # noqa: F401  (binds vector_index singleton to tmp_data_dir/chroma_db)
    finally:
        app_config.DATA_DIR = real_data_dir
