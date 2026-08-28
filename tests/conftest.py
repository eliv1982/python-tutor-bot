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
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-key"

# Stage 2A: the accepted 137-test baseline mocks OpenAI at the SDK boundary
# (openai_client.client.chat.completions.create) throughout. Pinning the
# test-session provider to "openai" here keeps every one of those existing
# mocks valid unchanged — router.py/rag/query.py now call services/text_llm.py,
# which itself dispatches to services/openai_client.py when LLM_PROVIDER is
# "openai", exactly the object those tests already patch. This is TEST-ONLY
# compatibility behavior: config.py's own default (LLM_PROVIDER=anthropic)
# is what actually ships to production and is completely unaffected by this
# override. tests/test_stage2a_text_llm_provider.py exercises the Anthropic
# path explicitly, per-test, via monkeypatch — see that file.
os.environ["LLM_PROVIDER"] = "openai"

# chromadb.config.Settings is a pydantic-settings BaseSettings model, so it
# picks up ANONYMIZED_TELEMETRY from the environment automatically (no
# Chroma Settings object needs to be constructed or threaded through
# rag/index.py for this). Set before rag.index (imported below) or anything
# else can construct a Chroma client, so the test suite's offline guarantee
# doesn't depend on chromadb's default best-effort telemetry call succeeding
# or failing quietly against a real endpoint.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- Stage 1F-B remediation: localhost-proxy bypass (Codex finding) -------
#
# pytest.ini enforces `--disable-socket --allow-hosts=127.0.0.1,::1`. The
# loopback allowance is required for Windows' asyncio ProactorEventLoop
# (see pytest.ini's comment), but it is a socket-layer allowance — it says
# nothing about what a client sends once connected. An independent audit
# demonstrated that a real HTTP(S) proxy listening on 127.0.0.1 could
# receive a `CONNECT api.openai.com:443 HTTP/1.1` and forward it externally,
# completely invisibly to pytest-socket, if any HTTP_PROXY/HTTPS_PROXY/
# ALL_PROXY environment variable pointed a trust_env-honoring client (the
# OpenAI SDK's underlying httpx2.Client defaults to `trust_env=True`) at a
# loopback address.
#
# Stage 1F-C remediation (second independent audit) established that this
# env-variable cleanup can only ever be defense-in-depth, never the primary
# guarantee: a trust_env=True client also auto-discovers a proxy from
# OS-level configuration (Windows Registry / macOS system config) even with
# every one of these variables absent, and langchain-openai's OpenAIEmbeddings
# separately honors its own OPENAI_PROXY variable. The primary guarantee is
# now that every provider HTTP client this app constructs is built with an
# explicit trust_env=False (services/openai_client.py, rag/index.py) —
# see tests/test_stage1f_offline_enforcement.py for the regression proof of
# all three bypass routes and why each is/isn't at risk.
#
# This env-variable-boundary cleanup remains worth keeping anyway: it's a
# second, independent layer that would still stop anything in this codebase
# that ever constructs a trust_env-honoring client WITHOUT going through the
# hardened production constructors above (e.g. a future ad-hoc script).
#
# Popped (never read/logged) before any project or provider module is
# imported, so no client constructed anywhere in the test session — now or
# later — can pick up a proxy from the ambient environment. OPENAI_PROXY is
# included even though it isn't a "conventional" proxy variable: it's the
# provider-specific one langchain-openai's OpenAIEmbeddings reads directly
# (see rag/index.py).
PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "OPENAI_PROXY",
)


def neutralize_proxy_env() -> None:
    """Remove every conventional + provider-specific proxy env var.

    Values are never read/logged.
    """
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)


neutralize_proxy_env()


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

    Several application modules read filesystem paths from config.py — some
    bind a copy at their OWN first-import time (`from config import
    SOME_PATH`), some read `config.DATA_DIR` fresh on every call. Either
    way, they need a redirected value in place before they're ever
    imported/called for the first time in the session:

    - `rag/index.py`'s `vector_index = VectorIndex()` singleton persists to
      `DATA_DIR / "chroma_db"` (bound at rag.index's own import time).
    - `utils/logging.py`'s `logger` singleton opens a `FileHandler` on
      `LOG_FILE` (bound at utils.logging's own import time).
    - `utils/helpers.py`'s `save_file_async()` (used by the real voice
      handler to store a downloaded .ogg) reads `DATA_DIR` (bound at
      utils.helpers' own import time — Stage 1F-B remediation: this used to
      hardcode `BASE_DIR / "data"`, bypassing this redirect entirely, which
      is exactly how a real voice-handler test was found writing a real
      file under the repo's real `data/` directory).

    Earlier revisions of this fixture redirected DATA_DIR/LOG_FILE only for
    the duration of the two proactive imports below, then restored the real
    values — which meant any module imported LATER during collection (like
    utils/helpers.py, imported by ordinary test files, not by this fixture)
    would bind to the REAL path instead. The redirect below is now left in
    place for the ENTIRE test session (no restore): every module that reads
    config.DATA_DIR/LOG_FILE at any point in the session — proactively
    triggered here or naturally imported later during collection — sees
    only the temp path. This changes no production code path outside
    pytest (config.py's real defaults are untouched; only this module's own
    copy of the already-imported `config` module's attributes is patched).
    """
    import config as app_config

    # Stage 1F-C: re-run the same neutralization AFTER config.py's own
    # load_dotenv() has already executed (triggered by the `import config`
    # above, config.py's own first import in the session). load_dotenv()
    # defaults to override=False, but that only means it won't clobber a
    # variable that's already *present* — the module-level neutralize call
    # above removed these variables entirely, so from load_dotenv()'s point
    # of view they're simply unset and get reintroduced from the developer's
    # real .env file if it happens to define any of them. Popping them again
    # here closes that gap for the remainder of the session. This is
    # explicitly defense-in-depth, not the primary guarantee: the primary
    # guarantee is that services/openai_client.py and rag/index.py build
    # their provider HTTP clients with trust_env=False, so even a variable
    # that DID survive both neutralization passes could not be used to
    # redirect either client. See PROXY_ENV_VARS' comment above.
    neutralize_proxy_env()

    session_root = Path(tempfile.mkdtemp(prefix="pytest_pytutorbot_session_"))
    config.add_cleanup(lambda: shutil.rmtree(session_root, ignore_errors=True))

    tmp_data_dir = session_root / "data"
    tmp_data_dir.mkdir()
    app_config.DATA_DIR = tmp_data_dir

    tmp_log_file = session_root / "logs" / "bot.log"
    tmp_log_file.parent.mkdir()
    app_config.LOG_FILE = tmp_log_file

    import utils.logging  # noqa: F401  (binds its FileHandler to tmp_log_file)
    import rag.index  # noqa: F401  (binds vector_index singleton to tmp_data_dir/chroma_db)
