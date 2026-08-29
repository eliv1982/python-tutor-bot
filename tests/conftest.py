"""
Pytest configuration for offline regression tests.

Credentials are assigned deterministically (plain assignment, not
setdefault) BEFORE any project module is imported. This guarantees tests
always run with dummy values and can never inherit a real token/key from
the ambient environment or a developer's local .env file: config.py's
load_dotenv() never overrides a variable that is already present in
os.environ.
"""

import logging
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

    Stage 2B-D: `rag/index.py`'s VectorIndex singleton and
    `utils/logging.py`'s FileHandler are no longer created merely by
    importing their modules (Blockers 4/G) — each now requires an EXPLICIT
    call (`get_vector_index()` / `configure_logging()`) before any Qdrant
    state or bot.log is created. This fixture therefore no longer needs to
    "win a race" against those modules' own first import; it only needs
    the redirected paths to be in place before whichever test is the FIRST
    to actually call one of those explicit entry points, anywhere in the
    session — which is trivially satisfied by doing the redirect here, in
    a hook that runs before any test module is even collected.

    Several application modules still read filesystem paths from config.py
    (and, since Stage 2B-D, from the pure rag/constants.py module some of
    those paths now canonically live in) — some bind a copy at their OWN
    first-import time (`from config import SOME_PATH`), some read
    `rag_constants.DATA_DIR` fresh at the point of use. Either way, they
    need a redirected value in place before they're ever imported/called
    for the first time in the session:

    - `rag/index.py`'s `get_vector_index()` singleton persists to
      `rag_constants.DATA_DIR / "qdrant"`, read fresh at construction time
      (never bound at rag.index's own import time — Stage 2B-D removed the
      eager `vector_index = VectorIndex()` singleton entirely).
    - `utils/logging.py`'s `configure_logging()` opens a `FileHandler` on
      `config.LOG_FILE`, read fresh at call time.
    - `utils/helpers.py`'s `save_file_async()` (used by the real voice
      handler to store a downloaded .ogg) reads `config.DATA_DIR` (bound at
      utils.helpers' own import time — Stage 1F-B remediation: this used to
      hardcode `BASE_DIR / "data"`, bypassing this redirect entirely, which
      is exactly how a real voice-handler test was found writing a real
      file under the repo's real `data/` directory).

    The redirect below is left in place for the ENTIRE test session (no
    restore): every module/call that reads config.DATA_DIR/LOG_FILE or
    rag_constants.DATA_DIR at any point in the session sees only the temp
    path. This changes no production code path outside pytest (config.py's
    and rag/constants.py's real defaults are untouched; only these already-
    imported modules' own copies of their attributes are patched).
    """
    import config as app_config
    import rag.constants as rag_constants

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
    # rag/index.py's get_vector_index() reads rag_constants.DATA_DIR (not
    # config.DATA_DIR) for its default persist_directory — see rag/index.py
    # Section H. Redirected here too so the FIRST EXPLICIT get_vector_index()
    # call anywhere in the session — whenever/wherever that happens to be —
    # never resolves into the real repository's data/qdrant.
    rag_constants.DATA_DIR = tmp_data_dir
    rag_constants.DOCUMENTS_DIR = tmp_data_dir / "documents"
    rag_constants.MANAGED_UPLOADS_DIR = rag_constants.DOCUMENTS_DIR / "uploads"
    # Stage 2B-E Section M (Codex non-blocking finding): config.py
    # re-exports DOCUMENTS_DIR/MANAGED_UPLOADS_DIR too (bound at config's
    # own first-import time, same as DATA_DIR above) — redirecting only
    # rag_constants' copies left app_config.DOCUMENTS_DIR/MANAGED_UPLOADS_DIR
    # still pointing at the real repository paths for any test/module that
    # reads them via `from config import ...` / `config.DOCUMENTS_DIR`. No
    # test actually leaked real state through this gap, but it's a latent
    # footgun for the next one that does — redirect both here too, to the
    # SAME temp values rag_constants already uses.
    app_config.DOCUMENTS_DIR = rag_constants.DOCUMENTS_DIR
    app_config.MANAGED_UPLOADS_DIR = rag_constants.MANAGED_UPLOADS_DIR

    tmp_log_file = session_root / "logs" / "bot.log"
    tmp_log_file.parent.mkdir()
    app_config.LOG_FILE = tmp_log_file

    # Explicit call (Stage 2B-D Section G) — installs a real FileHandler
    # against tmp_log_file for the duration of the test session, exactly
    # mirroring production's real startup behavior but against a temp path.
    # Never touches the real, non-redirected bot.log.
    from utils.logging import configure_logging
    configure_logging()

    # Stage 2B-C Section K (Codex finding): pytest.Config.add_cleanup()
    # callbacks run in LIFO order (last registered runs FIRST), so
    # registering this AFTER the rmtree cleanup above means it runs BEFORE
    # it — closing the shared VectorIndex singleton's local-persistent
    # Qdrant client (releasing its storage-path lock/file handles), if one
    # was ever constructed this session, and the shared logger's
    # FileHandler (flushing/releasing tmp_log_file) BEFORE shutil.rmtree()
    # ever attempts to remove session_root. Without this, Windows keeps
    # those handles open past the end of the test session, and
    # shutil.rmtree(..., ignore_errors=True) then silently leaves
    # session_root undeleted (a PermissionError swallowed by
    # ignore_errors) instead of actually reclaiming the disposable temp
    # tree. This never touches the real, non-redirected bot.log.
    def _close_session_resources():
        import rag.index
        import utils.logging

        try:
            rag.index.close_vector_index()
        except Exception:
            pass
        for handler in list(utils.logging.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                except Exception:
                    pass
                utils.logging.logger.removeHandler(handler)

    config.add_cleanup(_close_session_resources)
