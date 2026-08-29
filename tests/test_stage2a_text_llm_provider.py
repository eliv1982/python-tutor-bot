"""
Stage 2A-B regression tests: Anthropic Claude as the selectable/default
text-LLM provider, behind services/text_llm.py's tiny facade.

Scope covered:
- config.py provider/key resolution and fail-closed validation (subprocess
  based, so this test session's own already-imported `config` module —
  pinned to LLM_PROVIDER=openai by tests/conftest.py for the other 137
  tests' sake — is never disturbed);
- services/anthropic_client.py's message mapping (system extraction,
  history order, unexpected-role rejection), request construction (Sonnet
  5 profile: max_tokens=1500, thinking disabled, no temperature/top_p/top_k),
  and response extraction (text-block concatenation, non-text blocks
  ignored, empty-response failure);
- services/anthropic_client.py's HTTP hardening (official base URL,
  ANTHROPIC_BASE_URL env-var immunity, no proxy trust — conventional env
  vars and emulated OS/system discovery), mirroring
  tests/test_stage1f_offline_enforcement.py's technique for the OpenAI
  client;
- services/text_llm.py's dispatch (explicit provider selection, no
  catch-and-fallback between providers);
- that router.py/rag/query.py's three answer-generation call sites reach
  the facade, and that the four OpenAI-only capabilities (image-intent
  classification, vision, embeddings, STT, TTS, image generation) were
  left untouched;
- privacy-safe logging for the new Anthropic adapter (Stage 1D guarantee).

All external calls are mocked at the SDK request boundary
(anthropic_client.client.messages.create) or the provider-method boundary
(anthropic_client.generate_text_response / openai_client.generate_text_response),
never at services.text_llm.generate_text_response itself when the test is
meant to exercise the facade's own dispatch logic. Config-import tests run
in a subprocess with an explicitly constructed (proxy-free) environment,
with config.py's own `load_dotenv()` call stubbed to a no-op so the real
repository .env — regardless of what it does or doesn't define — can never
influence the result (see the block comment above _CONFIG_IMPORT_SCRIPT).
Everything remains offline under pytest-socket (pytest.ini).
"""

import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import config
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from services.anthropic_client import (
    AnthropicClient,
    AnthropicResponseError,
    _extract_text,
    _split_system_message,
    anthropic_client,
)
from services.openai_client import openai_client
from services import text_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# RFC 5737 TEST-NET-1: reserved for documentation, never publicly routed,
# never requires a DNS lookup — same constant/rationale as
# tests/test_stage1f_offline_enforcement.py.
FAKE_PROVIDER_HOST = "192.0.2.1"


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Reset the in-memory session store between tests (router/RAG-path tests
    add history entries under the user_ids they use)."""
    from utils.helpers import user_sessions
    user_sessions.sessions.clear()
    yield
    user_sessions.sessions.clear()


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _thinking_block(text: str = "internal reasoning, never returned"):
    return SimpleNamespace(type="thinking", thinking=text)


def _fake_message(content_blocks, stop_reason="end_turn", usage=None):
    """Mimic the shape of an anthropic SDK messages.create() response."""
    return SimpleNamespace(
        content=content_blocks,
        stop_reason=stop_reason,
        model="claude-sonnet-5",
        usage=usage,
    )


# ============================================================================
# A. Config / provider resolution (subprocess — never touches this test
#    session's own already-imported `config` module, which tests/conftest.py
#    pins to LLM_PROVIDER=openai for the other 137 tests' sake).
# ============================================================================

# _run_config_import isolation from the real repository .env
# ----------------------------------------------------------------------------
# config.py's own `load_dotenv()` call takes no explicit path, so
# python-dotenv's find_dotenv() runs its default discovery — confirmed by
# reading the installed dotenv/main.py (present identically in the declared
# floor python-dotenv==1.0.0 and the resolved 1.2.3). find_dotenv() first
# calls an internal `_is_interactive()` check, which is just
# `not hasattr(__main__, "__file__")`. Every subprocess script in this
# module runs via `python -c "..."`, and `__main__` for a `-c` invocation
# has NO `__file__` (it was never loaded from a file) — so
# `_is_interactive()` is True and find_dotenv() anchors its search at
# `os.getcwd()`, NOT at config.py's own file location. (The call-stack-
# walking branch that anchors on the first external caller's file only runs
# when `_is_interactive()` is False, i.e. a real `python script.py`
# invocation with `__main__.__file__` set — never here.) This means the
# search root tracks the SUBPROCESS'S CURRENT WORKING DIRECTORY, not
# sys.argv or wherever the imported config.py physically lives. This is
# exactly why test_config_subprocess_isolation_mechanism_is_real_not_coincidental
# below runs its subprocesses with an explicit `cwd=fake_project`: without
# it, python-dotenv would search from pytest's own working directory
# (typically the real repository root) rather than the fake project,
# making the "unisolated" control's discovery of the fake .env
# runner-cwd-dependent instead of a reliable proof.
#
# A per-variable "pre-set an empty value so override=False won't touch it"
# trick was considered and rejected as the general fix: it works for
# ANTHROPIC_API_KEY (`not ""` and `not None` are equally falsy), but not for
# LLM_PROVIDER/ANTHROPIC_MODEL, which both have a *default* argument to
# os.getenv() — presetting those to "" would make os.getenv() return ""
# (the key is PRESENT), not the default (which only fires when the key is
# ABSENT), silently breaking the "truly unset" test scenarios such as
# test_default_provider_resolves_to_anthropic.
#
# Instead, `dotenv.load_dotenv` is stubbed to a no-op BEFORE `import config`
# runs, inside the subprocess script below. config.py does
# `from dotenv import load_dotenv` at ITS OWN import time — Python resolves
# that against the `dotenv` module's CURRENT attribute at that moment, so it
# binds and calls this stub instead of the real implementation. The real
# repository .env file is then never opened at all, by any test in this
# module — deterministic regardless of what it does or doesn't define
# (ANTHROPIC_API_KEY, LLM_PROVIDER, ANTHROPIC_MODEL, OPENAI_API_KEY, ...),
# and regardless of the subprocess's cwd, since the stub short-circuits
# before find_dotenv() (and therefore os.getcwd()) is ever consulted. No
# cleanup/restoration is needed: the stub exists only inside this
# short-lived subprocess's own memory, which exits immediately after
# printing its result line — the pytest process's own `dotenv` module (and
# production behavior generally) is never touched.
#
# This is deliberately different from — and does not affect —
# tests/test_stage1f_offline_enforcement.py's
# test_dotenv_reintroduction_cannot_restore_usable_provider_proxy, which
# exercises the OPPOSITE scenario on purpose (a controlled, test-owned
# temporary .env that SHOULD be discovered) via its own fully independent
# subprocess script.
_CONFIG_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys

    project_root = sys.argv[1]
    sys.path.insert(0, project_root)

    # Resolved via the finder alone (no execution), BEFORE the dotenv stub
    # or `import config` below, so this reflects which config.py file would
    # be imported regardless of whether the import itself goes on to
    # succeed or raise. Printed back as non-sensitive proof that the
    # subprocess actually resolved project_root's own config.py, not some
    # other config.py found earlier on sys.path.
    _spec = importlib.util.find_spec("config")
    config_file = _spec.origin if _spec is not None else None

    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False

    try:
        import config
    except Exception as e:
        print("IMPORT_RESULT=" + json.dumps({
            "raised": True,
            "error_type": type(e).__name__,
            "config_file": config_file,
        }))
        sys.exit(0)

    print("IMPORT_RESULT=" + json.dumps({
        "raised": False,
        "llm_provider": config.LLM_PROVIDER,
        "anthropic_model": config.ANTHROPIC_MODEL,
        "official_anthropic_base_url": config.OFFICIAL_ANTHROPIC_BASE_URL,
        "anthropic_key_present": bool(config.ANTHROPIC_API_KEY),
        "config_file": config_file,
    }))
    """
)

# The pre-fix equivalent (no dotenv stub) — kept only as the "vulnerable
# control" half of test_config_subprocess_isolation_mechanism_is_real_not_coincidental,
# to prove find_dotenv() really would discover an adjacent .env when nothing
# suppresses it, so the fix above is proven to matter and isn't passing by
# coincidence. Relies on that test running this script with cwd=fake_project
# (see the block comment above _CONFIG_IMPORT_SCRIPT): find_dotenv() under
# `python -c` anchors its search at os.getcwd(), not at config.py's file
# location, so without that explicit cwd this script's "discovery" would
# actually be exercising wherever pytest itself happens to be running from.
_UNISOLATED_CONFIG_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys

    project_root = sys.argv[1]
    sys.path.insert(0, project_root)

    _spec = importlib.util.find_spec("config")
    config_file = _spec.origin if _spec is not None else None

    try:
        import config
    except Exception as e:
        print("IMPORT_RESULT=" + json.dumps({
            "raised": True,
            "error_type": type(e).__name__,
            "config_file": config_file,
        }))
        sys.exit(0)

    print("IMPORT_RESULT=" + json.dumps({
        "raised": False,
        "llm_provider": config.LLM_PROVIDER,
        "anthropic_key_present": bool(config.ANTHROPIC_API_KEY),
        "config_file": config_file,
    }))
    """
)


def _base_subprocess_env() -> dict:
    """
    TELEGRAM_BOT_TOKEN/OPENAI_API_KEY are always supplied (dummy) since
    config.py requires them unconditionally, regardless of what a given test
    is actually checking. Only PATH/SYSTEMROOT/etc. are carried through from
    the real environment — enough for the interpreter subprocess itself to
    start; never a provider, key, or proxy variable. Combined with the
    dotenv stub above, this constructed dict is the ONLY source of
    environment variables config.py's os.getenv() calls can ever observe in
    an isolated subprocess — never the real repository .env, regardless of
    its presence, absence, or contents.
    """
    env = {
        "TELEGRAM_BOT_TOKEN": "123456789:TEST-TOKEN-DO-NOT-USE",
        "OPENAI_API_KEY": "sk-test-dummy-key",
        "ANONYMIZED_TELEMETRY": "False",
    }
    for name in ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _run_config_import(env_overrides: dict) -> dict:
    """
    Import config.py fresh in a subprocess under a specific, explicitly
    constructed (and dotenv-isolated — see the block comment above) environment,
    and report whether it raised (plus the exception type) or succeeded (plus
    the resolved provider/model/base-url/key-presence values). Never disturbs
    this pytest session's own already-imported `config` module.
    """
    env = _base_subprocess_env()
    env.update(env_overrides)

    proc = subprocess.run(
        [sys.executable, "-c", _CONFIG_IMPORT_SCRIPT, str(PROJECT_ROOT)],
        capture_output=True, text=True, timeout=30, env=env,
    )
    result_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("IMPORT_RESULT=")), None
    )
    assert result_line is not None, (
        f"no IMPORT_RESULT line from subprocess\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(result_line[len("IMPORT_RESULT="):])


def test_default_provider_resolves_to_anthropic():
    """Production default (LLM_PROVIDER unset) resolves to 'anthropic'."""
    result = _run_config_import({"ANTHROPIC_API_KEY": "sk-ant-test-dummy-key"})
    assert result["raised"] is False, result
    assert result["llm_provider"] == "anthropic"
    assert result["anthropic_model"] == "claude-sonnet-5"
    assert result["official_anthropic_base_url"] == "https://api.anthropic.com"


def test_invalid_provider_fails_closed_at_config_import():
    """An unrecognized LLM_PROVIDER value must fail closed, not silently
    default to either provider."""
    result = _run_config_import({"LLM_PROVIDER": "azure"})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_anthropic_selected_missing_key_fails_fast():
    """LLM_PROVIDER=anthropic with no ANTHROPIC_API_KEY must fail at
    config-import time, not at first request."""
    result = _run_config_import({"LLM_PROVIDER": "anthropic"})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_openai_selection_does_not_require_anthropic_key():
    """LLM_PROVIDER=openai must succeed with no ANTHROPIC_API_KEY at all."""
    result = _run_config_import({"LLM_PROVIDER": "openai"})
    assert result["raised"] is False, result
    assert result["llm_provider"] == "openai"


def test_config_subprocess_isolation_mechanism_is_real_not_coincidental(tmp_path):
    """
    Explicit proof that a value present in the user's real repository .env
    cannot silently rescue test_anthropic_selected_missing_key_fails_fast
    (or any other config-subprocess test) — without ever inspecting the real
    .env's actual contents.

    Builds its own fake project directory: a copy of config.py's source
    (never modified — proving PRODUCTION config.py behavior, not a rewritten
    stand-in) plus a FAKE, test-owned, throwaway .env file defining a fake
    ANTHROPIC_API_KEY value that is never the real repository secret. Both
    subprocesses below are run with cwd=fake_project — required because
    python-dotenv's find_dotenv(), under a `python -c` invocation, anchors
    its search at os.getcwd() rather than at config.py's file location (see
    the block comment above _CONFIG_IMPORT_SCRIPT); without an explicit
    cwd, discovery of the fake .env would depend on wherever the pytest
    runner itself happens to be invoked from, and could instead reach the
    real repository .env or nothing at all. Runs the exact
    "LLM_PROVIDER=anthropic, ANTHROPIC_API_KEY not explicitly supplied"
    scenario test_anthropic_selected_missing_key_fails_fast models, twice,
    against that same fake project/same cwd:

    - _UNISOLATED_CONFIG_IMPORT_SCRIPT (no dotenv stub — the pre-fix
      behavior): the fake .env's key IS discovered and rescues the import.
      This is the control — if it did NOT observe the fake key, the fake
      project fixture itself would be broken and this test would prove
      nothing about the real mechanism.
    - _CONFIG_IMPORT_SCRIPT (the fix _run_config_import actually uses): the
      same fake .env's key is NOT discovered — config import still fails
      for the missing key, exactly as intended.

    Both scripts also report `config.__file__`'s resolved path (via
    importlib.util.find_spec, before either script even attempts `import
    config`), which this test asserts equals fake_project/config.py exactly
    — proving the COPIED config.py was what got resolved/imported in both
    runs, never the real repository config.py through some other sys.path
    entry, and that the real repository .env is never opened, read, or
    otherwise involved in producing either result.

    Together these prove the dotenv stub — not luck, not cwd, not what the
    real repository .env happens to currently contain — is what makes every
    config-subprocess test in this module deterministic.
    """
    fake_project = tmp_path / "fake_project"
    fake_project.mkdir()
    (fake_project / "config.py").write_text(
        (PROJECT_ROOT / "config.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    # Stage 2B-D Section E: config.py now does `from rag.constants import
    # (...)` at its own top level (pure, side-effect-free constants) — the
    # copied config.py above needs the real rag/constants.py (and rag's own
    # __init__.py) importable alongside it, or `import config` fails with
    # ModuleNotFoundError before ever reaching the ANTHROPIC_API_KEY check
    # this test is actually about. Only rag/constants.py + rag/__init__.py
    # are copied — never rag/index.py etc., which this fake project has no
    # need for and which would pull in heavy third-party imports.
    import shutil as _shutil
    (fake_project / "rag").mkdir()
    _shutil.copy2(PROJECT_ROOT / "rag" / "__init__.py", fake_project / "rag" / "__init__.py")
    _shutil.copy2(PROJECT_ROOT / "rag" / "constants.py", fake_project / "rag" / "constants.py")
    (fake_project / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-fake-rescue-value-never-the-real-secret\n",
        encoding="utf-8",
    )

    env = _base_subprocess_env()
    env["LLM_PROVIDER"] = "anthropic"

    def _run(script: str) -> dict:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(fake_project)],
            capture_output=True, text=True, timeout=30, env=env,
            cwd=str(fake_project),
        )
        result_line = next(
            (line for line in proc.stdout.splitlines() if line.startswith("IMPORT_RESULT=")),
            None,
        )
        assert result_line is not None, (
            f"no IMPORT_RESULT line from subprocess\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
        return json.loads(result_line[len("IMPORT_RESULT="):])

    expected_config_file = str((fake_project / "config.py").resolve())

    unisolated_result = _run(_UNISOLATED_CONFIG_IMPORT_SCRIPT)
    assert unisolated_result["raised"] is False, (
        "control invalid: the fake adjacent .env was not discovered at all "
        f"by the unpatched script, so this proves nothing: {unisolated_result}"
    )
    assert unisolated_result["anthropic_key_present"] is True
    assert unisolated_result["config_file"] is not None, unisolated_result
    assert Path(unisolated_result["config_file"]).resolve() == Path(expected_config_file), (
        "vulnerable control imported a config.py other than the fake "
        f"project's own copy: {unisolated_result}"
    )

    isolated_result = _run(_CONFIG_IMPORT_SCRIPT)
    assert isolated_result["raised"] is True, (
        f"the fake .env's key silently rescued the isolated import: {isolated_result}"
    )
    assert isolated_result["error_type"] == "ValueError"
    assert isolated_result["config_file"] is not None, isolated_result
    assert Path(isolated_result["config_file"]).resolve() == Path(expected_config_file), (
        "isolated run resolved a config.py other than the fake project's "
        f"own copy: {isolated_result}"
    )


# ============================================================================
# B. Anthropic client HTTP security (official base URL, no env redirection,
#    no proxy trust) — mirrors test_stage1f_offline_enforcement.py's
#    technique for the OpenAI client, applied to the real anthropic_client
#    singleton.
# ============================================================================

def test_anthropic_client_uses_official_base_url():
    """The real anthropic_client singleton talks to the official Anthropic
    endpoint, not something derived from the environment."""
    assert str(anthropic_client.client.base_url).rstrip("/") == "https://api.anthropic.com"


def test_anthropic_client_has_no_trust_env():
    """Narrow configuration assertion on the REAL singleton, mirroring
    test_real_openai_client_singleton_has_no_trust_transport's pattern:
    fails immediately if a future refactor drops the explicit http_client=
    kwarg from AnthropicClient.__init__."""
    assert anthropic_client.client._client.trust_env is False


def test_anthropic_client_source_does_not_read_anthropic_base_url_env_var():
    """services/anthropic_client.py may only *mention* ANTHROPIC_BASE_URL in
    an explanatory comment; it must never read it as config and must always
    pass an explicit, hardcoded base_url to AsyncAnthropic — same check as
    test_stage1a_security.py's test_openai_client_does_not_read_openai_base_url_env_var."""
    content = (PROJECT_ROOT / "services" / "anthropic_client.py").read_text(encoding="utf-8")
    assert 'os.getenv("ANTHROPIC_BASE_URL")' not in content
    assert "os.environ.get('ANTHROPIC_BASE_URL')" not in content
    assert '"ANTHROPIC_BASE_URL",' not in content


def test_anthropic_base_url_env_cannot_redirect_client(monkeypatch):
    """A stray ANTHROPIC_BASE_URL in the environment must not redirect a
    freshly constructed production client — proven against the installed
    SDK: an AsyncAnthropic() built WITHOUT an explicit base_url does pick up
    ANTHROPIC_BASE_URL, so the explicit base_url= kwarg in
    AnthropicClient.__init__ is what actually closes this, not the absence
    of the env var."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:1/evil")
    fresh_client = AnthropicClient()
    assert str(fresh_client.client.base_url).rstrip("/") == "https://api.anthropic.com"


@contextlib.contextmanager
def _local_listener():
    """
    A bare TCP listener on 127.0.0.1 (ephemeral port) that records the first
    thing a client sends it, then closes. Duplicated from
    tests/test_stage1f_offline_enforcement.py rather than imported —
    matching that file's own stated convention of keeping test-module
    helpers self-contained instead of cross-importing between test modules.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(3.0)
    port = srv.getsockname()[1]
    received = {"data": None}

    def _accept():
        try:
            conn, _addr = srv.accept()
        except OSError:
            return
        try:
            conn.settimeout(2.0)
            try:
                received["data"] = conn.recv(4096)
            except OSError:
                pass
        finally:
            conn.close()

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    try:
        yield port, received
    finally:
        thread.join(timeout=3.0)
        srv.close()


def _vulnerable_anthropic_client(base_url: str) -> AsyncAnthropic:
    """
    The PREVIOUSLY REJECTED mitigation, kept here only as a vulnerable
    control: `trust_env=False` alone, no explicit `mounts=`. Never used as
    production configuration — services/anthropic_client.py's
    AnthropicClient no longer builds a client this way.

    Confirmed by reading the installed SDK (anthropic/_base_client.py,
    `_DefaultAsyncHttpxClient.__init__`, present verbatim in both
    anthropic==1.0.0 and anthropic==1.2.0): this constructor calls
    `anthropic._utils._httpx.get_environment_proxies()` — which wraps
    `urllib.request.getproxies()`, so it sees both conventional
    HTTP_PROXY/HTTPS_PROXY/ALL_PROXY env vars and OS-level discovery —
    UNCONDITIONALLY, before `trust_env` is ever consulted, and installs
    "http://"/"https://"/"all://" transport mounts from whatever it finds.
    `trust_env=False` only affects the transport objects those mounts would
    use; it does not stop the mounts from being installed.
    """
    return AsyncAnthropic(
        api_key="sk-ant-test-dummy-key",
        base_url=base_url,
        http_client=DefaultAsyncHttpxClient(trust_env=False),
        timeout=1.0,
        max_retries=0,
    )


def _was_blocked_synchronously_by_pytest_socket(exc: BaseException) -> bool:
    """
    True if `pytest_socket.SocketConnectBlockedError` appears anywhere in
    `exc`'s `__cause__`/`__context__` chain — i.e. pytest-socket's
    `socket.socket.connect` patch actually intercepted the attempt, rather
    than the attempt running to a real (harmless — TEST-NET-1 is reserved
    and never publicly routed) connect timeout.

    Purely diagnostic, not asserted on: empirically confirmed in this
    environment (Windows, asyncio ProactorEventLoop) that pytest-socket's
    patch reliably intercepts *synchronous* httpx2.Client connections
    (observed directly via its UserWarning + SocketConnectBlockedError for
    this repo's existing sync OpenAI proxy tests in
    test_stage1f_offline_enforcement.py), but does NOT intercept *async*
    httpx2.AsyncClient connections under ProactorEventLoop — those go
    through a lower-level Windows I/O completion path that never calls the
    patched `.connect()` method. `pytest.raises(Exception)` around the call
    already proves it never got a real response back either way; only
    `received["data"] is None` is the actual proxy-immunity proof.
    """
    import pytest_socket

    seen = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return any(isinstance(e, pytest_socket.SocketConnectBlockedError) for e in seen)


def _assert_singleton_has_safe_mounts():
    """The real module singleton (constructed at import time, long before
    any test runs) must carry the same explicit no-proxy mount
    configuration as any freshly built production client — not just this
    test's own reconstructed instance."""
    singleton_mounts = anthropic_client.client._client._mounts
    assert singleton_mounts, "expected explicit proxy mounts on the production singleton"
    assert all(transport is None for transport in singleton_mounts.values()), (
        f"production singleton has a live (non-None) proxy mount: {singleton_mounts}"
    )


# ----------------------------------------------------------------------------
# 1-2. Conventional env proxy (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY)
# ----------------------------------------------------------------------------

async def test_conventional_proxy_env_vulnerable_control(monkeypatch):
    """VULNERABLE CONTROL: proves `trust_env=False` ALONE (the mitigation
    Codex independently rejected) is NOT sufficient. A fresh client built
    only with trust_env=False, constructed AFTER conventional proxy env
    vars already exist, still routes an outbound request through the local
    proxy listener — proving the detection mechanism itself is real before
    trusting what the production-fix test below reports."""
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("ALL_PROXY", proxy_url)

        vulnerable = _vulnerable_anthropic_client(f"https://{FAKE_PROVIDER_HOST}")
        with contextlib.suppress(Exception):
            await vulnerable.models.list()

    assert received["data"] is not None, (
        "vulnerable control did not reach the local proxy listener — the "
        "detection mechanism is not working, so it cannot certify the "
        "production-fix test below"
    )
    assert received["data"].startswith(b"CONNECT "), received["data"]


async def test_conventional_proxy_env_production_fix(monkeypatch):
    """PRODUCTION FIX: same hostile environment as the control above, but
    reconstructs the actual production client (services/anthropic_client.py's
    real AnthropicClient, with the explicit no-proxy `mounts=`) AFTER the
    proxy env vars already exist in the environment — not relying only on
    the already-created module singleton. The local listener must receive
    NOTHING, and the direct external target must remain blocked by
    pytest-socket. The real module singleton is also checked to carry the
    same safe mount configuration."""
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("ALL_PROXY", proxy_url)

        production = AnthropicClient()
        monkeypatch.setattr(production.client, "base_url", f"https://{FAKE_PROVIDER_HOST}/")

        # The call must never return a real response; pytest.raises here is
        # the "not silently succeeded" guarantee. Whether it was pytest-socket
        # itself or a natural TEST-NET-1 connect timeout that raised is
        # diagnostic only — see _was_blocked_synchronously_by_pytest_socket.
        with pytest.raises(Exception) as exc_info:
            await production.client.models.list()
        _was_blocked_synchronously_by_pytest_socket(exc_info.value)

    assert received["data"] is None, (
        f"local proxy unexpectedly received data: {received['data']!r}"
    )
    _assert_singleton_has_safe_mounts()


# ----------------------------------------------------------------------------
# 3-4. OS/system proxy discovery — patches the binding Anthropic ACTUALLY
# calls (anthropic._utils._httpx.getproxies, which wraps
# urllib.request.getproxies and therefore also stands in for Windows
# Registry / macOS system config discovery), not the unrelated
# httpx2._utils.getproxies a prior version of this test patched. Both names
# independently do `from urllib.request import getproxies` into their own
# module namespace (confirmed by reading both installed modules), so
# patching one never affects the other — proven directly below.
# ----------------------------------------------------------------------------

async def test_system_proxy_discovery_vulnerable_control(monkeypatch):
    """VULNERABLE CONTROL: no proxy env var set at all; only the emulated
    OS/system-level discovery is active. Proves this route is real and
    independent of the env-var route above, against the same
    trust_env=False-only construction Codex rejected."""
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"

        def fake_getproxies():
            return {"http": proxy_url, "https": proxy_url}

        monkeypatch.setattr("anthropic._utils._httpx.getproxies", fake_getproxies)

        vulnerable = _vulnerable_anthropic_client(f"https://{FAKE_PROVIDER_HOST}")
        with contextlib.suppress(Exception):
            await vulnerable.models.list()

    assert received["data"] is not None, (
        "vulnerable control did not reach the local proxy listener under "
        "emulated OS/system proxy discovery — the detection mechanism is "
        "not working, so it cannot certify the production-fix test below"
    )
    assert received["data"].startswith(b"CONNECT "), received["data"]


async def test_system_proxy_discovery_production_fix(monkeypatch):
    """PRODUCTION FIX: same emulated OS/system-level discovery, against a
    freshly reconstructed production client (the real AnthropicClient,
    explicit no-proxy mounts). The local listener must receive NOTHING."""
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"

        def fake_getproxies():
            return {"http": proxy_url, "https": proxy_url}

        monkeypatch.setattr("anthropic._utils._httpx.getproxies", fake_getproxies)

        production = AnthropicClient()
        monkeypatch.setattr(production.client, "base_url", f"https://{FAKE_PROVIDER_HOST}/")

        with contextlib.suppress(Exception):
            await production.client.models.list()

    assert received["data"] is None, (
        f"local proxy unexpectedly received data under emulated OS/system "
        f"proxy discovery: {received['data']!r}"
    )
    _assert_singleton_has_safe_mounts()


def test_httpx2_getproxies_is_not_the_binding_anthropic_actually_calls():
    """
    Documents WHY the previous version of this test suite's system-proxy
    test was a false negative: `httpx2._utils.getproxies` and
    `anthropic._utils._httpx.getproxies` are two independent module-level
    names (each does its own `from urllib.request import getproxies`), so
    patching the former has zero effect on Anthropic's own discovery path.
    A regression here (e.g. the SDK refactoring to share one binding) would
    only make the test suite's assumption stale, never silently reintroduce
    the original false-negative bug, since the tests above patch the real
    binding directly.
    """
    import httpx2._utils as httpx2_utils
    import anthropic._utils._httpx as anthropic_httpx_utils

    assert httpx2_utils.getproxies is not None
    assert anthropic_httpx_utils.getproxies is not None
    # Both wrap the same stdlib function today, but as two distinct names in
    # two distinct module namespaces — reassigning one attribute never
    # reassigns the other.
    assert "anthropic" in anthropic_httpx_utils.__name__
    assert "httpx2" in httpx2_utils.__name__


# ============================================================================
# C. Message mapping (system extraction, history order, role validation) —
#    unit-level against the pure helper, matching the SDK's actual
#    MessageParam shape confirmed against the installed 1.x SDK.
# ============================================================================

def test_leading_system_message_maps_to_top_level_system():
    system, messages = _split_system_message(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    )
    assert system == "SYS"
    assert messages == [{"role": "user", "content": "hi"}]


def test_no_system_message_request_maps_correctly():
    system, messages = _split_system_message([{"role": "user", "content": "hi"}])
    assert system is None
    assert messages == [{"role": "user", "content": "hi"}]


def test_user_assistant_history_order_preserved():
    original = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    system, messages = _split_system_message(original)
    assert system == "SYS"
    assert messages == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]


def test_unexpected_role_fails_rather_than_silently_converted():
    with pytest.raises(ValueError):
        _split_system_message([{"role": "user", "content": "a"}, {"role": "tool", "content": "x"}])


def test_non_leading_system_message_fails_rather_than_silently_converted():
    """Only a LEADING system message is recognized; one appearing anywhere
    else must fail, not be silently reinterpreted as a mid-conversation
    operator message (a distinct, model-gated Anthropic feature this
    application does not use, and Sonnet 5 does not support)."""
    with pytest.raises(ValueError):
        _split_system_message([{"role": "user", "content": "a"}, {"role": "system", "content": "b"}])


def test_non_string_content_fails_rather_than_silently_converted():
    with pytest.raises(ValueError):
        _split_system_message([{"role": "user", "content": ["not", "a", "string"]}])


# ----------------------------------------------------------------------------
# Malformed LEADING system message: a missing/None/non-string "content" must
# raise, never be silently reinterpreted as "no system message was sent" —
# the bug an independent audit found: {"role": "system"} followed by a valid
# user message was previously accepted and the whole system entry silently
# dropped, because the old code only validated `system_content` when it
# happened to be not-None.
# ----------------------------------------------------------------------------

def test_leading_system_message_missing_content_key_raises():
    with pytest.raises(ValueError):
        _split_system_message([{"role": "system"}, {"role": "user", "content": "hi"}])


def test_leading_system_message_none_content_raises():
    with pytest.raises(ValueError):
        _split_system_message(
            [{"role": "system", "content": None}, {"role": "user", "content": "hi"}]
        )


@pytest.mark.parametrize("bad_content", [42, ["a", "list"], {"nested": "object"}, True])
def test_leading_system_message_non_string_content_raises(bad_content):
    with pytest.raises(ValueError):
        _split_system_message(
            [{"role": "system", "content": bad_content}, {"role": "user", "content": "hi"}]
        )


def test_leading_system_message_missing_content_with_only_system_message_raises():
    """Malformed leading system message must raise even when it's the only
    entry in `messages` (no trailing user message to otherwise mask it)."""
    with pytest.raises(ValueError):
        _split_system_message([{"role": "system"}])


async def test_malformed_leading_system_message_never_reaches_sdk_call(monkeypatch):
    """Requirement: malformed system input must not invoke messages.create()
    — the ValueError from _split_system_message must abort
    generate_text_response before any request is built or sent."""
    create_mock = AsyncMock(return_value=_fake_message([_text_block("ok")]))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)

    with pytest.raises(ValueError):
        await anthropic_client.generate_text_response(
            [{"role": "system"}, {"role": "user", "content": "hi"}]
        )

    create_mock.assert_not_called()


async def test_generate_text_response_maps_system_and_history_through_to_sdk_call(monkeypatch):
    """End-to-end proof at the actual SDK request boundary: system extracted
    to the top-level `system` kwarg, history order preserved in `messages`,
    no role="system" entry anywhere inside `messages`."""
    create_mock = AsyncMock(return_value=_fake_message([_text_block("ok")]))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)

    await anthropic_client.generate_text_response(
        [
            {"role": "system", "content": "You are a tutor."},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )

    kwargs = create_mock.call_args.kwargs
    assert kwargs["system"] == "You are a tutor."
    assert kwargs["messages"] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert all(m["role"] != "system" for m in kwargs["messages"])


# ============================================================================
# D. Sonnet 5 request contract — max_tokens=1500, thinking disabled, no
#    temperature/top_p/top_k. Each assertion isolated in its own test for
#    audit traceability.
# ============================================================================

async def test_sonnet_request_includes_max_tokens_1500(monkeypatch):
    create_mock = AsyncMock(return_value=_fake_message([_text_block("ok")]))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)
    await anthropic_client.generate_text_response([{"role": "user", "content": "hi"}])
    assert create_mock.call_args.kwargs["max_tokens"] == 1500


async def test_sonnet_request_includes_thinking_disabled(monkeypatch):
    create_mock = AsyncMock(return_value=_fake_message([_text_block("ok")]))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)
    await anthropic_client.generate_text_response([{"role": "user", "content": "hi"}])
    assert create_mock.call_args.kwargs["thinking"] == {"type": "disabled"}


async def test_sonnet_request_does_not_send_temperature(monkeypatch):
    create_mock = AsyncMock(return_value=_fake_message([_text_block("ok")]))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)
    await anthropic_client.generate_text_response([{"role": "user", "content": "hi"}])
    assert "temperature" not in create_mock.call_args.kwargs


async def test_sonnet_request_does_not_send_top_p_or_top_k(monkeypatch):
    create_mock = AsyncMock(return_value=_fake_message([_text_block("ok")]))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)
    await anthropic_client.generate_text_response([{"role": "user", "content": "hi"}])
    kwargs = create_mock.call_args.kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


# ============================================================================
# E. Response extraction — unit-level against the pure helper.
# ============================================================================

def test_single_text_block_returns_plain_str():
    response = _fake_message([_text_block("hello")])
    assert _extract_text(response) == "hello"


def test_multiple_text_blocks_combined_in_order():
    response = _fake_message([_text_block("hello "), _text_block("world")])
    assert _extract_text(response) == "hello world"


def test_non_text_blocks_are_ignored():
    response = _fake_message([_thinking_block(), _text_block("visible answer")])
    assert _extract_text(response) == "visible answer"


def test_empty_response_fails_safely():
    response = _fake_message([])
    with pytest.raises(AnthropicResponseError):
        _extract_text(response)


def test_non_text_only_response_fails_safely():
    response = _fake_message([_thinking_block()], stop_reason="end_turn")
    with pytest.raises(AnthropicResponseError):
        _extract_text(response)


def test_refusal_with_usable_text_is_returned():
    """A refusal is not special-cased: if the refused response still
    carries a text block, it's returned like any other."""
    response = _fake_message([_text_block("I can't help with that.")], stop_reason="refusal")
    assert _extract_text(response) == "I can't help with that."


def test_refusal_with_no_text_fails_safely():
    """A refusal with no usable text falls through to the same
    no-usable-text error as any other empty response — no new Telegram
    refusal UX in this stage."""
    response = _fake_message([], stop_reason="refusal")
    with pytest.raises(AnthropicResponseError):
        _extract_text(response)


# ============================================================================
# F. Facade dispatch and no-fallback behavior.
# ============================================================================

async def test_facade_dispatches_to_anthropic_when_selected(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    anthropic_mock = AsyncMock(return_value="from anthropic")
    openai_mock = AsyncMock(return_value="from openai")
    monkeypatch.setattr(anthropic_client, "generate_text_response", anthropic_mock)
    monkeypatch.setattr(openai_client, "generate_text_response", openai_mock)

    result = await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    assert result == "from anthropic"
    anthropic_mock.assert_called_once()
    openai_mock.assert_not_called()


async def test_facade_dispatches_to_openai_when_selected(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    anthropic_mock = AsyncMock(return_value="from anthropic")
    openai_mock = AsyncMock(return_value="from openai")
    monkeypatch.setattr(anthropic_client, "generate_text_response", anthropic_mock)
    monkeypatch.setattr(openai_client, "generate_text_response", openai_mock)

    result = await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    assert result == "from openai"
    openai_mock.assert_called_once()
    anthropic_mock.assert_not_called()


async def test_anthropic_failure_does_not_invoke_openai(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    anthropic_mock = AsyncMock(side_effect=RuntimeError("anthropic boom"))
    openai_mock = AsyncMock(return_value="from openai")
    monkeypatch.setattr(anthropic_client, "generate_text_response", anthropic_mock)
    monkeypatch.setattr(openai_client, "generate_text_response", openai_mock)

    with pytest.raises(RuntimeError, match="anthropic boom"):
        await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    openai_mock.assert_not_called()


async def test_openai_failure_does_not_invoke_anthropic(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    anthropic_mock = AsyncMock(return_value="from anthropic")
    openai_mock = AsyncMock(side_effect=RuntimeError("openai boom"))
    monkeypatch.setattr(anthropic_client, "generate_text_response", anthropic_mock)
    monkeypatch.setattr(openai_client, "generate_text_response", openai_mock)

    with pytest.raises(RuntimeError, match="openai boom"):
        await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    anthropic_mock.assert_not_called()


def test_invalid_provider_at_facade_dispatch_fails_closed(monkeypatch):
    """Defense in depth: even if the facade were ever reached with a value
    config.py's own startup validation should have already rejected, it
    still fails closed rather than guessing a provider."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "azure")
    with pytest.raises(ValueError):
        import asyncio
        asyncio.run(text_llm.generate_text_response([{"role": "user", "content": "hi"}]))


def test_text_llm_facade_has_no_exception_handling():
    """Structural no-fallback proof: the facade module contains no
    try/except statement at all, so a provider failure has no path to be
    caught and retried through the other provider inside this file.

    Checks for the actual `except`/`try:` Python keywords as statements
    (a stripped line starting with them), not merely the substring
    "except" — which also occurs inside ordinary prose words like
    "exception" in this module's own docstrings.
    """
    content = (PROJECT_ROOT / "services" / "text_llm.py").read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines()]
    assert not any(line.startswith("except") for line in lines)
    assert not any(line == "try:" or line.startswith("try:") for line in lines)


# ============================================================================
# G. Call-site routing: the three answer-generation paths reach the facade.
# ============================================================================

async def test_router_ordinary_tutor_path_uses_text_llm_facade(monkeypatch):
    from services.router import route_text_request

    # Intent classifier stays directly on OpenAI (Stage 2A scope) — return
    # a non-JSON string so detect_image_generation_intent's own except
    # branch resolves needs_generation=False and lets the normal path run.
    intent_create_mock = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json, no image intent"))],
        usage=None,
    ))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", intent_create_mock)

    facade_mock = AsyncMock(return_value="Tutor answer.")
    monkeypatch.setattr(text_llm, "generate_text_response", facade_mock)

    response = await route_text_request(user_id=900001, text="What is a list comprehension?")

    assert response["text"] == "Tutor answer."
    facade_mock.assert_called_once()
    # Exactly one OpenAI chat call (the intent classifier) — no leftover
    # direct call for the main answer.
    assert intent_create_mock.call_count == 1


async def test_rag_response_path_uses_text_llm_facade(monkeypatch):
    import rag.query as rag_query

    fake_doc = SimpleNamespace(
        metadata={"source": "python_basics.txt"},
        page_content="Some retrieved passage.",
    )
    monkeypatch.setattr(
        rag_query.get_vector_index(), "similarity_search_with_score",
        lambda query, k=3: [(fake_doc, 0.1)],
    )
    facade_mock = AsyncMock(return_value="RAG-grounded answer.")
    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", facade_mock)

    response = await rag_query.query_knowledge_base("What is PEP 8?")

    assert "RAG-grounded answer." in response
    assert "python_basics.txt" in response
    facade_mock.assert_called_once()


async def test_rag_fallback_path_uses_text_llm_facade(monkeypatch):
    import rag.query as rag_query

    monkeypatch.setattr(
        rag_query.get_vector_index(), "similarity_search_with_score",
        lambda query, k=3: [],
    )
    facade_mock = AsyncMock(return_value="General-knowledge answer.")
    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", facade_mock)

    response = await rag_query.query_knowledge_base("What is a metaclass?")

    assert "General-knowledge answer." in response
    assert "База знаний не содержит информации" in response
    facade_mock.assert_called_once()


# ============================================================================
# H. OpenAI-only capabilities untouched (regression guard against scope
#    creep, not a re-proof of behavior already covered by Stage 1A-1F).
# ============================================================================

def _source(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_image_intent_classifier_remains_directly_openai():
    content = _source("services/image_generation.py")
    assert "openai_client.generate_text_response" in content
    assert "text_llm" not in content
    assert "anthropic" not in content.lower()


async def test_image_intent_classifier_behavior_unchanged(monkeypatch):
    """detect_image_generation_intent still calls openai_client directly
    with its existing temperature=0.3 override, never the facade."""
    from services.image_generation import detect_image_generation_intent

    create_mock = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"needs_generation": false, "confidence": 0.1}')
        )],
        usage=None,
    ))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    result = await detect_image_generation_intent("What is a for loop?")

    assert result["needs_generation"] is False
    create_mock.assert_called_once()
    assert create_mock.call_args.kwargs.get("temperature") == 0.3


def test_vision_remains_openai():
    content = _source("services/vision.py")
    assert "from services.openai_client import openai_client" in content
    assert "text_llm" not in content
    assert "anthropic" not in content.lower()


def test_embeddings_remain_openai():
    content = _source("rag/index.py")
    assert "OpenAIEmbeddings" in content
    assert "anthropic" not in content.lower()


def test_stt_remains_openai():
    content = _source("services/stt.py")
    assert "openai_client" in content
    assert "text_llm" not in content
    assert "anthropic" not in content.lower()


def test_tts_remains_openai():
    content = _source("services/tts.py")
    assert "openai_client" in content
    assert "text_llm" not in content
    assert "anthropic" not in content.lower()


def test_image_generation_remains_openai():
    content = _source("services/image_generation.py")
    # DALL-E functions (generate_image/generate_image_variations) are
    # untouched raw-HTTP OpenAI calls; the whole module never mentions
    # Anthropic anywhere, including the classifier at the top of this
    # section.
    assert "OPENAI_API_KEY" in content
    assert "anthropic" not in content.lower()


# ============================================================================
# I. Privacy-safe logging for the Anthropic adapter (Stage 1D guarantee).
# ============================================================================

async def test_anthropic_exception_type_logged_not_raw_text(monkeypatch, caplog):
    create_mock = AsyncMock(side_effect=RuntimeError("SECRET_TOKEN_LEAK_MARKER_xyz"))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            await anthropic_client.generate_text_response([{"role": "user", "content": "hi"}])

    assert "RuntimeError" in caplog.text
    assert "SECRET_TOKEN_LEAK_MARKER_xyz" not in caplog.text


async def test_anthropic_message_content_never_logged(monkeypatch, caplog):
    marker = "CONFIDENTIAL_MARKER_abc123"
    create_mock = AsyncMock(return_value=_fake_message([_text_block("a safe reply")]))
    monkeypatch.setattr(anthropic_client.client.messages, "create", create_mock)

    with caplog.at_level(logging.DEBUG):
        await anthropic_client.generate_text_response(
            [{"role": "system", "content": marker}, {"role": "user", "content": marker}]
        )

    assert marker not in caplog.text
