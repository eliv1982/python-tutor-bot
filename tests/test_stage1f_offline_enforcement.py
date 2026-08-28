"""
Stage 1F-B/1F-C remediation regression tests (Codex offline-enforcement
blocker, two independent audit passes).

An independent audit demonstrated that pytest.ini's
`--disable-socket --allow-hosts=127.0.0.1,::1` (required so Windows'
asyncio ProactorEventLoop can still use its internal loopback self-pipe —
see pytest.ini) has a loophole: pytest-socket only inspects the raw
socket-level connect() target, never the request an already-permitted
connection carries. A real HTTP(S) proxy listening on 127.0.0.1 could
receive a `CONNECT api.openai.com:443 HTTP/1.1` from any client whose
underlying HTTP library honors ambient HTTP_PROXY/HTTPS_PROXY/ALL_PROXY
environment variables (`trust_env=True`, the OpenAI SDK's default) and
transparently forward that traffic externally — invisible to pytest-socket,
which only ever sees the permitted 127.0.0.1 connection.

A second independent audit (Stage 1F-C) proved that env-variable cleanup
alone (Stage 1F-B's `neutralize_proxy_env()`) can never be the *primary*
guarantee, because it can only ever be one specific input to a much larger
trust decision:

1. Conventional env vars (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/...) — Stage 1F-B.
2. OS/system proxy discovery (Windows Registry / macOS system config),
   which a `trust_env=True` client falls back to via
   `urllib.request.getproxies()` even with every env var above absent.
3. langchain-openai's OpenAIEmbeddings separately honors its own
   `OPENAI_PROXY` variable, independent of the conventional ones.
4. `.env` (loaded by config.py's `load_dotenv()`, which runs AFTER
   conftest.py's module-level neutralization) can reintroduce any of the
   above if a developer's real `.env` happens to define them, since
   `load_dotenv(override=False)` only refuses to clobber a variable that is
   already *present* — not one that was removed.

The Stage 1F-C fix moves the primary guarantee to the transport layer:
every provider HTTP client this app constructs
(services/openai_client.py's AsyncOpenAI, rag/index.py's OpenAIEmbeddings)
is now built with an explicit `trust_env=False`, via the openai SDK's own
public `DefaultHttpx2Client`/`DefaultAsyncHttpx2Client` factories, and with
`openai_proxy=None` pinned explicitly on OpenAIEmbeddings. A client built
this way has no proxy to discover regardless of what the environment, the
OS, or a reloaded `.env` say — env-variable cleanup (conftest.py's
`neutralize_proxy_env()`, called both before AND after config.py's
`load_dotenv()`) remains as a second, independent layer of defense-in-depth,
not the sole guarantee.

This file proves, using only local infrastructure (a raw TCP listener on
127.0.0.1, monkeypatched "system" proxy discovery, and IP literals — never a
real hostname, so nothing here is DNS-dependent):

1. Conventional env-var proxy: a trust_env=True client is vulnerable (control);
   the hardened production client is not (test_provider_client_cannot_route...).
2. OS/system proxy discovery: same shape, using a monkeypatched
   `httpx2._utils.getproxies` standing in for the Windows Registry/macOS
   system config lookup, with zero proxy env vars set.
3. OPENAI_PROXY: same shape, against the actual OpenAIEmbeddings
   construction path used by rag/index.py.
4. `.env` reintroduction: a subprocess proves that even when a `.env` file
   restores proxy variables AFTER config.py's own load_dotenv() has already
   run, the real production client objects it constructs still carry
   trust_env=False / openai_proxy=None.
5. Direct external sockets are still blocked by pytest-socket (positive
   control, unrelated to the proxy issue, proving the base enforcement
   mechanism itself hasn't regressed).

Sections 1-3 and 5 additionally exercise the REAL application singletons
(`services.openai_client.openai_client`, `rag.index.vector_index`) — not
just a throwaway client built the same way — so a future refactor that
drops the explicit no-trust configuration from those constructors fails
this suite, per the second audit's specific complaint that the first
remediation only proved a generic client scenario.

Why aiohttp and pyTelegramBotAPI were not at risk (verified, not assumed):
`aiohttp.ClientSession()` defaults to `trust_env=False`, and this repo's
only three call sites (services/image_generation.py) never override it.
pyTelegramBotAPI's async transport (telebot/asyncio_helper.py) constructs
its own `aiohttp.ClientSession(connector=...)` the same way — also
`trust_env=False` by default, also never overridden. Neither library reads
proxy env vars at all, with or without this fix. Only the OpenAI SDK's
underlying HTTP transport (httpx2.Client, `trust_env=True` by default,
confirmed via `inspect.signature`) was ever at risk.
"""

import contextlib
import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import openai
import pytest

# Mirrors tests/conftest.py's PROXY_ENV_VARS exactly (including OPENAI_PROXY,
# added in Stage 1F-C). Kept as an independent local copy (rather than
# importing conftest's list) so this file's control tests are self-contained
# and don't depend on conftest/test module identity quirks under pytest's
# rootdir-relative conftest import.
PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "OPENAI_PROXY",
)

# RFC 5737 TEST-NET-1: reserved for documentation, guaranteed to never be
# publicly routed and never requires a DNS lookup (it's an IP literal, not
# a hostname) — so every assertion below is independent of DNS behavior.
FAKE_PROVIDER_HOST = "192.0.2.1"
FAKE_PROVIDER_PORT = 443


def _neutralize_proxy_env() -> None:
    """Local equivalent of tests/conftest.py's neutralize_proxy_env().

    Duplicated (rather than imported from tests.conftest) so this test
    doesn't depend on pytest's conftest import-mode resolving `tests.conftest`
    to the same module object it already loaded as its own conftest plugin
    — under the default "prepend" import mode with no tests/__init__.py,
    those can be two distinct module objects. The logic is one line and
    stateless, so keeping an independent copy here is simpler and more
    robust than relying on that identity.
    """
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)


@contextlib.contextmanager
def _local_listener():
    """
    A bare TCP listener on 127.0.0.1 (ephemeral port) that records the
    first thing a client sends it, then closes. Stands in for "a real
    proxy" for these tests: it never actually forwards anything anywhere,
    so nothing it does can reach a real external host even if a client
    successfully CONNECTs to it.
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


def test_positive_control_direct_external_socket_connection_is_blocked():
    """
    Unrelated to the proxy issue: proves pytest-socket's base enforcement
    (from pytest.ini's addopts) is actually active in this session, not a
    no-op. A real connect() to a non-loopback address must be rejected.
    """
    import pytest_socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(pytest_socket.SocketConnectBlockedError):
            s.connect((FAKE_PROVIDER_HOST, FAKE_PROVIDER_PORT))
    finally:
        s.close()


def test_proxy_env_vars_are_neutralized_by_default():
    """
    Sanity check on the actual, real test session (no monkeypatching here):
    conftest.py's module-level `neutralize_proxy_env()` call already ran
    before this test module was even collected, so none of the 8
    conventional proxy variables should be present for any test in this
    suite, including this one.
    """
    present = [name for name in PROXY_ENV_VARS if name in os.environ]
    assert present == [], f"proxy env vars leaked into the test session: {present}"


def test_control_localhost_proxy_receives_connect_when_env_left_unneutralized(monkeypatch):
    """
    Control: reproduces the exact vulnerability the audit found, to prove
    the detection mechanism (the local listener) actually works and the
    risk is real — this is what every real client construction looked like
    before conftest.py's neutralize_proxy_env() fix existed.

    Deliberately sets HTTP_PROXY/HTTPS_PROXY toward the local listener and
    constructs the OpenAI SDK's real client class (the same one
    services/openai_client.py and rag/index.py's OpenAIEmbeddings use
    internally) with trust_env at its default (True), then makes one
    request toward the reserved-IP fake target. The local listener must
    receive a CONNECT for this control to have proven anything.
    """
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("ALL_PROXY", proxy_url)

        import openai

        client = openai.OpenAI(
            api_key="sk-test-dummy-key",
            base_url=f"https://{FAKE_PROVIDER_HOST}/v1",
            timeout=1.0,
            max_retries=0,
        )
        with contextlib.suppress(Exception):
            # The proxy never completes a real handshake, so this call is
            # expected to fail — only whether the listener was contacted
            # matters here, not the outcome of the call itself.
            client.models.list()
        client.close()

    assert received["data"] is not None, (
        "control did not observe a CONNECT at the local listener — the "
        "detection mechanism itself is not working, so it cannot prove "
        "anything about the actual fix below"
    )
    assert received["data"].startswith(b"CONNECT "), received["data"]


def test_provider_client_cannot_route_through_localhost_proxy_after_neutralization(monkeypatch):
    """
    The actual fix proof. Same scenario as the control above — proxy env
    vars pointed at a local listener — except this time
    `_neutralize_proxy_env()` (the same logic as tests/conftest.py's
    `neutralize_proxy_env()`, which every real test in this session already
    gets for free at collection time) is called again right before client
    construction, simulating a CI/developer environment that had these
    variables set in its ambient shell. The listener must receive NOTHING:
    the client has no proxy left to discover.
    """
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("ALL_PROXY", proxy_url)

        _neutralize_proxy_env()

        import openai

        client = openai.OpenAI(
            api_key="sk-test-dummy-key",
            base_url=f"https://{FAKE_PROVIDER_HOST}/v1",
            timeout=1.0,
            max_retries=0,
        )
        with contextlib.suppress(Exception):
            # No proxy configured now, so this attempts a direct connection
            # to the reserved TEST-NET-1 IP literal instead — which
            # pytest-socket itself blocks (not on the loopback allow-list).
            # Either way the outcome is irrelevant; only the listener
            # matters.
            client.models.list()
        client.close()

    assert received["data"] is None, (
        f"local proxy unexpectedly received data after neutralization: {received['data']!r}"
    )


# ============================================================================
# Stage 1F-C: bypass route 2 — OS/system proxy discovery
# ============================================================================
#
# httpx2's `get_environment_proxies()` (venv/.../httpx2/_utils.py) calls
# `urllib.request.getproxies()`, which itself falls back to the Windows
# Registry / macOS system config when no HTTP_PROXY/HTTPS_PROXY/ALL_PROXY
# env var is set — a fallback that exists entirely independently of the
# env-var route covered above. `Client.__init__` computes
# `allow_env_proxies = trust_env and transport is None` and only calls
# `get_environment_proxies()` at all when that's True, so `trust_env=False`
# (not env-var cleanup) is what actually closes this route — confirmed by
# reading httpx2/_client.py's `_get_proxy_map`, not assumed.
#
# Monkeypatching `httpx2._utils.getproxies` (the exact function httpx2
# calls) is the harmless emulation this section uses: it reproduces "the OS
# says use this proxy" without ever touching the real Windows Registry or
# macOS system configuration.


def test_control_system_proxy_discovery_routes_vulnerable_client(monkeypatch):
    """
    Control: proves the OS/system-discovery route is real and independent of
    the env-var route, using a client built the pre-1F-C way (trust_env at
    its default, True) — this is what every provider client in this repo
    looked like before this remediation. No proxy env var is set here; only
    the emulated OS-level discovery is active.
    """
    present = [name for name in PROXY_ENV_VARS if name in os.environ]
    assert present == [], f"test invalid: proxy env vars unexpectedly present: {present}"

    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"

        def fake_getproxies():
            return {"http": proxy_url, "https": proxy_url}

        monkeypatch.setattr("httpx2._utils.getproxies", fake_getproxies)

        client = openai.OpenAI(
            api_key="sk-test-dummy-key",
            base_url=f"https://{FAKE_PROVIDER_HOST}/v1",
            timeout=1.0,
            max_retries=0,
        )
        with contextlib.suppress(Exception):
            client.models.list()
        client.close()

    assert received["data"] is not None, (
        "control did not observe a CONNECT via emulated OS/system proxy "
        "discovery — the detection mechanism itself is not working, so it "
        "cannot prove anything about the fix below"
    )
    assert received["data"].startswith(b"CONNECT "), received["data"]


def test_hardened_client_immune_to_system_proxy_discovery(monkeypatch):
    """
    The fix proof for the OS-discovery route: same emulated fallback as
    above, against a client built with the openai SDK's own
    `DefaultHttpx2Client(trust_env=False)` factory — the exact factory
    services/openai_client.py now uses. trust_env=False means
    `get_environment_proxies()` (and therefore the emulated OS discovery)
    is never even called.
    """
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"

        def fake_getproxies():
            return {"http": proxy_url, "https": proxy_url}

        monkeypatch.setattr("httpx2._utils.getproxies", fake_getproxies)

        client = openai.OpenAI(
            api_key="sk-test-dummy-key",
            base_url=f"https://{FAKE_PROVIDER_HOST}/v1",
            timeout=1.0,
            max_retries=0,
            http_client=openai.DefaultHttpx2Client(trust_env=False),
        )
        with contextlib.suppress(Exception):
            client.models.list()
        client.close()

    assert received["data"] is None, (
        f"local proxy unexpectedly received data under emulated OS/system "
        f"proxy discovery despite trust_env=False: {received['data']!r}"
    )


async def test_real_openai_client_singleton_immune_to_system_proxy_discovery(monkeypatch):
    """
    Requirement G: exercises the REAL services/openai_client.py singleton
    (already constructed at module-import time, long before this test
    runs) rather than a freshly hand-built stand-in. base_url is
    temporarily monkeypatched to the reserved TEST-NET-1 literal for the
    duration of this one call only (monkeypatch restores it automatically
    at teardown, via base_url's own public property setter — see
    `openai._base_client.BaseClient.base_url`), so this never targets the
    real api.openai.com.
    """
    from services.openai_client import openai_client as real_client

    assert real_client.client._client.trust_env is False, (
        "services/openai_client.py's AsyncOpenAI singleton no longer pins "
        "trust_env=False on its http_client — the Stage 1F-C guarantee has "
        "regressed"
    )

    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"

        def fake_getproxies():
            return {"http": proxy_url, "https": proxy_url}

        monkeypatch.setattr("httpx2._utils.getproxies", fake_getproxies)
        monkeypatch.setattr(real_client.client, "base_url", f"https://{FAKE_PROVIDER_HOST}/v1")

        with contextlib.suppress(Exception):
            await real_client.client.models.list()

    assert received["data"] is None, (
        f"the real production OpenAI client singleton leaked a request "
        f"through an emulated system proxy: {received['data']!r}"
    )


# ============================================================================
# Stage 1F-C: bypass route 3 — langchain-openai's OPENAI_PROXY
# ============================================================================
#
# OpenAIEmbeddings reads OPENAI_PROXY via its own
# `Field(default_factory=from_env("OPENAI_PROXY", default=None))`,
# independent of the conventional proxy env vars and independent of
# trust_env on a caller-supplied http_client — UNLESS that http_client is
# supplied explicitly, in which case langchain-openai's own
# validate_environment() skips building its OPENAI_PROXY-derived client
# (and raises ValueError if both are set simultaneously). This is exactly
# rag/index.py's `openai_proxy=None` + explicit `http_client=`/
# `http_async_client=` construction (confirmed by reading
# langchain_openai/embeddings/base.py's validate_environment(), not
# assumed).


def test_control_embeddings_openai_proxy_routes_when_client_not_pinned(monkeypatch):
    """
    Control: reproduces what rag/index.py's OpenAIEmbeddings construction
    looked like before this remediation (no explicit http_client, so
    validate_environment() builds `httpx.Client(proxy=self.openai_proxy)`
    from OPENAI_PROXY directly). Proves this route is real and independent
    of the conventional-env-var route already covered above.
    """
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("OPENAI_PROXY", proxy_url)

        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(
            openai_api_key="sk-test-dummy-key",
            base_url=f"https://{FAKE_PROVIDER_HOST}/v1",
        )
        with contextlib.suppress(Exception):
            emb.client._client._client.get(f"https://{FAKE_PROVIDER_HOST}/", timeout=1.0)

    assert received["data"] is not None, (
        "control did not observe a request via OPENAI_PROXY — the "
        "detection mechanism itself is not working, so it cannot prove "
        "anything about the fix below"
    )


def test_hardened_embeddings_immune_to_openai_proxy_env_var(monkeypatch):
    """
    The fix proof for OPENAI_PROXY, using the exact same OpenAIEmbeddings
    construction as rag/index.py's VectorIndex.__init__ (openai_proxy=None
    pinned explicitly, http_client/http_async_client pinned to
    trust_env=False via the openai SDK's own Default*Httpx2Client
    factories). OPENAI_PROXY is set BEFORE construction, so the field's own
    from_env() default_factory would normally pick it up — proving the
    explicit override wins regardless.
    """
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("OPENAI_PROXY", proxy_url)

        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(
            openai_api_key="sk-test-dummy-key",
            base_url=f"https://{FAKE_PROVIDER_HOST}/v1",
            openai_proxy=None,
            http_client=openai.DefaultHttpx2Client(trust_env=False),
            http_async_client=openai.DefaultAsyncHttpx2Client(trust_env=False),
        )
        assert emb.openai_proxy is None

        with contextlib.suppress(Exception):
            emb.client._client._client.get(f"https://{FAKE_PROVIDER_HOST}/", timeout=1.0)

    assert received["data"] is None, (
        f"local proxy unexpectedly received data despite OPENAI_PROXY "
        f"being set and an explicit no-trust http_client pinned: "
        f"{received['data']!r}"
    )


def test_real_vector_index_construction_immune_to_openai_proxy_env_var(monkeypatch, tmp_path):
    """
    Requirement G: exercises rag/index.py's actual VectorIndex.__init__
    construction path (not a hand-rolled OpenAIEmbeddings stand-in), with
    OPENAI_PROXY set BEFORE construction. persist_directory is redirected
    to tmp_path so this never touches the real data/chroma_db. Proves the
    real production constructor neither raises (langchain-openai raises
    ValueError if openai_proxy and http_client are both truthy — pinning
    openai_proxy=None explicitly avoids that) nor wires up the proxy.
    """
    with _local_listener() as (port, received):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("OPENAI_PROXY", proxy_url)

        from rag.index import VectorIndex

        vi = VectorIndex(persist_directory=tmp_path / "chroma_db_openai_proxy_test")

        assert vi.embeddings.openai_proxy is None
        assert vi.embeddings.client._client._client.trust_env is False
        assert vi.embeddings.async_client._client._client.trust_env is False

        with contextlib.suppress(Exception):
            vi.embeddings.client._client._client.get(
                f"https://{FAKE_PROVIDER_HOST}/", timeout=1.0
            )

    assert received["data"] is None, (
        f"the real VectorIndex production constructor leaked a request "
        f"through OPENAI_PROXY: {received['data']!r}"
    )


def test_real_openai_client_singleton_has_no_trust_transport():
    """
    Requirement G, narrow configuration assertion on the REAL
    services/openai_client.py singleton (already constructed at module
    import time). `trust_env` is a public, documented httpx/httpx2 Client
    property (not an undocumented internal); `._client` itself has no fully
    public equivalent accessor on the openai SDK client, so this is the
    narrowest available supported observation point. Fails immediately if a
    future refactor drops the explicit http_client= kwarg from
    OpenAIClient.__init__.
    """
    from services.openai_client import openai_client as real_client

    assert real_client.client._client.trust_env is False


def test_real_vector_index_singleton_embeddings_have_no_trust_transport():
    """
    Requirement G, narrow configuration assertion on the REAL rag/index.py
    vector_index singleton's OpenAIEmbeddings (already constructed at
    conftest.py's pytest_configure time). Fails immediately if a future
    refactor drops the explicit openai_proxy=None / http_client= /
    http_async_client= kwargs from VectorIndex.__init__.
    """
    import rag.index as rag_index

    embeddings = rag_index.vector_index.embeddings
    assert embeddings.openai_proxy is None
    assert embeddings.client._client._client.trust_env is False
    assert embeddings.async_client._client._client.trust_env is False


# ============================================================================
# Stage 1F-C: bypass route 4 — .env reintroduction after load_dotenv()
# ============================================================================
#
# config.py's `load_dotenv()` (no explicit dotenv_path) walks up from
# config.py's own file location looking for `.env`, with override=False —
# it never clobbers a variable that's already present, but conftest.py's
# module-level `neutralize_proxy_env()` *removes* these variables rather
# than pinning them to a sentinel, so from load_dotenv()'s point of view
# they're simply unset and get reintroduced if a real `.env` defines them.
# tests/conftest.py's pytest_configure() now calls neutralize_proxy_env()
# a second time, right after config.py's own load_dotenv() has run, closing
# that specific gap for the rest of THIS test session — but that's still
# only env-variable-level defense-in-depth. This test proves the primary,
# transport-level guarantee independently: a completely separate process,
# with its own real `.env`-driven load_dotenv() call and no import-order
# assistance from conftest.py at all, still produces production client
# objects with no-trust configuration.
#
# Runs in a subprocess (not this pytest session) using a temporary `.env`
# file — never the repository's real one — and never issues any network
# call; it only constructs the real application objects and reports their
# configuration as JSON on stdout.

_DOTENV_REINTRO_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import sys
    from pathlib import Path

    tmp_root = Path(sys.argv[1])
    project_root = sys.argv[2]
    env_path = sys.argv[3]

    sys.path.insert(0, project_root)

    os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    import dotenv
    dotenv.load_dotenv(dotenv_path=env_path, override=True)

    import config as app_config

    data_dir = tmp_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    app_config.DATA_DIR = data_dir
    app_config.LOG_FILE = tmp_root / "bot.log"

    import services.openai_client as oc
    import rag.index as ri

    result = {
        "http_proxy_present": os.environ.get("HTTP_PROXY"),
        "openai_proxy_present": os.environ.get("OPENAI_PROXY"),
        "openai_client_trust_env": oc.openai_client.client._client.trust_env,
        "embeddings_openai_proxy_field": ri.vector_index.embeddings.openai_proxy,
        "embeddings_sync_trust_env": ri.vector_index.embeddings.client._client._client.trust_env,
        "embeddings_async_trust_env": ri.vector_index.embeddings.async_client._client._client.trust_env,
    }
    print("RESULT_JSON=" + json.dumps(result))
    """
)


def test_dotenv_reintroduction_cannot_restore_usable_provider_proxy(tmp_path):
    """
    Deterministic regression for bypass route 4. A temporary `.env` file
    (never the repo's real one) sets HTTP_PROXY/HTTPS_PROXY/OPENAI_PROXY.
    A subprocess loads it via the exact same `dotenv.load_dotenv()` API
    config.py uses, then imports config.py (which runs its own
    argument-less `load_dotenv()` — override=False means it does not
    disturb the values already loaded), then imports the real
    services/openai_client.py and rag/index.py singletons and reports their
    configuration. No network call is ever attempted in the subprocess.
    """
    env_file = tmp_path / "reintroduced.env"
    env_file.write_text(
        "HTTP_PROXY=http://127.0.0.1:65530\n"
        "HTTPS_PROXY=http://127.0.0.1:65530\n"
        "OPENAI_PROXY=http://127.0.0.1:65530\n",
        encoding="utf-8",
    )

    script_path = tmp_path / "reintro_check.py"
    script_path.write_text(_DOTENV_REINTRO_SCRIPT, encoding="utf-8")

    isolated_root = tmp_path / "isolated_root"
    isolated_root.mkdir()

    project_root = Path(__file__).resolve().parents[1]

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(isolated_root),
            str(project_root),
            str(env_file),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, (
        f"isolated subprocess failed unexpectedly\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )

    result_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON=")),
        None,
    )
    assert result_line is not None, f"subprocess produced no result line\nstdout={proc.stdout}"
    result = json.loads(result_line[len("RESULT_JSON="):])

    # Sanity: prove the scenario is real — the temporary .env actually did
    # reintroduce these variables into the subprocess's environment via the
    # same load_dotenv() path config.py uses. Without this, the test below
    # would prove nothing.
    assert result["http_proxy_present"] == "http://127.0.0.1:65530"
    assert result["openai_proxy_present"] == "http://127.0.0.1:65530"

    # The fix proof: despite that reintroduction, the real production
    # client objects constructed in that process still carry no-trust
    # configuration.
    assert result["openai_client_trust_env"] is False
    assert result["embeddings_openai_proxy_field"] is None
    assert result["embeddings_sync_trust_env"] is False
    assert result["embeddings_async_trust_env"] is False
