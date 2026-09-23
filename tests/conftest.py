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
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-key"
# Stage 5C defense-in-depth (mirrors this file's own neutralize_proxy_env()
# philosophy): an obviously-unreachable default so that if some future code
# path ever bypassed the _default_fake_preferences autouse fixture below and
# genuinely tried to connect, it fails fast and loudly instead of silently
# reaching a real local PostgreSQL a developer happens to have running.
# tests/test_stage5c_*.py's Docker-Postgres fixtures override
# db.settings.DATABASE_URL directly (a module-attribute monkeypatch, not
# this env var) for the one test process that actually needs a real DB.
os.environ["DATABASE_URL"] = "postgresql+psycopg://invalid:invalid@127.0.0.1:1/pytest_should_never_connect"

# Stage 6A: web_config.py fails closed at import time without this (same
# posture as OPENAI_API_KEY/ANTHROPIC_API_KEY above) — set deterministically
# here so any test importing web/*.py or app/auth_session.py never depends
# on a developer's local .env defining it. Never a real secret.
os.environ["SESSION_SECRET_KEY"] = "test-session-secret-key-do-not-use-in-production"

# Stage 6B: github_oauth_config.py fails closed at import time without
# these (same posture as SESSION_SECRET_KEY above) — web.app.create_app()
# imports web.github_oauth unconditionally (see web/app.py's own
# docstring), so any test importing web.app/web.github_oauth transitively
# requires these. Never real credentials; GITHUB_REDIRECT_URI uses an
# https:// scheme purely so it validates under WEB_ENV's untouched
# "production" test default (see web_config.py) — no test ever dials it
# for real (all GitHub HTTP is mocked at the httpx transport layer, see
# services/github_oauth_client.py's own docstring), and the actual inbound
# TestClient request path (e.g. "/api/auth/github/callback") is unrelated
# to this configured value, which only ever appears as an outbound
# parameter/body value this suite asserts on.
os.environ["GITHUB_CLIENT_ID"] = "test-github-client-id-do-not-use"
os.environ["GITHUB_CLIENT_SECRET"] = "test-github-client-secret-do-not-use"
os.environ["GITHUB_REDIRECT_URI"] = "https://testserver.example/api/auth/github/callback"

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


@pytest.fixture(autouse=True)
def _default_fake_preferences(monkeypatch):
    """
    Stage 5C added a PostgreSQL-backed identity/preferences layer
    (db.identity/db.preferences) that app/identity.py's resolve_user_uuid()
    and app/session.py's UserSession.get_mode/set_mode/get_voice/set_voice
    call via asyncio.to_thread(). Without a fake here, the FIRST test in the
    session to exercise any of these would construct the real sync DB engine
    singleton against DATABASE_URL — hanging/failing against the
    deliberately-unreachable default above, or (if that default were ever
    weakened) silently writing to a real local PostgreSQL instance, exactly
    the "must never mutate real runtime state" violation this suite's other
    fixtures already guard against for the filesystem (see pytest_configure()
    below). Every test defaults to fully in-memory, per-test-isolated fakes
    here — same shape as the real db.* contract (stable UUID per Telegram
    id, (None, None) preferences for an unseen user) but backed by plain
    dicts.

    tests/test_stage5c_identity.py / test_stage5c_preferences.py /
    test_stage5c_documents_catalog.py / test_stage5c_migration.py define a
    same-named fixture that shadows this one (same mechanism
    tests/test_stage1c_access_control.py already uses against
    _default_test_access_allowed above) so they exercise the REAL db.*
    functions against a real disposable PostgreSQL container instead — see
    postgres_container()/postgres_db() below.
    """
    import db.identity as db_identity
    import db.preferences as db_preferences

    telegram_to_uuid = {}
    preferences = {}

    def fake_resolve(telegram_id):
        if telegram_id not in telegram_to_uuid:
            telegram_to_uuid[telegram_id] = uuid.uuid4()
        return telegram_to_uuid[telegram_id]

    def fake_get_preferences(user_id):
        row = preferences.get(user_id, {})
        return row.get("mode"), row.get("voice")

    def fake_set_mode(user_id, mode):
        preferences.setdefault(user_id, {})["mode"] = mode

    def fake_set_voice(user_id, voice):
        preferences.setdefault(user_id, {})["voice"] = voice

    monkeypatch.setattr(db_identity, "resolve_or_create_user_by_telegram_id_sync", fake_resolve)
    monkeypatch.setattr(db_preferences, "get_preferences_sync", fake_get_preferences)
    monkeypatch.setattr(db_preferences, "set_mode_sync", fake_set_mode)
    monkeypatch.setattr(db_preferences, "set_voice_sync", fake_set_voice)


def _install_fake_documents_catalog(monkeypatch) -> None:
    """
    Shared implementation, factored out (Stage 5C corrective pass #2,
    Section 5) so it can be reused verbatim by both the autouse fixture
    immediately below AND tests/test_stage5c_migration.py's own
    conditional override — that module's offline "Section A" tests need
    this exact in-memory fake (no real PostgreSQL reachable without
    postgres_db), while its "Section B" tests take `postgres_db` and must
    exercise REAL db.documents against it. A previous version of that
    module shadowed `_default_fake_preferences` only, leaving THIS fake
    silently active even for its "against a REAL disposable PostgreSQL
    container" tests — their own catalog assertions were therefore
    secretly checking this in-memory dict, never PostgreSQL.

    Same rationale as _default_fake_preferences above, for the document
    ownership/catalog layer (db.documents) — app/documents.py's
    _store_document_exclusively()/_load_and_index_document()/
    _cleanup_new_upload() call db.documents.create_pending_sync()/
    mark_active_sync()/delete_sync() on EVERY managed-upload ingest, inside
    the executor-thread worker functions (see db/engine.py's module
    docstring for why these are sync, not async). Any test that exercises
    the real ingest pipeline — most of tests/test_stage1b_document_upload.py,
    tests/test_stage2c_upload_lifecycle.py, etc. — would otherwise need a
    real reachable PostgreSQL. Fakes mirror the real contract: create_pending
    inserts a 'pending' row, mark_active flips it to 'active' and raises if
    no such row exists, delete is a no-op if absent.
    """
    import db.documents as db_documents

    catalog = {}

    def fake_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        catalog[document_id] = {
            "owner_user_id": owner_user_id,
            "stored_name": stored_name,
            "display_name": display_name,
            "content_sha256": content_sha256,
            "status": "pending",
            # Stage 7A-3: db_documents.DocumentRecord now carries created_at
            # (needed for the documents-list/detail API's DTO) — a plain
            # datetime.now() is a faithful enough stand-in for this fake's
            # purpose (nothing offline compares it against real elapsed
            # time), never the real server_default=func.now() a genuine
            # PostgreSQL row would carry.
            "created_at": datetime.now(),
        }

    def fake_mark_active(*, document_id):
        if document_id not in catalog:
            raise RuntimeError("mark_active_sync: no pending document row found to update")
        catalog[document_id]["status"] = "active"

    def fake_delete(*, document_id):
        catalog.pop(document_id, None)

    def fake_reconcile_ambiguous_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        # Mirrors db.documents.reconcile_ambiguous_create_pending_sync()'s
        # exact conditional-delete contract (Stage 5C corrective pass #3,
        # Blocker 3) against this same in-memory dict — without this fake,
        # app/documents.py's own call to the REAL function would try to
        # reach the deliberately-unreachable poisoned DATABASE_URL (see
        # this file's top-level comment) and simply return False from its
        # own except-clause, silently making every offline "ordinary
        # create_pending failure" test look like an incomplete cleanup.
        row = catalog.get(document_id)
        if row is None:
            return True
        if (
            row["owner_user_id"] != owner_user_id
            or row["stored_name"] != stored_name
            or row["display_name"] != display_name
            or row["content_sha256"] != content_sha256
            or row["status"] != "pending"
        ):
            return False
        catalog.pop(document_id, None)
        return True

    def fake_get(*, document_id):
        row = catalog.get(document_id)
        if row is None:
            return None
        return db_documents.DocumentRecord(
            id=document_id,
            owner_user_id=row["owner_user_id"],
            stored_name=row["stored_name"],
            display_name=row["display_name"],
            content_sha256=row["content_sha256"],
            status=row["status"],
            created_at=row["created_at"],
        )

    def fake_get_active_owners(document_ids):
        return {
            doc_id: catalog[doc_id]["owner_user_id"]
            for doc_id in document_ids
            if doc_id in catalog and catalog[doc_id]["status"] in db_documents.ACTIVE_STATUSES
        }

    monkeypatch.setattr(db_documents, "create_pending_sync", fake_create_pending)
    monkeypatch.setattr(db_documents, "mark_active_sync", fake_mark_active)
    monkeypatch.setattr(db_documents, "delete_sync", fake_delete)
    monkeypatch.setattr(db_documents, "reconcile_ambiguous_create_pending_sync", fake_reconcile_ambiguous_create_pending)
    monkeypatch.setattr(db_documents, "get_sync", fake_get)
    monkeypatch.setattr(db_documents, "get_active_owners_sync", fake_get_active_owners)


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog(monkeypatch):
    """
    Same rationale as _default_fake_preferences immediately above, for the
    document ownership/catalog layer (db.documents) — see
    _install_fake_documents_catalog() above for the actual fake and its
    full rationale.

    tests/test_stage5c_documents_catalog.py / test_stage5c_migration.py
    define a same-named fixture that shadows this one to exercise the REAL
    db.documents functions against a real disposable PostgreSQL container.
    """
    _install_fake_documents_catalog(monkeypatch)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


# Stage 5C corrective pass: an explicit test PostgreSQL DSN — set by CI
# (see .github/workflows/tests.yml's `postgres` service) — always takes
# priority over any local-Docker fallback. Never the ambient DATABASE_URL
# (that name is deliberately poisoned at module import time, above, so
# nothing can ever silently fall back to it).
TEST_POSTGRES_DSN_ENV = "TEST_POSTGRES_DSN"

# When set truthy, the absence/unreadiness of a required PostgreSQL proof
# is a hard test FAILURE, never a silent skip — this is what CI sets so a
# broken/misconfigured service can never make the Stage 5C PostgreSQL
# proof quietly disappear from the required-checks signal. Local
# developer runs leave this unset, so a missing Docker/image remains a
# clearly-reported skip rather than blocking unrelated work.
REQUIRE_POSTGRES_ENV = "PYTEST_REQUIRE_POSTGRES"

_POSTGRES_IMAGE = "postgres:16-alpine"


def _postgres_required() -> bool:
    return os.environ.get(REQUIRE_POSTGRES_ENV, "").strip().lower() in ("1", "true", "yes")


def _unavailable(reason: str) -> None:
    """Either pytest.fail() (REQUIRE_POSTGRES_ENV set — e.g. CI) or
    pytest.skip() (ordinary local dev run) for the same underlying
    condition — see REQUIRE_POSTGRES_ENV's own docstring above."""
    if _postgres_required():
        pytest.fail(f"PostgreSQL proof is required ({REQUIRE_POSTGRES_ENV}=1) but unavailable: {reason}")
    else:
        pytest.skip(f"PostgreSQL is not available — skipping PostgreSQL-backed tests: {reason}")


def _image_available_locally(image: str) -> bool:
    """True only if `image` is already present in the local Docker image
    cache — checked with `docker image inspect`, which never touches the
    network/registry (unlike `docker pull`/`docker run` without
    --pull=never, which fall back to pulling on a cache miss). Used to
    decide whether a LOCAL developer run may start a disposable container
    at all: Section 7's requirement that ordinary test execution must
    never itself trigger an unexpected Docker Hub pull."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _wait_for_real_postgres_connection(dsn: str, *, timeout_seconds: float) -> Optional[str]:
    """
    Deterministic readiness (Section 8): a container/service reporting
    "healthy" (or an in-container `pg_isready` succeeding once) can still
    be observing PostgreSQL's temporary initialization server in the
    narrow window immediately before the official server restarts —
    Codex's documented race. The only proof that actually matters is a
    REAL host-side client successfully connecting and executing a trivial
    query, repeatedly, until it stops flaking.

    Requires `_STABLE_SUCCESSES` consecutive successful `SELECT 1` round
    trips (never just one) before declaring readiness — a single success
    could still land in that same restart window by chance. Bounded by
    `timeout_seconds`; returns None on success, or a short SANITIZED
    failure reason on timeout (never the DSN itself, which embeds a
    password) for a caller to report.
    """
    import psycopg

    # psycopg.connect() speaks plain `postgresql://` connection strings —
    # not SQLAlchemy's `+<driver>` dialect suffix (`postgresql+psycopg://`,
    # the form every DSN in this codebase otherwise uses for
    # db.settings.DATABASE_URL/create_engine()). Strip it for this one
    # direct-driver readiness probe only; the DSN yielded to callers keeps
    # the SQLAlchemy form unchanged.
    psycopg_dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)

    _STABLE_SUCCESSES = 3
    deadline = time.monotonic() + timeout_seconds
    consecutive_successes = 0
    last_error_type = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(psycopg_dsn, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            consecutive_successes += 1
            if consecutive_successes >= _STABLE_SUCCESSES:
                return None
        except Exception as e:
            consecutive_successes = 0
            last_error_type = type(e).__name__
        time.sleep(0.3)
    return f"no stable connection within {timeout_seconds:.0f}s (last error_type={last_error_type})"


@pytest.fixture(scope="session")
def postgres_container():
    """
    Session-scoped PostgreSQL DSN (Stage 5C) for tests that must prove
    REAL PostgreSQL-specific behavior (unique constraints, the
    pg_advisory_xact_lock race-safety pattern, the actual Alembic
    migration path) — never pretend SQLite/mock behavior proves that.

    Three paths, in priority order (Section 6/7):

    1. TEST_POSTGRES_DSN_ENV is set — the CI path (see
       .github/workflows/tests.yml's `postgres` service). Used directly;
       this fixture never starts/stops anything itself, since the service
       container's lifecycle belongs to the CI job, not this test run.
       Readiness is still verified for real (see below) — CI's own service
       healthcheck proves the SERVICE started, not that a genuine
       host-side client round trip through the mapped port succeeds.
    2. No explicit DSN, but Docker is on PATH AND the pinned image
       (`postgres:16-alpine`) is ALREADY cached locally — starts a
       disposable, `--pull=never` (Section 7: local test execution must
       never itself trigger an unexpected Docker Hub pull), `--rm`, no-
       volume-mount container bound to 127.0.0.1 on a Docker-assigned free
       port (an explicitly pytest-socket-allowed host — see pytest.ini),
       removed in the `finally` block below.
    3. Neither — `_unavailable()` skips (ordinary local dev run) or fails
       (REQUIRE_POSTGRES_ENV=1 — e.g. CI) with a clear, specific reason.
       This is what makes "PostgreSQL proof silently disappearing from
       CI" impossible: CI always sets REQUIRE_POSTGRES_ENV, so a
       misconfigured/absent service is a hard failure there, never a skip.
    """
    explicit_dsn = os.environ.get(TEST_POSTGRES_DSN_ENV)
    if explicit_dsn:
        failure = _wait_for_real_postgres_connection(explicit_dsn, timeout_seconds=60)
        if failure is not None:
            _unavailable(f"{TEST_POSTGRES_DSN_ENV} was set but never became reachable — {failure}")
            return
        yield explicit_dsn
        return

    if not _docker_available():
        _unavailable("docker is not on PATH")
        return
    if not _image_available_locally(_POSTGRES_IMAGE):
        _unavailable(
            f"image {_POSTGRES_IMAGE!r} is not already cached locally (local runs never auto-pull — "
            f"`docker pull {_POSTGRES_IMAGE}` once, or set {TEST_POSTGRES_DSN_ENV})"
        )
        return

    container_name = f"pytutorbot_test_pg_{uuid.uuid4().hex[:8]}"
    try:
        subprocess.run(
            [
                "docker", "run", "--rm", "-d", "--pull=never", "--name", container_name,
                "-e", "POSTGRES_PASSWORD=pytest", "-e", "POSTGRES_USER=pytest", "-e", "POSTGRES_DB=pytest",
                "-p", "127.0.0.1::5432",
                "--health-cmd", "pg_isready -U pytest -d pytest",
                "--health-interval=1s", "--health-timeout=3s", "--health-retries=30", "--health-start-period=5s",
                _POSTGRES_IMAGE,
            ],
            check=True, capture_output=True, text=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        _unavailable(f"could not start a disposable PostgreSQL container — error_type={type(e).__name__}")
        return

    try:
        port_output = subprocess.run(
            ["docker", "port", container_name, "5432/tcp"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        host_port = port_output.rsplit(":", 1)[-1]
        url = f"postgresql+psycopg://pytest:pytest@127.0.0.1:{host_port}/pytest"

        # Step 1: wait for Docker's OWN healthcheck (pg_isready run INSIDE
        # the container) to report "healthy" — bounded, since --health-
        # retries=30 at a 1s interval already caps this around ~35s.
        healthy = False
        for _ in range(60):
            inspect = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
                capture_output=True, text=True, timeout=5,
            )
            if inspect.returncode == 0 and inspect.stdout.strip() == "healthy":
                healthy = True
                break
            time.sleep(0.5)
        if not healthy:
            _unavailable("container health status never reached 'healthy'")
            return

        # Step 2 (Section 8): Docker's healthcheck proves the IN-CONTAINER
        # pg_isready succeeded — Codex's documented race is that this can
        # still observe the temporary initialization server immediately
        # before the official server restarts. Only a real, repeated,
        # HOST-SIDE psycopg connection through the actual mapped port
        # proves the server this test suite will actually talk to is the
        # genuine, stable one.
        failure = _wait_for_real_postgres_connection(url, timeout_seconds=30)
        if failure is not None:
            _unavailable(f"container reported healthy but host-side connection never stabilized — {failure}")
            return

        yield url
    finally:
        subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=30)


@pytest.fixture
def postgres_db(postgres_container, monkeypatch):
    """
    Function-scoped: redirects db.settings.DATABASE_URL (read fresh by
    db.engine.get_sync_engine() on every call — see its own module
    docstring on why that matters here) to the session's disposable
    PostgreSQL container, resets the engine singleton so a fresh Engine
    binds to the redirected URL, applies the REAL Alembic migration path
    (never Base.metadata.create_all() — proves the actual operator-facing
    migration, not just that the ORM models are internally consistent),
    and truncates every table first so each test starts from a clean slate
    with zero cross-test data leakage.

    Stage 6B adds `github_accounts`/`github_oauth_transactions` to the
    TRUNCATE list below for the identical cross-test-isolation reason as
    every other non-singleton table here. Stage 6C corrective pass
    (independent-audit MAJOR 1) adds `github_unlink_tombstones` for the
    same reason — a non-singleton, per-GitHub-identity table that must
    start empty for every test.

    `web_session_policy` (Stage 6A independent-audit corrective pass #3)
    and `github_oauth_admission` (Stage 6B independent-audit corrective
    pass #1, MAJOR 2) are deliberately NOT in that TRUNCATE list: both are
    singleton rows, seeded ONCE by their own migration
    (alembic/versions/0002_web_sessions.py /
    alembic/versions/0003_github_oauth.py) — TRUNCATEing either would
    leave its table empty and make the next `SELECT ... FOR UPDATE ...
    .scalar_one()`/`.one()` against it raise NoResultFound for every
    subsequent test in the session (the container, and therefore its
    already-migrated schema, is session-scoped — `command.upgrade(cfg,
    "head")` below is a no-op on every test after the first, since
    alembic_version already reads "head"). Instead, each singleton's one
    row is reset to a known default state before every test — the same
    "known clean state" guarantee TRUNCATE gives the other tables, without
    ever deleting either row itself. For `github_oauth_admission`, without
    this reset `starts_in_window` would keep accumulating across every
    test in the session instead of giving each test the same fresh
    admission budget, and `window_start` would drift arbitrarily far into
    the past — resetting both here mirrors db.oauth_transactions.
    create_sync()'s own "reset when the window has elapsed" logic, just
    performed unconditionally for test isolation rather than only when a
    real window has actually elapsed. Stage 6C corrective pass
    (independent-audit MAJOR 1) also resets `unlink_generation` to 0 here,
    for the identical reason — without it, the global OAuth-generation
    counter would keep climbing across every test in the session instead
    of giving each test the same fresh baseline generation.
    """
    import db.engine as db_engine
    import db.settings as db_settings

    monkeypatch.setattr(db_settings, "DATABASE_URL", postgres_container)
    monkeypatch.setattr(db_engine, "_sync_engine", None)

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.attributes["sqlalchemy_url"] = postgres_container
    command.upgrade(cfg, "head")

    from sqlalchemy import text
    engine = db_engine.get_sync_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "TRUNCATE users, telegram_accounts, user_preferences, documents, web_sessions, "
            "github_accounts, github_oauth_transactions, telegram_link_attempts, "
            "github_unlink_tombstones CASCADE"
        ))
        conn.execute(text("UPDATE web_session_policy SET current_secure = true, updated_at = now()"))
        conn.execute(text(
            "UPDATE github_oauth_admission SET window_start = now(), starts_in_window = 0, unlink_generation = 0"
        ))

    yield postgres_container


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
