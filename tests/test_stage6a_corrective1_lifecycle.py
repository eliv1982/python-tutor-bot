"""
Stage 6A independent-audit corrective pass #1 — the three MINOR
architecture/lifecycle findings:

1. Configuration layering: app/auth_session.py must be importable without
   SESSION_SECRET_KEY (it only needs session_config.SESSION_TTL_SECONDS,
   never web_config.py's CSRF secret/cookie posture) — proven both
   statically (it must not even import web_config) and dynamically (a
   subprocess import with no SESSION_SECRET_KEY set at all must succeed).
2. Cancellation-created orphan session — evaluated and documented as a
   non-blocking, bounded lifecycle issue (see app/auth_session.py's
   create_session() docstring for the full reasoning); deliberately no
   cross-thread cancellation/reconciliation machinery was added for it, so
   there is no new *behavior* here to regression-test beyond what
   tests/test_stage6a_app_auth_session.py already covers.
3. DB engine shutdown: web/app.py's create_app() now registers a lifespan
   hook that disposes the shared sync DB engine on ASGI shutdown, mirroring
   main.py's own shutdown_bot() for the Telegram adapter — proven below by
   actually entering/exiting a TestClient context and observing the shared
   engine singleton get disposed.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from starlette.testclient import TestClient

from web.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- finding 1: app/auth_session.py import isolation ------------------------


def test_app_auth_session_does_not_statically_import_web_config():
    """AST-level check (not merely 'it happened to work this run') that
    app/auth_session.py has no `import web_config` / `from web_config
    import ...` anywhere — the actual coupling the audit flagged."""
    source = (PROJECT_ROOT / "app" / "auth_session.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    assert "web_config" not in imported_names


_IMPORT_APP_AUTH_SESSION_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    project_root = sys.argv[1]
    sys.path.insert(0, project_root)

    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False

    try:
        import app.auth_session
    except Exception as e:
        print("IMPORT_RESULT=" + json.dumps({"raised": True, "error_type": type(e).__name__}))
        sys.exit(0)

    print("IMPORT_RESULT=" + json.dumps({"raised": False}))
    """
)


def _base_subprocess_env() -> dict:
    env = {}
    for name in ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def test_app_auth_session_imports_successfully_with_no_session_secret_key_at_all():
    """Dynamic proof to match the static one above: a completely bare
    environment (deliberately no SESSION_SECRET_KEY, no WEB_COOKIE_SECURE,
    no WEB_ENV — none of web_config.py's variables) must still let
    `import app.auth_session` succeed, in a fresh subprocess/interpreter
    where nothing has imported web_config.py yet."""
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_APP_AUTH_SESSION_SCRIPT, str(PROJECT_ROOT)],
        capture_output=True, text=True, timeout=30, env=_base_subprocess_env(),
    )
    result_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("IMPORT_RESULT=")), None
    )
    assert result_line is not None, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    result = json.loads(result_line[len("IMPORT_RESULT="):])
    assert result["raised"] is False


# --- finding 3: FastAPI lifespan disposes the shared DB engine -------------


def test_app_lifespan_disposes_the_shared_db_engine_on_shutdown(postgres_db):
    import db.engine as db_engine

    with TestClient(create_app()) as client:
        response = client.get("/healthz")
        assert response.status_code == 200

        # Construct the shared engine singleton explicitly within the
        # running lifespan, so there is something real for shutdown to
        # dispose.
        db_engine.get_sync_engine()
        assert db_engine._sync_engine is not None

    assert db_engine._sync_engine is None


def test_engine_is_usable_again_after_a_lifespan_shutdown(postgres_db):
    """close_db()'s own contract (db/engine.py) is "safe no-op if never
    constructed, and a later get_sync_engine() call constructs a fresh
    one" — confirm the lifespan hook preserves that: the adapter isn't left
    unable to serve a next request merely because a PREVIOUS TestClient's
    lifespan already shut one engine down (relevant for repeated
    `with TestClient(...)` blocks against the same process, as tests
    themselves do)."""
    import db.engine as db_engine

    with TestClient(create_app()) as client:
        client.get("/healthz")
        db_engine.get_sync_engine()

    with TestClient(create_app()) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        engine = db_engine.get_sync_engine()
        with engine.begin() as conn:
            from sqlalchemy import text
            assert conn.execute(text("SELECT 1")).scalar_one() == 1


def test_bare_testclient_construction_never_runs_the_lifespan(monkeypatch):
    """The existing test suite constructs `TestClient(create_app())`
    WITHOUT entering it as a context manager for the vast majority of
    Stage 6A tests — confirm that pattern still never triggers the new
    shutdown hook (i.e. never disposes an engine those other tests may
    still be relying on), so adding the hook is provably inert for every
    pre-existing test that doesn't opt into lifespan handling."""
    import db.engine as db_engine

    monkeypatch.setattr(db_engine, "_sync_engine", "sentinel-not-a-real-engine")
    client = TestClient(create_app())
    client.get("/healthz")
    assert db_engine._sync_engine == "sentinel-not-a-real-engine"
