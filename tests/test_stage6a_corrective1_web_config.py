"""
Stage 6A independent-audit corrective pass #1 — configuration-hardening
regression tests for web_config.py (Blocker 2, Major 4) and session_config.py
(Major 4's TTL half). All import-time validation, so every scenario is run
in an isolated subprocess (same technique
tests/test_stage2a_text_llm_provider.py's _CONFIG_IMPORT_SCRIPT already
established for config.py) — this session's own already-imported
web_config/session_config modules are never disturbed, and a developer's
real .env (regardless of its contents) can never influence the result
(dotenv.load_dotenv is stubbed to a no-op before either module is
imported).

Covers:
- Blocker 2: WEB_COOKIE_SECURE strict boolean parsing (malformed value
  fails closed, valid true/false spellings accepted) and the
  WEB_ENV=production + WEB_COOKIE_SECURE=false guard.
- Major 4: SESSION_SECRET_KEY minimum-length/whitespace validation, and
  WEB_SESSION_TTL_SECONDS bounds validation (session_config.py).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A 43-character (256-bit-equivalent) secret — comfortably over the
# 32-byte minimum — reused across tests that need a VALID secret to reach
# the behavior under test (boolean parsing / WEB_ENV guard / TTL), without
# each test needing to construct its own.
_VALID_SECRET = "a" * 43

_MODULE_IMPORT_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    project_root = sys.argv[1]
    module_name = sys.argv[2]
    sys.path.insert(0, project_root)

    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False

    try:
        module = __import__(module_name)
    except Exception as e:
        print("IMPORT_RESULT=" + json.dumps({
            "raised": True,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }))
        sys.exit(0)

    result = {"raised": False}
    for attr in ("COOKIE_SECURE", "WEB_ENV", "SESSION_SECRET_KEY", "SESSION_TTL_SECONDS"):
        if hasattr(module, attr):
            result[attr] = getattr(module, attr)
    print("IMPORT_RESULT=" + json.dumps(result))
    """
)


def _base_subprocess_env() -> dict:
    """Only enough for the interpreter itself to start — no
    SESSION_SECRET_KEY/WEB_ENV/WEB_COOKIE_SECURE/WEB_SESSION_TTL_SECONDS
    unless a test explicitly adds them via env_overrides, and never the
    real ambient environment's values for any of those (mirrors
    test_stage2a_text_llm_provider.py's _base_subprocess_env())."""
    env = {}
    for name in ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _run_module_import(module_name: str, env_overrides: dict) -> dict:
    env = _base_subprocess_env()
    env.update(env_overrides)

    proc = subprocess.run(
        [sys.executable, "-c", _MODULE_IMPORT_SCRIPT, str(PROJECT_ROOT), module_name],
        capture_output=True, text=True, timeout=30, env=env,
    )
    result_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("IMPORT_RESULT=")), None
    )
    assert result_line is not None, (
        f"no IMPORT_RESULT line from subprocess importing {module_name!r}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(result_line[len("IMPORT_RESULT="):])


# --- web_config.py: SESSION_SECRET_KEY (Major 4) ---------------------------


def test_missing_secret_fails_closed_at_import():
    result = _run_module_import("web_config", {})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_empty_secret_fails_closed_at_import():
    result = _run_module_import("web_config", {"SESSION_SECRET_KEY": ""})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_whitespace_only_secret_fails_closed_at_import():
    result = _run_module_import("web_config", {"SESSION_SECRET_KEY": "    \t   "})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_too_short_secret_fails_closed_at_import():
    result = _run_module_import("web_config", {"SESSION_SECRET_KEY": "short-secret"})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_one_character_secret_fails_closed_at_import():
    """The auditor's exact reproduction: a one-character secret used to be
    accepted."""
    result = _run_module_import("web_config", {"SESSION_SECRET_KEY": "x"})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_valid_strong_secret_is_accepted():
    result = _run_module_import("web_config", {"SESSION_SECRET_KEY": _VALID_SECRET})
    assert result["raised"] is False
    assert result["SESSION_SECRET_KEY"] == _VALID_SECRET


# --- web_config.py: WEB_COOKIE_SECURE strict boolean parsing (Blocker 2) ---


def test_malformed_boolean_fails_closed_at_import():
    """The auditor's exact reproduction: an unrecognized value used to be
    silently coerced to false, disabling the Secure cookie flag."""
    result = _run_module_import(
        "web_config",
        {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_COOKIE_SECURE": "definitely-not-a-boolean"},
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES"])
def test_valid_true_spellings_are_accepted(value):
    result = _run_module_import(
        "web_config", {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_COOKIE_SECURE": value}
    )
    assert result["raised"] is False
    assert result["COOKIE_SECURE"] is True


@pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "NO"])
def test_valid_false_spellings_are_accepted_in_development(value):
    result = _run_module_import(
        "web_config",
        {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_ENV": "development", "WEB_COOKIE_SECURE": value},
    )
    assert result["raised"] is False
    assert result["COOKIE_SECURE"] is False


def test_default_posture_is_secure_when_web_cookie_secure_is_unset():
    result = _run_module_import("web_config", {"SESSION_SECRET_KEY": _VALID_SECRET})
    assert result["raised"] is False
    assert result["COOKIE_SECURE"] is True
    assert result["WEB_ENV"] == "production"


# --- web_config.py: production/insecure-cookie guard (Blocker 2) -----------


def test_production_with_insecure_cookie_is_rejected():
    """The auditor's exact reproduction: no guard prevented
    WEB_ENV=production (or its unset default) from also setting
    WEB_COOKIE_SECURE=false."""
    result = _run_module_import(
        "web_config", {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_COOKIE_SECURE": "false"}
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_explicit_production_with_insecure_cookie_is_rejected():
    result = _run_module_import(
        "web_config",
        {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_ENV": "production", "WEB_COOKIE_SECURE": "false"},
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_development_with_insecure_cookie_is_allowed():
    result = _run_module_import(
        "web_config",
        {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_ENV": "development", "WEB_COOKIE_SECURE": "false"},
    )
    assert result["raised"] is False
    assert result["COOKIE_SECURE"] is False


def test_development_with_secure_cookie_is_allowed():
    result = _run_module_import(
        "web_config",
        {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_ENV": "development", "WEB_COOKIE_SECURE": "true"},
    )
    assert result["raised"] is False
    assert result["COOKIE_SECURE"] is True


def test_invalid_web_env_value_fails_closed():
    result = _run_module_import(
        "web_config", {"SESSION_SECRET_KEY": _VALID_SECRET, "WEB_ENV": "staging"}
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


# --- session_config.py: WEB_SESSION_TTL_SECONDS bounds (Major 4) -----------


def test_ttl_zero_is_rejected():
    result = _run_module_import("session_config", {"WEB_SESSION_TTL_SECONDS": "0"})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_negative_is_rejected():
    result = _run_module_import("session_config", {"WEB_SESSION_TTL_SECONDS": "-1"})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_excessively_large_is_rejected():
    """Well beyond the documented 180-day maximum."""
    result = _run_module_import("session_config", {"WEB_SESSION_TTL_SECONDS": str(60 * 60 * 24 * 3650)})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_non_integer_is_rejected():
    result = _run_module_import("session_config", {"WEB_SESSION_TTL_SECONDS": "not-a-number"})
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_default_is_fourteen_days():
    result = _run_module_import("session_config", {})
    assert result["raised"] is False
    assert result["SESSION_TTL_SECONDS"] == 60 * 60 * 24 * 14


def test_ttl_valid_custom_value_is_accepted():
    result = _run_module_import("session_config", {"WEB_SESSION_TTL_SECONDS": "3600"})
    assert result["raised"] is False
    assert result["SESSION_TTL_SECONDS"] == 3600


def test_ttl_at_the_maximum_boundary_is_accepted():
    max_ttl = 60 * 60 * 24 * 180
    result = _run_module_import("session_config", {"WEB_SESSION_TTL_SECONDS": str(max_ttl)})
    assert result["raised"] is False
    assert result["SESSION_TTL_SECONDS"] == max_ttl


# --- import-isolation: session_config.py never requires SESSION_SECRET_KEY ---


def test_session_config_imports_without_any_web_secret():
    """session_config.py (minor finding #1) must be importable with NO
    SESSION_SECRET_KEY/WEB_COOKIE_SECURE/WEB_ENV set at all — it has no
    concept of a CSRF secret or cookie posture, unlike web_config.py."""
    result = _run_module_import("session_config", {})
    assert result["raised"] is False
