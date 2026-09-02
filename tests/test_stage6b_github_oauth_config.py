"""
Stage 6B configuration-hardening regression tests for github_oauth_config.py
— all import-time validation, so every scenario runs in an isolated
subprocess, the exact same technique
tests/test_stage6a_corrective1_web_config.py already established for
web_config.py/session_config.py (itself borrowed from
tests/test_stage2a_text_llm_provider.py's _CONFIG_IMPORT_SCRIPT). This
session's own already-imported github_oauth_config module is never
disturbed, and a developer's real .env (regardless of its contents) can
never influence the result (dotenv.load_dotenv is stubbed to a no-op
before the module is imported).

github_oauth_config.py imports web_config.py (for WEB_ENV), which itself
fails closed without a valid SESSION_SECRET_KEY — every scenario below
therefore always supplies one, unless a test is specifically exercising
that upstream dependency.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_VALID_SECRET = "a" * 43
_VALID_CLIENT_ID = "test-client-id"
_VALID_CLIENT_SECRET = "test-client-secret-value"
_VALID_HTTPS_REDIRECT = "https://example.com/api/auth/github/callback"
_VALID_DEV_REDIRECT = "http://127.0.0.1:8000/api/auth/github/callback"

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
    for attr in (
        "GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET", "GITHUB_REDIRECT_URI",
        "OAUTH_TRANSACTION_TTL_SECONDS", "GITHUB_AUTHORIZE_URL", "GITHUB_TOKEN_URL",
        "GITHUB_USER_API_URL",
    ):
        if hasattr(module, attr):
            result[attr] = getattr(module, attr)
    print("IMPORT_RESULT=" + json.dumps(result))
    """
)


def _base_subprocess_env() -> dict:
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


def _valid_env(**overrides) -> dict:
    env = {
        "SESSION_SECRET_KEY": _VALID_SECRET,
        "GITHUB_CLIENT_ID": _VALID_CLIENT_ID,
        "GITHUB_CLIENT_SECRET": _VALID_CLIENT_SECRET,
        "GITHUB_REDIRECT_URI": _VALID_HTTPS_REDIRECT,
    }
    env.update(overrides)
    return env


# --- GITHUB_CLIENT_ID --------------------------------------------------------


def test_missing_client_id_fails_closed():
    env = _valid_env()
    del env["GITHUB_CLIENT_ID"]
    result = _run_module_import("github_oauth_config", env)
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_empty_client_id_fails_closed():
    result = _run_module_import("github_oauth_config", _valid_env(GITHUB_CLIENT_ID=""))
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_whitespace_containing_client_id_fails_closed():
    result = _run_module_import("github_oauth_config", _valid_env(GITHUB_CLIENT_ID="abc def"))
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_valid_client_id_is_accepted():
    result = _run_module_import("github_oauth_config", _valid_env())
    assert result["raised"] is False
    assert result["GITHUB_CLIENT_ID"] == _VALID_CLIENT_ID


# --- GITHUB_CLIENT_SECRET -----------------------------------------------------


def test_missing_client_secret_fails_closed():
    env = _valid_env()
    del env["GITHUB_CLIENT_SECRET"]
    result = _run_module_import("github_oauth_config", env)
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_empty_client_secret_fails_closed():
    result = _run_module_import("github_oauth_config", _valid_env(GITHUB_CLIENT_SECRET=""))
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_whitespace_only_client_secret_fails_closed():
    result = _run_module_import("github_oauth_config", _valid_env(GITHUB_CLIENT_SECRET="   "))
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_client_secret_is_never_echoed_in_the_error_message_of_an_unrelated_failure():
    """Even when SOME other field fails, the secret itself (if supplied)
    must never appear in the raised error's message."""
    env = _valid_env(GITHUB_REDIRECT_URI="ftp://not-allowed.example/callback")
    result = _run_module_import("github_oauth_config", env)
    assert result["raised"] is True
    assert _VALID_CLIENT_SECRET not in result["error_message"]


# --- GITHUB_REDIRECT_URI ------------------------------------------------------


def test_missing_redirect_uri_fails_closed():
    env = _valid_env()
    del env["GITHUB_REDIRECT_URI"]
    result = _run_module_import("github_oauth_config", env)
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_javascript_scheme_redirect_uri_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_REDIRECT_URI="javascript:alert(1)")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_data_scheme_redirect_uri_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_REDIRECT_URI="data:text/html,hi")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_https_redirect_uri_is_accepted_in_production():
    result = _run_module_import("github_oauth_config", _valid_env())
    assert result["raised"] is False
    assert result["GITHUB_REDIRECT_URI"] == _VALID_HTTPS_REDIRECT


def test_http_redirect_uri_is_rejected_in_production():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_REDIRECT_URI=_VALID_DEV_REDIRECT)
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_http_redirect_uri_is_rejected_even_with_explicit_production_web_env():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI=_VALID_DEV_REDIRECT, WEB_ENV="production"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_http_loopback_redirect_uri_is_accepted_in_development():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI=_VALID_DEV_REDIRECT, WEB_ENV="development"),
    )
    assert result["raised"] is False
    assert result["GITHUB_REDIRECT_URI"] == _VALID_DEV_REDIRECT


def test_http_localhost_redirect_uri_is_rejected_even_in_development():
    """Stage 6B independent-audit corrective pass #1, MINOR 3: `localhost`
    is deliberately NOT accepted, even though it resolves to loopback —
    only the literal 127.0.0.1/::1 spellings are, matching this
    repository's own documented manual-testing callback exactly."""
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(
            GITHUB_REDIRECT_URI="http://localhost:8000/api/auth/github/callback",
            WEB_ENV="development",
        ),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_http_ipv6_loopback_bracket_syntax_redirect_uri_is_accepted_in_development():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(
            GITHUB_REDIRECT_URI="http://[::1]:8000/api/auth/github/callback",
            WEB_ENV="development",
        ),
    )
    assert result["raised"] is False


# --- MINOR 3: userinfo / fragment / query / port / path hardening -----------


def test_redirect_uri_with_userinfo_is_rejected():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://user:pass@example.com/api/auth/github/callback"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_with_username_only_is_rejected():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://attacker@example.com/api/auth/github/callback"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_with_fragment_is_rejected():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://example.com/api/auth/github/callback#fragment"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_with_query_is_rejected():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://example.com/api/auth/github/callback?foo=bar"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_with_invalid_port_is_rejected_safely():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://example.com:999999/api/auth/github/callback"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_with_non_numeric_port_is_rejected_safely():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://example.com:abc/api/auth/github/callback"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_with_wrong_path_is_rejected():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://example.com/some/other/path"),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_with_valid_explicit_port_is_accepted():
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(GITHUB_REDIRECT_URI="https://example.com:8443/api/auth/github/callback"),
    )
    assert result["raised"] is False


def test_http_non_loopback_redirect_uri_is_rejected_even_in_development():
    """WEB_COOKIE_SECURE=false-style relaxation is scoped to loopback
    only — an arbitrary non-loopback http:// host must never be accepted
    merely because WEB_ENV=development."""
    result = _run_module_import(
        "github_oauth_config",
        _valid_env(
            GITHUB_REDIRECT_URI="http://example.com/api/auth/github/callback",
            WEB_ENV="development",
        ),
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_missing_host_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_REDIRECT_URI="https:///api/auth/github/callback")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_redirect_uri_missing_path_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_REDIRECT_URI="https://example.com")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


# --- GITHUB_OAUTH_TRANSACTION_TTL_SECONDS -------------------------------------


def test_ttl_default_is_ten_minutes():
    result = _run_module_import("github_oauth_config", _valid_env())
    assert result["raised"] is False
    assert result["OAUTH_TRANSACTION_TTL_SECONDS"] == 600


def test_ttl_zero_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_OAUTH_TRANSACTION_TTL_SECONDS="0")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_negative_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_OAUTH_TRANSACTION_TTL_SECONDS="-5")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_non_integer_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_OAUTH_TRANSACTION_TTL_SECONDS="soon")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_excessively_large_is_rejected():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_OAUTH_TRANSACTION_TTL_SECONDS="999999")
    )
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_ttl_valid_custom_value_is_accepted():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_OAUTH_TRANSACTION_TTL_SECONDS="300")
    )
    assert result["raised"] is False
    assert result["OAUTH_TRANSACTION_TTL_SECONDS"] == 300


def test_ttl_at_maximum_boundary_is_accepted():
    result = _run_module_import(
        "github_oauth_config", _valid_env(GITHUB_OAUTH_TRANSACTION_TTL_SECONDS="900")
    )
    assert result["raised"] is False
    assert result["OAUTH_TRANSACTION_TTL_SECONDS"] == 900


# --- fixed GitHub endpoint constants ------------------------------------------


def test_github_endpoint_constants_are_the_expected_official_urls():
    result = _run_module_import("github_oauth_config", _valid_env())
    assert result["raised"] is False
    assert result["GITHUB_AUTHORIZE_URL"] == "https://github.com/login/oauth/authorize"
    assert result["GITHUB_TOKEN_URL"] == "https://github.com/login/oauth/access_token"
    assert result["GITHUB_USER_API_URL"] == "https://api.github.com/user"
