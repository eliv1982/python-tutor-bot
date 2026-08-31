"""
Stage 6A regression tests: cookie construction (web/cookies.py) and the
CSRF double-submit derivation (web/csrf.py). No database needed — these
are pure functions over a Starlette Response object / plain strings.
"""

from datetime import datetime, timedelta, timezone

import pytest
from starlette.responses import Response

import web_config
from web.cookies import clear_session_cookie, set_session_cookie
from web.csrf import csrf_token_matches, derive_csrf_token


def _set_cookie_headers(response: Response) -> list[str]:
    return [v.decode("latin-1") for k, v in response.raw_headers if k == b"set-cookie"]


def test_derive_csrf_token_is_deterministic_for_the_same_input():
    token = "same-raw-session-token"
    assert derive_csrf_token(token) == derive_csrf_token(token)


def test_derive_csrf_token_differs_for_different_sessions():
    assert derive_csrf_token("session-a") != derive_csrf_token("session-b")


def test_csrf_token_matches_accepts_the_correct_derivation():
    raw = "a-raw-session-token"
    assert csrf_token_matches(raw_session_token=raw, submitted_token=derive_csrf_token(raw))


def test_csrf_token_matches_rejects_a_forged_value():
    raw = "a-raw-session-token"
    assert not csrf_token_matches(raw_session_token=raw, submitted_token="attacker-guessed-value")


def test_csrf_token_depends_on_the_secret_key(monkeypatch):
    raw = "a-raw-session-token"
    original = derive_csrf_token(raw)
    monkeypatch.setattr(web_config, "SESSION_SECRET_KEY", "a-completely-different-secret")
    assert derive_csrf_token(raw) != original


def test_set_session_cookie_sets_httponly_secure_samesite_and_path(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    response = Response()
    expires_at = datetime.now(timezone.utc) + timedelta(days=1)

    set_session_cookie(response, raw_token="raw-bearer-token-value", expires_at=expires_at)

    headers = _set_cookie_headers(response)
    assert len(headers) == 2

    session_header = next(h for h in headers if h.startswith("__Host-session="))
    assert "raw-bearer-token-value" in session_header
    assert "HttpOnly" in session_header
    assert "Secure" in session_header
    assert "SameSite=lax" in session_header
    assert "Path=/" in session_header

    csrf_header = next(h for h in headers if h.startswith("__Host-csrf_token="))
    assert "HttpOnly" not in csrf_header
    assert "Secure" in csrf_header
    assert "SameSite=lax" in csrf_header
    assert "Path=/" in csrf_header


def test_session_cookie_value_is_only_the_opaque_bearer_token(monkeypatch):
    """No canonical user UUID or other sensitive data may ever be encoded
    into the cookie — cookies.py never even receives a user id, only the
    raw bearer token, so this is a structural guarantee, verified here by
    asserting the emitted cookie carries EXACTLY the opaque value passed
    in and nothing else appended."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    response = Response()
    raw_token = "opaque-bearer-token-no-pii"
    set_session_cookie(response, raw_token=raw_token, expires_at=datetime.now(timezone.utc) + timedelta(days=1))

    headers = _set_cookie_headers(response)
    session_header = next(h for h in headers if h.startswith("__Host-session="))
    cookie_value = session_header.split(";")[0].split("=", 1)[1]
    assert cookie_value == raw_token


def test_cookie_names_use_host_prefix_only_when_secure(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    assert web_config.session_cookie_name() == "__Host-session"
    assert web_config.csrf_cookie_name() == "__Host-csrf_token"

    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    assert web_config.session_cookie_name() == "session"
    assert web_config.csrf_cookie_name() == "csrf_token"


def test_set_session_cookie_omits_secure_flag_when_disabled(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    response = Response()
    set_session_cookie(response, raw_token="tok", expires_at=datetime.now(timezone.utc) + timedelta(days=1))

    headers = _set_cookie_headers(response)
    session_header = next(h for h in headers if h.startswith("session="))
    assert "Secure" not in session_header


def test_clear_session_cookie_expires_both_cookies_immediately(monkeypatch):
    """Independent-audit corrective pass #1, Major 3: clear_session_cookie()
    now clears FOUR Set-Cookie headers, not two — the current-posture pair
    AND the other-posture's fixed pair (see web/cookies.py's own
    docstring). The pre-fix version only ever cleared two (whichever
    matched CURRENT WEB_COOKIE_SECURE), which is exactly the "stranded
    cookie" defect the audit reproduced — see
    tests/test_stage6a_corrective1_cookie_cleanup.py for the full
    exact-name/attribute proof across all four secure/insecure
    combinations; this test only re-confirms the basic "every emitted
    Set-Cookie actually expires immediately" property."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    response = Response()
    clear_session_cookie(response)

    headers = _set_cookie_headers(response)
    assert len(headers) == 4
    for header in headers:
        assert 'Max-Age=0' in header or "expires" in header.lower()
