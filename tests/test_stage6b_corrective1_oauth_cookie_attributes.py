"""
Stage 6B independent-audit corrective pass #1, MINOR 4A — direct, isolated
Set-Cookie attribute proof for the OAuth-binding cookie's deletion, in
BOTH secure and insecure cookie postures. Mirrors
tests/test_stage6a_corrective1_cookie_cleanup.py's own technique exactly:
construct a plain `starlette.responses.Response()` directly and inspect
`response.raw_headers` — no live server, no TestClient, no database.
"""

from starlette.responses import Response

import web_config
from web.github_oauth import _clear_oauth_state_cookie, _oauth_state_cookie_name


def _set_cookie_headers(response: Response) -> list[str]:
    return [v.decode("latin-1") for k, v in response.raw_headers if k == b"set-cookie"]


def _is_expired(header: str) -> bool:
    return "Max-Age=0" in header or "expires" in header.lower()


def test_secure_posture_cookie_name_uses_the_host_prefix():
    assert _oauth_state_cookie_name(secure=True) == "__Host-github_oauth_state"


def test_insecure_posture_cookie_name_is_the_bare_name():
    assert _oauth_state_cookie_name(secure=False) == "github_oauth_state"


def test_secure_posture_clear_emits_correct_attributes(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    response = Response()
    _clear_oauth_state_cookie(response)
    headers = _set_cookie_headers(response)

    assert len(headers) == 1
    header = headers[0]
    assert header.split("=", 1)[0] == "__Host-github_oauth_state"
    assert _is_expired(header)
    assert "Secure" in header
    assert "HttpOnly" in header
    assert "SameSite=lax" in header
    assert "Path=/" in header


def test_insecure_posture_clear_emits_correct_attributes(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    response = Response()
    _clear_oauth_state_cookie(response)
    headers = _set_cookie_headers(response)

    assert len(headers) == 1
    header = headers[0]
    assert header.split("=", 1)[0] == "github_oauth_state"
    assert _is_expired(header)
    assert "Secure" not in header
    assert "HttpOnly" in header
    assert "SameSite=lax" in header
    assert "Path=/" in header


def test_secure_and_insecure_clears_use_different_cookie_names():
    """The `__Host-` pair and the bare pair are DIFFERENT cookie names —
    clearing one never accidentally clears the other (unlike
    web/cookies.py's clear_session_cookie(), which deliberately clears
    BOTH pairs; the OAuth-binding cookie has no equivalent cross-posture
    staleness concern since it is always set and cleared within the same
    single request/response cycle of one login attempt, never carried
    across a posture change the way a long-lived session cookie could
    be)."""
    secure_response = Response()
    insecure_response = Response()

    original = web_config.COOKIE_SECURE
    try:
        web_config.COOKIE_SECURE = True
        _clear_oauth_state_cookie(secure_response)
        web_config.COOKIE_SECURE = False
        _clear_oauth_state_cookie(insecure_response)
    finally:
        web_config.COOKIE_SECURE = original

    secure_name = _set_cookie_headers(secure_response)[0].split("=", 1)[0]
    insecure_name = _set_cookie_headers(insecure_response)[0].split("=", 1)[0]
    assert secure_name != insecure_name
