"""
Stage 6A independent-audit corrective pass #1 — Major 3 real Set-Cookie
header regression proof: clear_session_cookie() must not strand a
previously-issued cookie merely because WEB_COOKIE_SECURE has since
changed.

Pre-fix reproduction: a session issued while COOKIE_SECURE=True sets
`__Host-session`/`__Host-csrf_token`. If WEB_COOKIE_SECURE later flips to
false, the old clear_session_cookie() cleared ONLY `session`/`csrf_token`
(cleared_names=['session', 'csrf_token']) — the actual browser cookies
(`__Host-session`/`__Host-csrf_token`) were never touched, i.e. stranded.

Fix: clear_session_cookie() now clears BOTH the current-posture pair AND
the other posture's fixed pair (web/cookies.py's own docstring covers the
browser-compatibility reasoning for the `__Host-` half specifically).

Every test here inspects the actual emitted Set-Cookie header strings —
exact cookie names and the Secure/HttpOnly/SameSite/Path/expiration
attributes — never merely a header count.
"""

from starlette.responses import Response

import web_config
from web.cookies import clear_session_cookie


def _set_cookie_headers(response: Response) -> list[str]:
    return [v.decode("latin-1") for k, v in response.raw_headers if k == b"set-cookie"]


def _by_name(headers: list[str], name: str) -> str:
    match = next((h for h in headers if h.split("=", 1)[0] == name), None)
    assert match is not None, f"no Set-Cookie header found for {name!r} among {headers!r}"
    return match


def _is_expired(header: str) -> bool:
    return "Max-Age=0" in header or "expires" in header.lower()


# --- 1: secure session issued -> secure logout ------------------------------


def test_secure_logout_clears_the_secure_names_with_correct_attributes(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    response = Response()
    clear_session_cookie(response)
    headers = _set_cookie_headers(response)

    session_header = _by_name(headers, "__Host-session")
    assert _is_expired(session_header)
    assert "Secure" in session_header
    assert "HttpOnly" in session_header
    assert "SameSite=lax" in session_header
    assert "Path=/" in session_header

    csrf_header = _by_name(headers, "__Host-csrf_token")
    assert _is_expired(csrf_header)
    assert "Secure" in csrf_header
    assert "HttpOnly" not in csrf_header
    assert "SameSite=lax" in csrf_header
    assert "Path=/" in csrf_header


# --- 2: insecure session issued -> insecure logout ---------------------------


def test_insecure_logout_clears_the_bare_names_with_correct_attributes(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    response = Response()
    clear_session_cookie(response)
    headers = _set_cookie_headers(response)

    session_header = _by_name(headers, "session")
    assert _is_expired(session_header)
    assert "Secure" not in session_header
    assert "HttpOnly" in session_header
    assert "SameSite=lax" in session_header
    assert "Path=/" in session_header

    csrf_header = _by_name(headers, "csrf_token")
    assert _is_expired(csrf_header)
    assert "Secure" not in csrf_header
    assert "HttpOnly" not in csrf_header
    assert "SameSite=lax" in csrf_header
    assert "Path=/" in csrf_header


# --- 3: secure names are ALSO cleared while running in insecure mode --------


def test_insecure_logout_also_attempts_to_clear_the_secure_names(monkeypatch):
    """Defense against a session that was issued back when COOKIE_SECURE
    was True, now being logged out after a config flip to False — the
    `__Host-` pair must still be attempted (with Secure=True, the only
    attribute combination a browser will ever accept for that name prefix)
    even though the CURRENT posture is insecure."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    response = Response()
    clear_session_cookie(response)
    headers = _set_cookie_headers(response)

    session_header = _by_name(headers, "__Host-session")
    assert _is_expired(session_header)
    assert "Secure" in session_header
    assert "Path=/" in session_header

    csrf_header = _by_name(headers, "__Host-csrf_token")
    assert _is_expired(csrf_header)
    assert "Secure" in csrf_header


# --- 4: bare/insecure legacy names are ALSO cleared while running secure ---


def test_secure_logout_also_clears_the_bare_insecure_names(monkeypatch):
    """The reverse direction: a session issued while COOKIE_SECURE was
    False (e.g. local dev, or a temporary misconfiguration) must not be
    left stranded once the deployment is back to secure/production."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", True)
    response = Response()
    clear_session_cookie(response)
    headers = _set_cookie_headers(response)

    session_header = _by_name(headers, "session")
    assert _is_expired(session_header)
    assert "Path=/" in session_header

    csrf_header = _by_name(headers, "csrf_token")
    assert _is_expired(csrf_header)


# --- exactly four headers, no unexpected fifth name -------------------------


def test_clear_session_cookie_emits_exactly_four_headers_covering_both_postures(monkeypatch):
    for secure in (True, False):
        monkeypatch.setattr(web_config, "COOKIE_SECURE", secure)
        response = Response()
        clear_session_cookie(response)
        headers = _set_cookie_headers(response)
        names = sorted(h.split("=", 1)[0] for h in headers)
        assert names == ["__Host-csrf_token", "__Host-session", "csrf_token", "session"]
