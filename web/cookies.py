"""
Browser cookie construction for the web adapter's server-side session
(Stage 6A).

Two cookies are ever set, both `Path=/`, `SameSite=Lax`, and `Secure`
whenever web_config.COOKIE_SECURE is true (the fail-safe default —
disabled only for explicit local plain-HTTP development):

- The SESSION cookie (web_config.session_cookie_name()) carries ONLY the
  opaque, high-entropy bearer token from app/auth_session.py — no PII, no
  encoded user id. `HttpOnly=True`: browser-side JavaScript can never read
  it, closing off theft via XSS.
- The CSRF cookie (web_config.csrf_cookie_name()) carries the token
  derived from it (web/csrf.py) and is deliberately `HttpOnly=False` —
  same-origin JavaScript MUST be able to read it to echo it back as a
  request header; that read/echo requirement is exactly what makes the
  double-submit comparison a CSRF defense (see web/csrf.py's own
  docstring).

Cookie NAMES/`secure` are read fresh from web_config at call time (never
bound once at import), so a test that monkeypatches web_config.
COOKIE_SECURE sees this module react immediately — same convention
db/settings.py's DATABASE_URL and rag/constants.py's DATA_DIR already use.

Neither cookie is ever logged: callers only ever pass values into these
functions, never receive them back for logging.

clear_session_cookie() additionally clears the OTHER posture's fixed
cookie names too (independent-audit corrective pass #1, Major 3): if
WEB_COOKIE_SECURE changes between when a session cookie was set and when
it is logged out, the browser is still holding a cookie under the OLD
posture's name (e.g. `__Host-session`) which the CURRENT posture's
name-only clear (e.g. `session`) would never touch, leaving it physically
stranded in the browser indefinitely. See web_config.session_cookie_name()/
csrf_cookie_name()'s own docstrings for the `secure` parameter this relies
on, and tests/test_stage6a_corrective1_cookie_cleanup.py for the exact
Set-Cookie header proof across all four secure/insecure combinations.
Clearing the `__Host-` pair still requires Secure=True on the deleting
Set-Cookie header no matter which posture is currently configured — that
is the browser's own unavoidable enforcement of the `__Host-` prefix
(a real plain-HTTP connection will simply not accept it), not a gap in
this function; the bare pair carries no such restriction and is always
safe to clear. Server-side revocation (app/auth_session.revoke_session(),
already called before this in web/routes.py's logout()) remains the
authoritative invalidation — this is defense-in-depth/UX cleanup on top of
that, not a substitute for it.
"""

from datetime import datetime, timezone

from starlette.responses import Response

import web_config
from web.csrf import derive_csrf_token

_SAMESITE = "lax"


def set_session_cookie(response: Response, *, raw_token: str, expires_at: datetime) -> None:
    max_age = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        key=web_config.session_cookie_name(),
        value=raw_token,
        max_age=max_age,
        expires=expires_at,
        path="/",
        httponly=True,
        secure=web_config.COOKIE_SECURE,
        samesite=_SAMESITE,
    )
    response.set_cookie(
        key=web_config.csrf_cookie_name(),
        value=derive_csrf_token(raw_token),
        max_age=max_age,
        expires=expires_at,
        path="/",
        httponly=False,
        secure=web_config.COOKIE_SECURE,
        samesite=_SAMESITE,
    )


def _delete_cookie_pair(response: Response, *, secure: bool) -> None:
    response.delete_cookie(
        key=web_config.session_cookie_name(secure),
        path="/",
        secure=secure,
        httponly=True,
        samesite=_SAMESITE,
    )
    response.delete_cookie(
        key=web_config.csrf_cookie_name(secure),
        path="/",
        secure=secure,
        httponly=False,
        samesite=_SAMESITE,
    )


def clear_session_cookie(response: Response) -> None:
    """Used on logout, AFTER the server-side session has already been
    revoked (see app/auth_session.py's revoke_session()) — this alone is
    never the mechanism that invalidates the session, only cosmetic
    browser-side cleanup.

    Clears BOTH the current-posture pair AND the other-posture's fixed pair
    (see this module's own docstring, Major 3) — never just the one
    matching today's WEB_COOKIE_SECURE."""
    _delete_cookie_pair(response, secure=web_config.COOKIE_SECURE)
    _delete_cookie_pair(response, secure=not web_config.COOKIE_SECURE)
