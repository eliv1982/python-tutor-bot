"""
Centralized FastAPI authentication/CSRF dependencies (Stage 6A).

Every protected route in web/routes.py declares these instead of reading
request.cookies/headers itself — "no ad-hoc cookie/session lookups" is a
hard requirement (see this stage's spec, Section 5), so this is the ONLY
place in the web adapter that reads the session cookie or resolves it to a
canonical user.

Semantics (all fail closed — see get_current_user_id() below):
  - no session cookie                                    -> 401
  - malformed / unknown / expired / revoked / wrong-posture -> 401
  - valid session for the CURRENT cookie posture          -> canonical uuid.UUID

get_current_user_id() passes web_config.COOKIE_SECURE to
app/auth_session.resolve_session_user_id() as `expected_secure` (Stage 6A
independent-audit corrective pass #2, Major 1) — a session created while
COOKIE_SECURE was the OTHER value must never authenticate here, in either
direction. This is the one deliberate exception to app/auth_session.py's
own "never import web_config" rule (see that module's docstring): the
posture value itself must come from HERE, the web adapter boundary, not
be re-derived inside the posture-agnostic application layer.

No user id supplied by the browser (query param, header, body) is ever
trusted — the ONLY input is the opaque session cookie, resolved entirely
server-side through app/auth_session.py.
"""

import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

import app.auth_session as auth_session
import web_config
from web.csrf import csrf_token_matches

CSRF_HEADER_NAME = "X-CSRF-Token"

_UNAUTHENTICATED = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


async def get_session_token(request: Request) -> str:
    """Raw bearer token straight from the cookie — never logged, never
    echoed back in any response. Raises 401 immediately for a missing or
    empty cookie value, before any database lookup is attempted."""
    raw_token = request.cookies.get(web_config.session_cookie_name())
    if not raw_token:
        raise _UNAUTHENTICATED
    return raw_token


async def get_current_user_id(raw_token: str = Depends(get_session_token)) -> uuid.UUID:
    """The one centralized current-user dependency. A malformed, unknown,
    expired, revoked, or wrong-posture token is indistinguishable here by
    design (see app/auth_session.resolve_session_user_id()'s own contract)
    — all of them fail the same way: 401."""
    user_id = await auth_session.resolve_session_user_id(raw_token, expected_secure=web_config.COOKIE_SECURE)
    if user_id is None:
        raise _UNAUTHENTICATED
    return user_id


async def require_csrf(request: Request, raw_token: str = Depends(get_session_token)) -> None:
    """Applied to every state-changing authenticated route (never to a
    safe GET/HEAD route, which must not mutate anything in the first
    place). Runs only after get_session_token() has already confirmed a
    session cookie is present; a missing/mismatched X-CSRF-Token header is
    a 403, not a 401 — the caller IS carrying a session cookie, it simply
    failed the CSRF proof, a materially different failure than "not
    authenticated" for a client to distinguish."""
    submitted = request.headers.get(CSRF_HEADER_NAME)
    if not submitted or not csrf_token_matches(raw_session_token=raw_token, submitted_token=submitted):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed")
