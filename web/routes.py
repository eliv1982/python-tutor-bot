"""
HTTP routes for the web adapter (Stage 6A). Thin — every route resolves
its user (if any) through web/dependencies.py's centralized dependencies
and delegates all real work to app/auth_session.py; no business logic or
ad-hoc cookie/session handling lives here.

GET routes never mutate session/application state (health, /api/me).
POST /api/logout is the only state-changing authenticated route in this
stage and requires both a valid session AND a valid CSRF proof.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import app.auth_session as auth_session
from web.cookies import clear_session_cookie
from web.dependencies import get_current_user_id, get_session_token, require_csrf
from web.schemas import CurrentUserResponse, HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness/readiness probe — no authentication, no database access."""
    return HealthResponse(status="ok")


@router.get("/api/me", response_model=CurrentUserResponse)
async def get_me(user_id: uuid.UUID = Depends(get_current_user_id)) -> CurrentUserResponse:
    profile = await auth_session.get_user_profile(user_id)
    if profile is None:
        # Defensive only: a resolved, currently-valid session whose user
        # row has vanished is an internal inconsistency, not the caller's
        # fault — reported as a generic 401 (fail closed) rather than
        # leaking that the row itself is the problem.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return CurrentUserResponse(id=profile.id, created_at=profile.created_at)


@router.post("/api/logout", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_csrf)])
async def logout(
    response: Response,
    raw_token: str = Depends(get_session_token),
    _user_id: uuid.UUID = Depends(get_current_user_id),
) -> None:
    """Revokes the server-side session FIRST, then clears both browser
    cookies — logout invalidates real server state, not merely the
    cookie (see app/auth_session.revoke_session()). Requires a currently
    VALID session (like any other protected route, via
    get_current_user_id — `_user_id` is unused, its only purpose is the
    401 gate) in addition to a matching CSRF proof; an already-invalid
    session has nothing further to revoke and must not report success."""
    await auth_session.revoke_session(raw_token)
    clear_session_cookie(response)
