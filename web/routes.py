"""
HTTP routes for the web adapter (Stage 6A/6C). Thin — every route resolves
its user (if any) through web/dependencies.py's centralized dependencies
and delegates all real work to app/auth_session.py / app/identity.py /
app/telegram_link.py; no business logic or ad-hoc cookie/session handling
lives here.

GET routes never mutate session/application state (health, /api/me).
POST /api/logout, POST /api/link/telegram/start, and POST /api/unlink/github
are the state-changing authenticated routes in this stage and all require
both a valid session AND a valid CSRF proof.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import app.auth_session as auth_session
import app.identity as identity
import app.telegram_link as telegram_link
import telegram_link_config
from web.cookies import clear_session_cookie
from web.dependencies import get_current_user_id, get_session_token, require_csrf
from web.schemas import (
    CurrentUserResponse,
    HealthResponse,
    LinkTelegramStartResponse,
    UnlinkGithubResponse,
)

router = APIRouter()

# Applied to POST /api/link/telegram/start's success response (Section G:
# "Cache-Control: no-store") — the response body embeds a one-time raw
# bearer secret and must never be served from any cache.
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}

# Generic, safe rejection text for every unlink outcome that isn't a clean
# success — deliberately identical regardless of WHY the rejection
# happened (Section L: "generic safe response"; mirrors Section I's
# identical posture for every REJECTED_* redemption outcome).
_UNLINK_REJECTED_DETAIL = "GitHub account cannot be unlinked right now"


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
    telegram_linked = await identity.is_telegram_linked(user_id)
    return CurrentUserResponse(id=profile.id, created_at=profile.created_at, telegram_linked=telegram_linked)


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


@router.post(
    "/api/link/telegram/start",
    response_model=LinkTelegramStartResponse,
    dependencies=[Depends(require_csrf)],
)
async def link_telegram_start(
    response: Response, user_id: uuid.UUID = Depends(get_current_user_id)
) -> LinkTelegramStartResponse:
    """
    Stage 6C, Section G/H. Requires a valid session and a valid CSRF proof
    (like POST /api/logout above). Returns a generic, safe 503 if Telegram
    linking is unavailable (TELEGRAM_BOT_USERNAME missing/malformed —
    Section H) and a generic, safe 409 if this account has no CURRENT
    GitHub mapping to link from (Section G) — neither leaks which
    condition applied beyond its own distinct, expected status code.
    """
    response.headers.update(_NO_STORE_HEADERS)

    if telegram_link_config.TELEGRAM_BOT_USERNAME is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Telegram linking is unavailable"
        )

    result = await telegram_link.start_link(user_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="No active GitHub account to link"
        )
    return LinkTelegramStartResponse(deep_link=result.deep_link, expires_at=result.expires_at)


@router.post(
    "/api/unlink/github",
    response_model=UnlinkGithubResponse,
    dependencies=[Depends(require_csrf)],
)
async def unlink_github(
    response: Response, user_id: uuid.UUID = Depends(get_current_user_id)
) -> UnlinkGithubResponse:
    """
    Stage 6C, Section L. Requires a valid session and a valid CSRF proof.
    Clears both browser cookies on a successful unlink (either branch —
    the canonical user's own session(s) are always gone/revoked either
    way, see app/telegram_link.py's unlink_github() / db.telegram_link.
    unlink_github_sync() docstrings for the exact per-branch mutation) —
    but deliberately NOT on a rejection (Section L, branch 3: "preserve
    mapping, sessions, attempts, user, and data" — nothing changed
    server-side, so the browser's still-valid session cookie must not be
    discarded either).
    """
    outcome = await telegram_link.unlink_github(user_id)

    if outcome == telegram_link.UnlinkOutcome.REJECTED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_UNLINK_REJECTED_DETAIL)

    clear_session_cookie(response)
    return UnlinkGithubResponse(status="ok")
