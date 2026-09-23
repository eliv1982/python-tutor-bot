"""
HTTP routes for the web adapter (Stage 6A/6C). Thin — every route resolves
its user (if any) through web/dependencies.py's centralized dependencies
and delegates all real work to app/auth_session.py / app/identity.py /
app/telegram_link.py; no business logic or ad-hoc cookie/session handling
lives here.

GET routes never mutate session/application state (health, /api/me,
/api/settings). POST /api/logout, POST /api/link/telegram/start,
POST /api/unlink/github, POST /api/chat, and PATCH /api/settings are the
state-changing (or generation-triggering) authenticated routes and all
require both a valid session AND a valid CSRF proof.

Stage 7A-2: POST /api/chat calls app/text_chat.py's run_text_chat() directly
with a server-fixed mode=BotMode.TEXT (web retrieval/RAG is deferred), with
an explicit client-supplied history — never app.session.user_sessions,
never a Telegram handler. Application exceptions are mapped to fixed public
details only; an unexpected exception is left to propagate as a 500.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import app.auth_session as auth_session
import app.identity as identity
import app.preferences as preferences
import app.telegram_link as telegram_link
import app.text_chat as text_chat
import telegram_link_config
from config import BotMode
from web.cookies import clear_session_cookie
from web.dependencies import get_current_user_id, get_session_token, require_csrf
from web.schemas import (
    ChatRequest,
    ChatResponse,
    CurrentUserResponse,
    HealthResponse,
    LinkTelegramStartResponse,
    SettingsResponse,
    SettingsUpdateRequest,
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

# Stage 7A-2: fixed public details — never exception text/attributes.
INVALID_REQUEST_DETAIL = "Invalid request"
_GENERATION_BUSY_DETAIL = "Generation is busy, try again shortly"
_GENERATION_TIMEOUT_DETAIL = "Generation timed out"
_GENERATION_FAILED_DETAIL = "Generation failed"


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


@router.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_csrf)])
async def chat(
    body: ChatRequest, response: Response, user_id: uuid.UUID = Depends(get_current_user_id)
) -> ChatResponse:
    """Stage 7A-2. Text-only, stateless: the reply is returned and nothing
    is persisted — the client owns and resends its own bounded history."""
    response.headers.update(_NO_STORE_HEADERS)
    history = [{"role": entry.role, "content": entry.content} for entry in body.history]

    try:
        result = await text_chat.run_text_chat(
            user_id=user_id, message=body.message, history=history, mode=BotMode.TEXT
        )
    except text_chat.TextChatValidationError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=INVALID_REQUEST_DETAIL,
            headers=_NO_STORE_HEADERS,
        ) from None
    except text_chat.GenerationBusyError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=_GENERATION_BUSY_DETAIL,
            headers=_NO_STORE_HEADERS,
        ) from None
    except text_chat.TextChatTimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=_GENERATION_TIMEOUT_DETAIL,
            headers=_NO_STORE_HEADERS,
        ) from None
    except text_chat.TextChatGenerationError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_GENERATION_FAILED_DETAIL,
            headers=_NO_STORE_HEADERS,
        ) from None

    return ChatResponse(text=result.text)


@router.get("/api/settings", response_model=SettingsResponse)
async def get_settings(user_id: uuid.UUID = Depends(get_current_user_id)) -> SettingsResponse:
    """Stage 7A-2. Read-only — never creates a preference row."""
    mode = await preferences.get_effective_mode(user_id)
    return SettingsResponse(mode=mode)


@router.patch("/api/settings", response_model=SettingsResponse, dependencies=[Depends(require_csrf)])
async def update_settings(
    body: SettingsUpdateRequest, user_id: uuid.UUID = Depends(get_current_user_id)
) -> SettingsResponse:
    """Stage 7A-2. Only the authenticated session's own preference row is
    ever written — no client-supplied identity exists in the request."""
    try:
        mode = await preferences.set_mode(user_id, body.mode)
    except preferences.PreferenceValidationError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=INVALID_REQUEST_DETAIL
        ) from None
    return SettingsResponse(mode=mode)
