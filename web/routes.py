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
with a server-fixed mode=BotMode.TEXT, with an explicit client-supplied
history — never app.session.user_sessions, never a Telegram handler.
Application exceptions are mapped to fixed public details only; an
unexpected exception is left to propagate as a 500.

Stage 7A-3: the web-deferred RAG/document surface above has arrived —
POST/GET/DELETE /api/documents(/{id}) and POST /api/retrieval/search
delegate to app/documents.py and app/retrieval.py, which themselves reach
rag/index.py + rag/query.py (Qdrant) and db/documents.py (the PostgreSQL
catalog). Importing this module therefore now transitively imports the
whole RAG/Qdrant stack (even for a request that only hits /api/chat) —
still never anything Telegram-specific (app.session, app.tutor, handlers,
telebot, telegram_config), which remains a hard boundary.
"""

import uuid
from pathlib import Path
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status

import app.auth_session as auth_session
import app.documents as app_documents
import app.identity as identity
import app.preferences as preferences
import app.retrieval as retrieval
import app.telegram_link as telegram_link
import app.text_chat as text_chat
import telegram_link_config
from config import BotMode
from rag.loader import SUPPORTED_EXTENSIONS
from web.cookies import clear_session_cookie
from web.dependencies import get_current_user_id, get_session_token, require_csrf
from web.schemas import (
    ChatRequest,
    ChatResponse,
    CurrentUserResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    HealthResponse,
    LinkTelegramStartResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResultItem,
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

# Stage 7A-3: fixed public details for documents/retrieval — same posture,
# never exception text/attributes.
_DOCUMENT_NOT_FOUND_DETAIL = "Document not found"
_UNSUPPORTED_FILE_TYPE_DETAIL = "Unsupported file type"
_FILE_TOO_LARGE_DETAIL = "File too large"
_DOCUMENT_PROCESSING_FAILED_DETAIL = "Document processing failed"
_DOCUMENT_DELETION_FAILED_DETAIL = "Document deletion failed"
_KNOWLEDGE_BASE_UNAVAILABLE_DETAIL = "Knowledge base unavailable"

_MAX_DISPLAY_NAME_LENGTH = 255
_DEFAULT_LIST_LIMIT = 20
_MAX_LIST_LIMIT = 100


class _InvalidUploadFilename(Exception):
    """`UploadFile.filename` missing/empty, path-only, or over the display-
    name length bound — maps to 422 INVALID_REQUEST_DETAIL."""


class _UnsupportedUploadExtension(Exception):
    """A syntactically valid filename whose extension isn't supported —
    maps to 422 _UNSUPPORTED_FILE_TYPE_DETAIL (a distinct detail from the
    one above, so the two 422s stay distinguishable to a client)."""


def _parse_upload_filename(filename: Optional[str]) -> Tuple[str, str]:
    """
    Derive (display_name, extension) from a raw, client-supplied
    `UploadFile.filename` (Stage 7A-3) — internal storage stays opaque
    UUID-based regardless (see app.documents._store_document_exclusively());
    this only decides the safe, user-facing display name and which
    extension gate to apply. Both `/` and `\\` are treated as path
    separators (Windows-authored filenames commonly arrive with the
    latter) — only the final leaf component is ever used, never a client-
    supplied directory. Unicode is preserved as-is: no case-folding or
    normalization beyond the case-insensitive extension check below.
    """
    if not filename:
        raise _InvalidUploadFilename()
    leaf = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not leaf:
        raise _InvalidUploadFilename()
    if len(leaf) > _MAX_DISPLAY_NAME_LENGTH:
        raise _InvalidUploadFilename()
    extension = Path(leaf).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise _UnsupportedUploadExtension()
    return leaf, extension


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
    return LinkTelegramStartResponse(
        deep_link=result.deep_link,
        bot_path=telegram_link_config.telegram_bot_path(),
        expires_at=result.expires_at,
    )


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
    """Stage 7A-2. Read-only — never creates a preference row. Reports the
    EFFECTIVE mode (Stage 7B-3P): the stored canonical mode, else the
    configured BOT_MODE default — the same resolution Telegram uses."""
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


def _to_document_summary_response(summary: "app_documents.DocumentSummary") -> DocumentSummaryResponse:
    return DocumentSummaryResponse(id=summary.id, display_name=summary.display_name, created_at=summary.created_at)


@router.post(
    "/api/documents",
    response_model=DocumentSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_csrf)],
)
async def upload_document(
    file: UploadFile = File(...), user_id: uuid.UUID = Depends(get_current_user_id)
) -> DocumentSummaryResponse:
    """Stage 7A-3. Reuses app.documents.ingest_document() unchanged — this
    is not a second upload transaction, only filename-policy parsing and
    result-code mapping around the existing one. Success is only ever
    reported once ingestion has reached 'active'."""
    try:
        display_name, extension = _parse_upload_filename(file.filename)
    except _InvalidUploadFilename:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=INVALID_REQUEST_DETAIL) from None
    except _UnsupportedUploadExtension:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=_UNSUPPORTED_FILE_TYPE_DETAIL
        ) from None

    file_bytes = await file.read()
    result = await app_documents.ingest_document(
        file_bytes=file_bytes, extension=extension, display_name=display_name, owner_user_id=user_id
    )
    if not result.success:
        if result.rejected_reason == "oversized":
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=_FILE_TOO_LARGE_DETAIL) from None
        if result.rejected_reason == "unsupported_extension":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=_UNSUPPORTED_FILE_TYPE_DETAIL
            ) from None
        if result.failure_reason == app_documents.INGEST_FAILURE_KNOWLEDGE_BASE_UNAVAILABLE:
            # Genuine Qdrant/index availability failure only (see
            # rag.index.is_index_unavailable_error()); every other
            # storage/parse/embedding/catalog failure stays the generic 500
            # below. Ingestion has already rolled back either way.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_KNOWLEDGE_BASE_UNAVAILABLE_DETAIL
            ) from None
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=_DOCUMENT_PROCESSING_FAILED_DETAIL
        ) from None

    summary = await app_documents.get_document(user_id, result.stored.document_uuid)
    if summary is None:
        # Not expected to be reachable (ingestion just committed this exact
        # row as 'active' under this exact owner) — fails closed rather
        # than ever fabricating a response from unverified local state.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=_DOCUMENT_PROCESSING_FAILED_DETAIL
        )
    return _to_document_summary_response(summary)


@router.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_LIST_LIMIT),
    offset: int = Query(default=0, ge=0),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> DocumentListResponse:
    """Stage 7A-3. Catalog only — never touches Qdrant/the filesystem.
    Only this session's own ACTIVE documents are ever returned."""
    summaries = await app_documents.list_documents(user_id, limit=limit, offset=offset)
    return DocumentListResponse(items=[_to_document_summary_response(s) for s in summaries])


@router.get("/api/documents/{document_id}", response_model=DocumentSummaryResponse)
async def get_document(document_id: uuid.UUID, user_id: uuid.UUID = Depends(get_current_user_id)) -> DocumentSummaryResponse:
    """Stage 7A-3. A missing id, a foreign owner, a 'pending' row, and a
    'deleting' row all produce the identical public 404 — see
    app.documents.get_document()'s own docstring."""
    summary = await app_documents.get_document(user_id, document_id)
    if summary is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DOCUMENT_NOT_FOUND_DETAIL)
    return _to_document_summary_response(summary)


@router.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_csrf)])
async def delete_document(document_id: uuid.UUID, user_id: uuid.UUID = Depends(get_current_user_id)) -> None:
    """Stage 7A-3. See app.documents.delete_document()'s own docstring for
    the full active -> deleting -> cleanup state machine and its concurrent-
    delete convergence semantics."""
    try:
        found = await app_documents.delete_document(user_id, document_id)
    except app_documents.KnowledgeBaseUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_KNOWLEDGE_BASE_UNAVAILABLE_DETAIL
        ) from None
    except app_documents.DocumentDeletionError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=_DOCUMENT_DELETION_FAILED_DETAIL
        ) from None
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DOCUMENT_NOT_FOUND_DETAIL)


@router.post("/api/retrieval/search", response_model=RetrievalResponse, dependencies=[Depends(require_csrf)])
async def search_retrieval(
    body: RetrievalRequest, user_id: uuid.UUID = Depends(get_current_user_id)
) -> RetrievalResponse:
    """Stage 7A-3. Raw validated similarity search only — no text
    generation is ever triggered from this route (see
    rag.query.search_documents())."""
    try:
        hits = await retrieval.search(owner_user_id=user_id, query=body.query, top_k=body.top_k)
    except retrieval.RetrievalValidationError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=INVALID_REQUEST_DETAIL) from None
    except retrieval.RetrievalUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_KNOWLEDGE_BASE_UNAVAILABLE_DETAIL
        ) from None
    return RetrievalResponse(
        results=[
            RetrievalResultItem(
                document_id=hit.document_id,
                source=hit.source,
                chunk_index=hit.chunk_index,
                page=hit.page,
                content=hit.content,
            )
            for hit in hits
        ]
    )
