"""Request/response models for the web adapter (Stage 6A/6C/7A-2) —
deliberately minimal: only fields already proven safe to expose. Never a
Telegram id, GitHub numeric id, provider credential, or other internal
persistence detail (Section M: "Do not expose Telegram numeric id or GitHub
numeric id").

Stage 7A-2 request models are strict (no type coercion) and reject unknown
fields. They validate only JSON shape/types; the agreed text/history size
limits and the mode allowlist stay in the application layer
(app/text_chat.py, app/preferences.py) as the single source of truth."""

import uuid
from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class CurrentUserResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    # Stage 6C, Section M: a safe boolean only — never the Telegram numeric
    # id itself. Resolved via a database existence query on the
    # authenticated UUID (db.identity.has_telegram_account_sync()).
    telegram_linked: bool


class HealthResponse(BaseModel):
    status: str


class LinkTelegramStartResponse(BaseModel):
    """Stage 6C, Section G — the deep link embeds the one-time raw bearer
    secret; this response model exists only so FastAPI can validate/
    serialize it, never so it gets logged/cached (see web/routes.py's
    `Cache-Control: no-store` on this route)."""

    deep_link: str
    expires_at: datetime


class UnlinkGithubResponse(BaseModel):
    """Stage 6C, Section L — deliberately just a fixed, generic status
    string: the three possible outcomes (kept-with-Telegram, deleted,
    rejected) must never be distinguishable from a successful response's
    shape alone beyond "did this succeed or not" (a rejection is a
    different HTTP status, not a different body shape)."""

    status: str


class ChatHistoryMessage(BaseModel):
    """Stage 7A-2 — one prior turn. `system` is never a client-suppliable
    role."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Stage 7A-2 — no `mode` and no user id: web chat is fixed to text mode
    server-side, and identity comes only from the session cookie."""

    model_config = ConfigDict(extra="forbid", strict=True)

    message: str
    history: List[ChatHistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    text: str


class SettingsResponse(BaseModel):
    mode: str


class SettingsUpdateRequest(BaseModel):
    """Stage 7A-2 — `mode` is required and must be a string; allowlist
    membership is checked by app/preferences.py."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: str
