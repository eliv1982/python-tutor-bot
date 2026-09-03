"""Response models for the web adapter (Stage 6A/6C) — deliberately
minimal: only fields already proven safe to expose. Never a Telegram id,
GitHub numeric id, provider credential, or other internal persistence
detail (Section M: "Do not expose Telegram numeric id or GitHub numeric
id")."""

import uuid
from datetime import datetime

from pydantic import BaseModel


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
