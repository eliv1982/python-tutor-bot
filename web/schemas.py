"""Response models for the web adapter (Stage 6A) — deliberately minimal:
only fields already proven safe to expose in app/auth_session.UserProfile
(canonical id, created_at). Never a Telegram id, provider credential, or
other internal persistence detail."""

import uuid
from datetime import datetime

from pydantic import BaseModel


class CurrentUserResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
