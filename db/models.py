"""
Concrete ORM models (Stage 5C) — plain SQLAlchemy 2.x declarative classes,
not a generic repository/interface hierarchy. One class per persisted
entity; db/identity.py, db/preferences.py, db/documents.py own the actual
operations against these.
"""

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class User(Base):
    """Canonical internal user identity. Every other table hangs off this
    UUID — Telegram (and, later, any other adapter) is external identity,
    never itself the primary key of anything else in this schema."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TelegramAccount(Base):
    """Telegram numeric id -> internal user UUID mapping. The Telegram id
    IS the primary key (it is already a natural unique identity — no
    redundant surrogate key), and user_id is UNIQUE so one internal user
    can never accumulate two Telegram accounts pointing at it either
    (linking/merging multiple Telegram accounts to one user is explicitly
    out of Stage 5C's scope)."""

    __tablename__ = "telegram_accounts"

    telegram_user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True
    )
    linked_at: Mapped[datetime] = mapped_column(server_default=func.now())


class UserPreference(Base):
    """Durable mode/voice preferences. One row per user, created lazily on
    first write (see db/preferences.py's upsert helpers) — no separate
    "create preferences row" step. Deliberately does NOT store chat
    history (Stage 5C explicitly keeps that ephemeral/in-memory)."""

    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    mode: Mapped[str | None] = mapped_column(String, nullable=True)
    voice: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Document(Base):
    """Durable ownership/catalog record for a managed Telegram upload.

    `id` deliberately reuses the upload's own storage UUID (the physical
    filename's stem, uuid4().hex — see app/documents.py's
    _store_document_exclusively()) rather than generating a separate id:
    this keeps the DB row, the physical file, the sidecar, and the RAG
    document_id string ("upload:<hex>") in a clean 1:1 correspondence with
    no separate id-mapping table needed.

    `status`: 'pending' (physical file + sidecar durable, indexing not yet
    confirmed) -> 'active' (indexing confirmed). A row stuck at 'pending'
    indicates an interrupted ingest (e.g. a process crash) — the same
    category of orphan a crash can already leave in the physical
    file/sidecar/Qdrant layers today, with no automatic recovery daemon
    (documented as a known follow-up, not implemented here). This table is
    NOT the document content store — the physical file + sidecar remain
    the durable content source (see rag/sidecar.py); this is ownership/
    catalog metadata only."""

    __tablename__ = "documents"
    __table_args__ = (CheckConstraint("status IN ('pending', 'active')", name="status_valid"),)

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    stored_name: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
