"""add 'deleting' to documents.status check constraint

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-23 00:00:00.000000

Stage 7A-3 — authenticated Documents + Retrieval API. Adds 'deleting' as a
third legal `documents.status` value, alongside the existing 'pending' and
'active' (see db/models.py's Document docstring): an authenticated,
owner-initiated document delete atomically transitions a row from 'active'
to 'deleting' (db.documents.begin_or_resume_delete_sync()) before any
physical/index/catalog cleanup runs, so a delete that fails partway leaves
a durable, resumable marker rather than either an inconsistent 'active' row
or an outright-missing one. `db.documents.ACTIVE_STATUSES` remains exactly
`{'active'}`, so 'deleting' rows — like 'pending' ones — are never listed,
never resolvable by detail lookup, and never retrieval-visible; nothing
about that gate changes here.

Purely additive to the existing CHECK constraint — no new column, no new
table, no data migration. Same drop-and-recreate idiom
alembic/versions/0004_telegram_link_attempts.py already uses for a
narrower CHECK constraint (there, adding a column's own bound; here,
widening an existing column's already-established allowlist).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0005'
down_revision: Union[str, None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(op.f('ck_documents_status_valid'), 'documents', type_='check')
    op.create_check_constraint(
        op.f('ck_documents_status_valid'), 'documents', "status IN ('pending', 'active', 'deleting')"
    )


def downgrade() -> None:
    op.drop_constraint(op.f('ck_documents_status_valid'), 'documents', type_='check')
    op.create_check_constraint(
        op.f('ck_documents_status_valid'), 'documents', "status IN ('pending', 'active')"
    )
