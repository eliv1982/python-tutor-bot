"""add telegram_link_attempts table, github OAuth generation/tombstone protocol

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-02 00:00:00.000000

Stage 6C — secure Telegram <-> GitHub/web identity linking. One new table,
additive only, never touching Stage 6A/6B's existing schema/policy
invariants:

- `telegram_link_attempts`: short-lived, replay-resistant server-side state
  for one authenticated web (GitHub) user's outstanding request to link
  their canonical identity to a Telegram account. `web_user_id` is the
  PRIMARY KEY (at most one outstanding attempt per canonical user by
  construction — see db/telegram_link.py's create_attempt_sync()), with an
  explicit `ON DELETE RESTRICT` foreign key to `users.id` — deliberately
  NOT the unspecified-FK default every other table in this schema uses,
  and deliberately NOT a cascade: a `users` row must never be deleted while
  it still holds an outstanding link attempt, and every code path that
  deletes a `users` row (merge-on-redemption, GitHub-only unlink) is
  required to delete this row first, in the same transaction, as part of
  the corrected lock order this stage introduces. `link_secret_hash`
  (SHA-256 digest of the raw bearer secret, never the raw secret itself)
  is UNIQUE with the same `octet_length(...) = 32` CHECK constraint every
  other digest column in this schema carries (`web_sessions.
  session_token_hash`, `github_oauth_transactions.state_hash`) — for the
  identical reason: `sa.LargeBinary(length=32)` alone compiles to an
  unconstrained BYTEA on PostgreSQL. `created_at`/`expires_at` are
  `TIMESTAMP WITH TIME ZONE`, mirroring every other short-lived bearer-
  secret table in this schema, with a plain b-tree index on `expires_at`
  for the same bounded-cleanup reason `github_oauth_transactions.
  expires_at` carries one. See db/models.py's TelegramLinkAttempt
  docstring for the full rationale.

Stage 6C corrective pass (independent-audit MAJOR 1) — while this migration
was still uncommitted, added a durable, database-serialized OAuth
generation/tombstone protocol closing a real race: an OAuth callback that
has already authenticated its transaction with GitHub, but has not yet
resolved/created the canonical GitHub mapping, could otherwise "undo" a
concurrent unlink by recreating (or re-attaching to) a mapping the unlink
just tore down. Amended directly into this still-unaccepted migration
rather than a separate 0005, for the identical "still pre-acceptance"
rationale 0003_github_oauth.py's own corrective-pass amendment documents:

- `github_oauth_admission.unlink_generation` (BIGINT, NOT NULL, default 0,
  CHECK >= 0): the ONE global, PostgreSQL-authoritative unlink-generation
  counter, stored on the SAME singleton row db.oauth_transactions.
  create_sync() already locks first for admission control — reusing that
  existing lock, rather than adding a new singleton, keeps this protocol's
  one shared serialization point for "capture the current generation" and
  "advance the generation" a single row-lock idiom this schema already
  established (db/models.py's GithubOAuthAdmission docstring).
- `github_oauth_transactions.auth_generation` (BIGINT, NOT NULL, default 0,
  CHECK >= 0): the exact `unlink_generation` value in effect at the moment
  `/api/auth/github/login` created this transaction (db.oauth_transactions.
  create_sync(), under the SAME admission-row lock) — carried through
  claim_sync() and threaded into the callback's generation-aware resolver
  (db.github_identity.resolve_or_create_user_by_github_id_for_oauth_sync()).
  A pre-existing transaction implicitly reads as generation 0 (the column
  default) and remains valid unless a LATER unlink tombstones its GitHub id
  at a higher generation.
- `github_unlink_tombstones`: one bounded row per GitHub identity that has
  ever been unlinked, recording the `unlink_generation` value that unlink
  established (`github_user_id` PRIMARY KEY, `unlink_generation` BIGINT NOT
  NULL CHECK > 0, `unlinked_at` TIMESTAMPTZ for diagnostics only).
  Deliberately carries NO foreign key to `github_accounts`: the whole point
  of this table is to remember a GitHub identity's unlink history AFTER its
  `github_accounts` mapping has been removed, so an FK to a row that no
  longer exists (by design) would be incoherent. The generation-aware
  resolver rejects any OAuth transaction whose `auth_generation` is stale
  relative to this table's `unlink_generation` for the same GitHub id —
  even when a NEWER mapping already exists for that id (Section D of the
  corrective pass: "the generation check applies even when a GitHub mapping
  already exists" — this is what stops a stale callback from ever
  attaching to a mapping a newer login already recreated). See
  db/telegram_link.py's module docstring and db/github_identity.py's
  resolve_or_create_user_by_github_id_for_oauth_sync() for the full
  transactional protocol both sides of this table implement.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0004'
down_revision: Union[str, None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'telegram_link_attempts',
        sa.Column('web_user_id', sa.UUID(), nullable=False),
        sa.Column('link_secret_hash', sa.LargeBinary(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            'octet_length(link_secret_hash) = 32',
            name=op.f('ck_telegram_link_attempts_link_secret_hash_length'),
        ),
        sa.ForeignKeyConstraint(
            ['web_user_id'], ['users.id'],
            name=op.f('fk_telegram_link_attempts_web_user_id_users'),
            ondelete='RESTRICT',
        ),
        sa.PrimaryKeyConstraint('web_user_id', name=op.f('pk_telegram_link_attempts')),
        sa.UniqueConstraint('link_secret_hash', name=op.f('uq_telegram_link_attempts_link_secret_hash')),
    )
    op.create_index(
        op.f('ix_telegram_link_attempts_expires_at'), 'telegram_link_attempts', ['expires_at'], unique=False
    )

    # --- Stage 6C corrective pass: OAuth generation/tombstone protocol ----

    op.add_column(
        'github_oauth_admission',
        sa.Column('unlink_generation', sa.BigInteger(), server_default='0', nullable=False),
    )
    op.create_check_constraint(
        op.f('ck_github_oauth_admission_unlink_generation_non_negative'),
        'github_oauth_admission',
        'unlink_generation >= 0',
    )

    op.add_column(
        'github_oauth_transactions',
        sa.Column('auth_generation', sa.BigInteger(), server_default='0', nullable=False),
    )
    op.create_check_constraint(
        op.f('ck_github_oauth_transactions_auth_generation_non_negative'),
        'github_oauth_transactions',
        'auth_generation >= 0',
    )

    op.create_table(
        'github_unlink_tombstones',
        sa.Column('github_user_id', sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column('unlink_generation', sa.BigInteger(), nullable=False),
        sa.Column('unlinked_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint(
            'unlink_generation > 0', name=op.f('ck_github_unlink_tombstones_unlink_generation_positive')
        ),
        sa.PrimaryKeyConstraint('github_user_id', name=op.f('pk_github_unlink_tombstones')),
    )


def downgrade() -> None:
    op.drop_table('github_unlink_tombstones')

    op.drop_constraint(
        op.f('ck_github_oauth_transactions_auth_generation_non_negative'),
        'github_oauth_transactions',
        type_='check',
    )
    op.drop_column('github_oauth_transactions', 'auth_generation')

    op.drop_constraint(
        op.f('ck_github_oauth_admission_unlink_generation_non_negative'), 'github_oauth_admission', type_='check'
    )
    op.drop_column('github_oauth_admission', 'unlink_generation')

    op.drop_index(op.f('ix_telegram_link_attempts_expires_at'), table_name='telegram_link_attempts')
    op.drop_table('telegram_link_attempts')
