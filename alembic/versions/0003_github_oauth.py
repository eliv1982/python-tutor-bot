"""add github oauth tables

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-31 14:56:53.070379

Stage 6B — GitHub OAuth authentication. Two new tables, additive only,
never touching Stage 6A's `web_sessions`/`web_session_policy` schema/
policy invariants:

- `github_accounts`: GitHub numeric user id -> `users.id` mapping —
  structurally identical to `telegram_accounts` (0001_initial_schema.py):
  natural-key PK (no redundant surrogate id), `user_id` UNIQUE so one
  internal user can never accumulate two GitHub accounts. See
  db/models.py's GithubAccount docstring.
- `github_oauth_transactions`: short-lived, replay-resistant OAuth
  state/PKCE transaction persistence. `state_hash` (SHA-256 digest of the
  random `state` value, never the raw value itself) is the PK, with the
  same `octet_length(...) = 32` CHECK constraint 0002_web_sessions.py
  added for `web_sessions.session_token_hash` and for the identical
  reason: `sa.LargeBinary(length=32)` alone compiles to an unconstrained
  BYTEA on PostgreSQL. `created_at`/`expires_at` are `TIMESTAMP WITH TIME
  ZONE` for the same reason 0002 made `web_sessions`' timestamps
  timezone-aware (Stage 6A independent-audit corrective pass #1,
  Blocker 1): db.oauth_transactions.claim_sync()/create_sync() compare
  `expires_at` against PostgreSQL's own now(). See db/models.py's
  GithubOAuthTransaction docstring for the full rationale.

Stage 6B independent-audit corrective pass #1 (still pre-acceptance,
hence amended directly rather than layering a repair migration on top —
see db/models.py's GithubOAuthTransaction/GithubOAuthAdmission docstrings
for the full rationale of every change below):

- MAJOR 2: added `github_oauth_admission`, a singleton table (`id` CHECK-
  constrained to exactly 1) holding the one PostgreSQL-authoritative
  GLOBAL rate-window/outstanding-row admission state for GitHub OAuth
  login starts — db.oauth_transactions.create_sync() locks this row
  (`SELECT ... FOR UPDATE`) as the first statement of an atomic
  transaction that also deletes expired `github_oauth_transactions` rows
  and refuses to insert a new one at all once either bound is exhausted.
  This is what makes unauthenticated, unbounded `GET
  /api/auth/github/login` storage growth impossible. Seeded with its one
  row (`window_start = now()`, `starts_in_window = 0`) immediately after
  creation, in this same migration — never lazily created by application
  code, mirroring 0002_web_sessions.py's identical seeding rationale for
  `web_session_policy`.
- MAJOR 2 (Section 9): added a plain b-tree index on
  `github_oauth_transactions.expires_at` — create_sync()'s admission-path
  cleanup deletes every expired row on every call; without this index
  that cleanup degrades to a sequential scan as the table churns.
- MAJOR 2 (Section 8): DROPPED `github_oauth_transactions.consumed_at`.
  claim_sync() no longer marks a row consumed with an UPDATE; it now
  atomically DELETEs the row it claims (`DELETE ... WHERE state_hash = :h
  AND expires_at > now() ... RETURNING code_verifier`), so a claimed
  transaction's cleartext PKCE verifier no longer lingers in the table
  indefinitely, and every row remaining in the table at any moment is, by
  construction, still live — exactly the "outstanding transaction count"
  the new admission control above counts against its hard cap. Since
  Stage 6B was still uncommitted at review time, this migration is
  amended directly rather than adding a separate 0004 drop-column
  migration.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: Union[str, None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'github_accounts',
        sa.Column('github_user_id', sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('linked_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_github_accounts_user_id_users')),
        sa.PrimaryKeyConstraint('github_user_id', name=op.f('pk_github_accounts')),
        sa.UniqueConstraint('user_id', name=op.f('uq_github_accounts_user_id')),
    )

    op.create_table(
        'github_oauth_transactions',
        sa.Column('state_hash', sa.LargeBinary(length=32), nullable=False),
        sa.Column('code_verifier', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            'octet_length(state_hash) = 32',
            name=op.f('ck_github_oauth_transactions_state_hash_length'),
        ),
        sa.PrimaryKeyConstraint('state_hash', name=op.f('pk_github_oauth_transactions')),
    )
    op.create_index(
        op.f('ix_github_oauth_transactions_expires_at'), 'github_oauth_transactions', ['expires_at'], unique=False
    )

    # github_oauth_admission: singleton global rate-window/outstanding-row
    # admission state (Stage 6B independent-audit corrective pass #1,
    # MAJOR 2) — not autogenerated, hand-added for the same reason
    # web_session_policy above already established.
    op.create_table(
        'github_oauth_admission',
        sa.Column('id', sa.SmallInteger(), autoincrement=False, nullable=False),
        sa.Column('window_start', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('starts_in_window', sa.Integer(), server_default='0', nullable=False),
        sa.CheckConstraint('id = 1', name=op.f('ck_github_oauth_admission_singleton_id')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_github_oauth_admission')),
    )
    # Seed the one and only row — application code never lazily creates
    # this row (see db/models.py's GithubOAuthAdmission docstring for why
    # that sidesteps a first-start race entirely, mirroring
    # web_session_policy's identical seeding rationale immediately above).
    op.execute(sa.text("INSERT INTO github_oauth_admission (id, window_start, starts_in_window) VALUES (1, now(), 0)"))


def downgrade() -> None:
    op.drop_table('github_oauth_admission')
    op.drop_index(op.f('ix_github_oauth_transactions_expires_at'), table_name='github_oauth_transactions')
    op.drop_table('github_oauth_transactions')
    op.drop_table('github_accounts')
