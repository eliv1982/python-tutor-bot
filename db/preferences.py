"""
Durable mode/voice preferences (Stage 5C) — SYNC (see db/engine.py's
module docstring). app/session.py/app/preferences.py wrap each of these
via utils.helpers.submit_worker()/await_worker() so their own async
get/set methods stay `async def` for callers — see db/engine.py's own
docstring (Stage 7A-3 unified-runtime corrective pass) for why this is no
longer a plain `asyncio.to_thread()`.

get_preferences_sync() returns (None, None) for a user with no row yet —
it deliberately does NOT apply BotMode/DEFAULT_VOICE fallback defaults
itself; the effective-default policy lives in app/preferences.py (the one
resolver Telegram and the web settings API both use), so a future default
change never requires a data migration here. This table stores persisted
customization only: nothing writes a default into it (Telegram's /start does
not), and a missing row simply means "the configured defaults apply".

set_mode_sync()/set_voice_sync() are independent single-column upserts
(not a combined "set both" call) — first write for a brand-new user
creates the row (INSERT ... ON CONFLICT DO UPDATE), and only the written
column changes; the other stays whatever it already was (or NULL, for a
genuinely new row where only one of the two has ever been set).

Lock discipline (Stage 7B-3P): both writers first take `FOR KEY SHARE` on
the owning `users` row, THEN upsert. That is the same lock the
`user_preferences.user_id -> users.id` foreign-key check would take
implicitly, just taken BEFORE the row is written rather than after. The
order matters: db.telegram_link.redeem_attempt_sync() holds `FOR UPDATE` on
both users rows while it transfers a GitHub/web user's preference row onto
the Telegram user. A writer that inserted first and only then hit the FK
check would already hold an uncommitted `user_preferences` row for that key
while waiting on redemption's users lock — and redemption's transfer would
wait on that uncommitted row: a genuine PostgreSQL deadlock (reproduced
during Stage 7B-3P). Locking users first makes the writer wait BEFORE
writing anything, so lock order is uniformly `users` -> `user_preferences`
in both directions, and whichever side commits first is simply visible to
the other. If the user row is already gone (a merge deleted it) nothing is
locked and the INSERT fails with the same foreign-key error it always did.

Ø / D / M classification (Stage 7B-3P). For Telegram-link identity merging
(db.telegram_link.redeem_attempt_sync) a user's preference state is one of:

    Ø  ABSENT              no user_preferences row
    D  DEFAULT_EQUIVALENT  a row whose removal changes no observable behavior
    M  MATERIAL            everything else

A row is MATERIAL iff `voice IS NOT NULL OR mode NOT IN (NULL, the current
DEFAULT_MODE)`. So (mode NULL, voice NULL) and (mode = DEFAULT_MODE, voice
NULL) are D; a canonical non-default mode, any non-NULL voice (even one equal
to DEFAULT_VOICE — automatic code never initialized voice, so a stored voice
is always a customization) and a legacy/noncanonical mode (which reads as
DEFAULT_MODE but is data we do not silently discard) are M. `updated_at` never
participates. The accepted tradeoff: an explicit selection of the current
default mode and a historical automatic initialization to it are
indistinguishable and both D, because dropping that row leaves the effective
mode unchanged; if BOT_MODE later changes, a stored old-default row becomes M
(it no longer equals the default) and is never silently discarded.
"""

import uuid
from enum import Enum
from typing import Optional, Tuple

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import User, UserPreference


def get_preferences_sync(user_id: uuid.UUID) -> Tuple[Optional[str], Optional[str]]:
    with Session(get_sync_engine()) as session:
        result = session.execute(
            select(UserPreference.mode, UserPreference.voice).where(UserPreference.user_id == user_id)
        )
        row = result.first()
        if row is None:
            return None, None
        return row.mode, row.voice


def _lock_owner_row(session: Session, user_id: uuid.UUID) -> bool:
    """Takes `FOR KEY SHARE` on the owning `users` row. Returns whether that
    row exists (nothing is locked when it does not)."""
    return (
        session.execute(
            select(User.id).where(User.id == user_id).with_for_update(read=True, key_share=True)
        ).first()
        is not None
    )


class PreferenceState(Enum):
    """A user's preference state for link-merge purposes — see this module's
    docstring ("Ø / D / M classification")."""

    ABSENT = "absent"  # Ø
    DEFAULT_EQUIVALENT = "default_equivalent"  # D
    MATERIAL = "material"  # M


def _material_predicate(default_mode: str):
    """SQL for "this row is MATERIAL": `voice IS NOT NULL OR (mode IS NOT NULL
    AND mode <> :default_mode)`. Written with explicit IS NOT NULL terms so
    NULL semantics never decide the result."""
    return or_(
        UserPreference.voice.is_not(None),
        and_(UserPreference.mode.is_not(None), UserPreference.mode != default_mode),
    )


def preference_state(session: Session, user_id: uuid.UUID, default_mode: str) -> PreferenceState:
    """Classifies `user_id`'s preference row inside the caller's transaction
    (redemption calls this under its users-row locks, so the answer cannot
    change before that transaction ends — see this module's lock discipline).
    Read-only: never repairs or rewrites what it classifies."""
    material = session.execute(
        select(_material_predicate(default_mode)).where(UserPreference.user_id == user_id)
    ).scalar_one_or_none()
    if material is None:
        return PreferenceState.ABSENT
    return PreferenceState.MATERIAL if material else PreferenceState.DEFAULT_EQUIVALENT


def classify_preference_sync(user_id: uuid.UUID, default_mode: str) -> PreferenceState:
    """Standalone, read-only Ø/D/M classification (its own short
    transaction) — the same predicate redemption uses."""
    with Session(get_sync_engine()) as session:
        return preference_state(session, user_id, default_mode)


def set_mode_sync(user_id: uuid.UUID, mode: str) -> None:
    with Session(get_sync_engine()) as session:
        _lock_owner_row(session, user_id)
        stmt = pg_insert(UserPreference).values(user_id=user_id, mode=mode)
        stmt = stmt.on_conflict_do_update(index_elements=[UserPreference.user_id], set_={"mode": mode})
        session.execute(stmt)
        session.commit()


def set_voice_sync(user_id: uuid.UUID, voice: str) -> None:
    with Session(get_sync_engine()) as session:
        _lock_owner_row(session, user_id)
        stmt = pg_insert(UserPreference).values(user_id=user_id, voice=voice)
        stmt = stmt.on_conflict_do_update(index_elements=[UserPreference.user_id], set_={"voice": voice})
        session.execute(stmt)
        session.commit()
