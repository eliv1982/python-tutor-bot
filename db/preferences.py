"""
Durable mode/voice preferences (Stage 5C) — SYNC (see db/engine.py's
module docstring). app/session.py/app/preferences.py wrap each of these
via utils.helpers.submit_worker()/await_worker() so their own async
get/set methods stay `async def` for callers — see db/engine.py's own
docstring (Stage 7A-3 unified-runtime corrective pass) for why this is no
longer a plain `asyncio.to_thread()`.

get_preferences_sync() returns (None, None) for a user with no row yet —
it deliberately does NOT apply BotMode/DEFAULT_VOICE fallback defaults
itself; that stays in app/session.py (mirroring exactly the fallback the
in-memory implementation already used: `self.sessions.get(key, "text")`),
so a future default change never requires a data migration here.

set_mode_sync()/set_voice_sync() are independent single-column upserts
(not a combined "set both" call) — first write for a brand-new user
creates the row (INSERT ... ON CONFLICT DO UPDATE), and only the written
column changes; the other stays whatever it already was (or NULL, for a
genuinely new row where only one of the two has ever been set).
"""

import uuid
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import UserPreference


def get_preferences_sync(user_id: uuid.UUID) -> Tuple[Optional[str], Optional[str]]:
    with Session(get_sync_engine()) as session:
        result = session.execute(
            select(UserPreference.mode, UserPreference.voice).where(UserPreference.user_id == user_id)
        )
        row = result.first()
        if row is None:
            return None, None
        return row.mode, row.voice


def set_mode_sync(user_id: uuid.UUID, mode: str) -> None:
    with Session(get_sync_engine()) as session:
        stmt = pg_insert(UserPreference).values(user_id=user_id, mode=mode)
        stmt = stmt.on_conflict_do_update(index_elements=[UserPreference.user_id], set_={"mode": mode})
        session.execute(stmt)
        session.commit()


def set_voice_sync(user_id: uuid.UUID, voice: str) -> None:
    with Session(get_sync_engine()) as session:
        stmt = pg_insert(UserPreference).values(user_id=user_id, voice=voice)
        stmt = stmt.on_conflict_do_update(index_elements=[UserPreference.user_id], set_={"voice": voice})
        session.execute(stmt)
        session.commit()
