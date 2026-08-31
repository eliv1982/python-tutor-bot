"""
Lazy singleton DB engine (Stage 5C) — mirrors rag/index.py's
get_vector_index()/close_vector_index() convention: nothing is
constructed merely by importing this module; a real Engine is created
only the first time something explicitly calls get_sync_engine(), guarded
by a lock so concurrent first-callers never race each other into
constructing two.

ONE sync engine, deliberately — not the async/sync split an async
application might suggest at first glance. Empirically verified against a
real local Postgres on this Windows deployment target: psycopg's async
mode raises `InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to
run in async mode` under asyncio.run()'s DEFAULT event loop on Windows.
Switching the whole application/test-suite's event loop policy to
SelectorEventLoop to work around this was rejected: pytest.ini/
tests/conftest.py/tests/test_stage1f_offline_enforcement.py/
tests/test_stage2a_text_llm_provider.py all explicitly document and rely
on ProactorEventLoop's self-pipe behavior for the already-accepted
Stage 1F offline-enforcement guarantees — too large a blast radius against
security-critical, already-reviewed test infrastructure, for no benefit
this stage actually needs.

Instead: ONE sync psycopg engine, used everywhere (db/identity.py,
db/preferences.py, db/documents.py), offloaded to a worker thread at
whichever async boundary calls it:
  - db/documents.py is called from INSIDE the existing executor-thread
    functions in app/documents.py (via utils.helpers.submit_worker()/
    await_worker()) — the same cancellation-safety mechanism already used
    for physical-file/sidecar writes and Qdrant reconciliation. A real OS
    thread survives asyncio cancellation of the awaiting Task.
  - db/identity.py/db/preferences.py are called via plain
    asyncio.to_thread() from app/identity.py/app/session.py — the exact
    idiom rag/query.py's similarity search and handlers/start.py's
    /stats command already use for read-mostly/idempotent blocking calls,
    where an ordinary (unshielded) await is already the accepted pattern:
    on cancellation, at worst nothing was created/updated and the
    caller's request simply fails and can be retried.

This is a smaller, more consistent design than a two-engine split would
have been, not a compromise — it reuses ONE existing concurrency idiom
(asyncio.to_thread()/submit_worker()) for every blocking-I/O boundary in
this codebase, sync DB access included.

DATABASE_URL is read as a FRESH module-attribute access
(db_settings.DATABASE_URL) inside get_sync_engine(), never bound at this
module's own import time — this exactly mirrors how rag/index.py reads
rag_constants.DATA_DIR fresh at VectorIndex construction time rather than
at import time, and is load-bearing: tests/conftest.py's Docker-Postgres
fixture only learns the container's assigned port after start, and must
monkeypatch.setattr(db_settings, "DATABASE_URL", ...) after every module
is already imported.
"""

import threading
from typing import Optional

from sqlalchemy import Engine, create_engine

import db.settings as db_settings

_sync_engine: Optional[Engine] = None
_lock = threading.Lock()


def get_sync_engine() -> Engine:
    """Return the shared sync Engine, constructing it on first call. A
    SQLAlchemy Engine and its connection pool are thread-safe by design —
    this is the normal, supported way to share one Engine across many
    worker-thread calls. Sized explicitly rather than left at the library
    default (5+10): utils.helpers.submit_worker()/asyncio.to_thread() both
    run on the asyncio loop's default executor, whose own default size
    (min(32, cpu_count()+4)) can exceed the default pool under concurrent
    requests — not a correctness issue (checkout just blocks briefly,
    graceful backpressure), but sized deliberately rather than left as an
    unconsidered coincidence."""
    global _sync_engine
    if _sync_engine is None:
        with _lock:
            if _sync_engine is None:
                _sync_engine = create_engine(db_settings.DATABASE_URL, pool_size=10, max_overflow=20)
    return _sync_engine


def close_db() -> None:
    """Dispose the engine (if constructed) and reset the singleton so a
    later get_sync_engine() call constructs a fresh one. Safe to call even
    if it was never constructed this process. Plain sync call (Engine.
    dispose() is synchronous) — main.py's shutdown_bot() calls this
    directly, same as its existing close_vector_index() call."""
    global _sync_engine
    with _lock:
        engine, _sync_engine = _sync_engine, None
    if engine is not None:
        engine.dispose()
