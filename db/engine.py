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
db/preferences.py, db/documents.py, db/auth_sessions.py,
db/telegram_link.py, db/github_identity.py, db/oauth_transactions.py),
offloaded to a worker thread at whichever async boundary calls it, via
utils.helpers.submit_worker()/await_worker() — see that module's own
docstring for the exact mechanism.

Stage 7A-3 unified-runtime corrective pass (independent-audit MAJOR — a
cancelled Task awaiting plain `asyncio.to_thread()` does not stop or await
the worker thread): every one of these call sites used to be split into
two idioms — db/documents.py's multi-step physical-file/sidecar/Qdrant
writes went through submit_worker()/await_worker(), while every other
db/*.py module (identity, preferences, auth_sessions, telegram_link,
github_identity, oauth_transactions) went through a plain, unshielded
`asyncio.to_thread()`, reasoned about purely at the BUSINESS level: each
of those calls is a single, idempotent-enough statement, so on ordinary
per-request cancellation "at worst nothing was created/updated and the
caller's request simply fails and can be retried" — a reasoning that
remains true today and is NOT what this pass changes.

What that reasoning never accounted for is PROCESS-SHUTDOWN-level
resource lifecycle safety: service_main.py's unified composition root
settles every adapter-owned Task (Telegram's polling + its
`_pending_tasks`, Uvicorn's `serve()` + its lifespan task + its
`server_state.tasks`) and only THEN calls `close_resources()` (this
module's own `close_db()`, plus `rag.index.close_vector_index()`). A
plain `asyncio.to_thread()` call lets its owning Task report itself
"settled" (cancelled) the instant cancellation is requested, regardless of
whether the underlying OS thread is still actually running the sync DB
call against `get_sync_engine()` — so `close_db()` could begin disposing
the shared engine while one of these threads is still using a connection
checked out from its pool. Every blocking-I/O boundary in this codebase
now goes through the SAME submit_worker()/await_worker() primitive
specifically so that never happens: the owning Task cannot settle until
the thread genuinely has, which is exactly the property `_settle_
telegram_pending_tasks()`/the Uvicorn request-task equivalent in
service_main.py rely on to make "adapter settled" transitively imply "its
resource-sensitive worker descendants settled" too.

This is a smaller, more consistent design than a two-primitive split
would have been, not a compromise — ONE concurrency idiom
(submit_worker()/await_worker()) for every blocking-I/O boundary in this
codebase, sync DB access included.

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
