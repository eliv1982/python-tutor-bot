"""
Unified runtime-topology composition root (Stage 7A-3 prerequisite).

Local Qdrant (rag/index.py) opens with `QdrantClient(path=...)`, which
holds an exclusive on-disk storage lock for the life of the client — two
separate OS processes (the historical `main.py` Telegram process and
`web_main.py` web process) can never safely open the SAME local Qdrant
directory at once. Stage 7A-3 needs both adapters to reach the same
document/retrieval layer, so this module runs Telegram polling (bot.py's
existing AsyncTeleBot instance) and the FastAPI/Uvicorn web adapter
(web.app.create_app()) in ONE process, sharing exactly one process-local
`rag.index.get_vector_index()` singleton — never a second Qdrant client.

Reuse, not a rewrite: Telegram setup/shutdown is main.py's own
setup_bot()/shutdown_bot(), imported and called directly; the web adapter
is web.app.create_app() (owns_db_lifecycle=False — see that module's
docstring) served via uvicorn's programmatic Server API. Nothing about
route registration, authentication, session setup, or FastAPI lifespan
behavior is duplicated here.

Supervision (corrective pass — independent-audit MAJOR 1/2, MINOR
simultaneous-failure/cleanup-masking/Telegram-stop findings):
run_unified_service() still runs both adapters as plain asyncio.Task
objects supervised by a hand-rolled asyncio.wait(FIRST_COMPLETED) loop
rather than asyncio.TaskGroup — an adapter terminating unexpectedly must
let its SIBLING finish its own graceful shutdown path first (uvicorn.
Server in particular only closes its sockets and runs the FastAPI
lifespan's shutdown when its own `main_loop()` notices `should_exit` and
returns NORMALLY; a raw `Task.cancel()`, TaskGroup's automatic response
to a sibling raising, interrupts `Server.serve()` wherever it currently
is and skips that self-shutdown entirely). This module therefore always
signals a cooperative stop first and gives the sibling a bounded grace
period to exit on its own, force-cancelling only as a fallback.

The corrective pass rewrote HOW that supervision is made cancellation-
safe. `_supervise()` (not `run_unified_service()` itself) is now the
single place that owns the whole algorithm: wait for either adapter to
finish, settle whichever one hasn't, retrieve every adapter's exception
deterministically (stable Telegram-then-Uvicorn order — never
`next(iter(done))`, MINOR finding), close shared resources exactly once,
then propagate the correctly-prioritized outcome (see `_supervise()`'s
own docstring for the exact precedence). The two phases that MUST run to
completion before the next one begins — settling every adapter task, and
closing shared resources — are each run through `_run_uncancellable()`,
which keeps re-attaching to that phase's own independent Task no matter
how many times run_unified_service()'s OWN task is cancelled (once,
during the initial wait; again, while settlement or cleanup is already
in progress; any number of additional times). The previous
implementation's `_graceful_stop()` had no such protection — a
cancellation landing while it was itself already running (e.g. a second
Ctrl+C during sibling teardown) escaped mid-settlement, letting
`close_resources()` run while an adapter task was still alive. See
`_run_uncancellable()`'s docstring for exactly how this is achieved with
plain `asyncio.shield()`, and tests/test_stage7a3_unified_runtime.py's
"cancellation while ... already in progress" tests for the regression
proof.

Uvicorn's own lifespan child task (independent-audit MAJOR 2): installed
uvicorn 0.52.4's `LifespanOn.startup()` (uvicorn/lifespan/on.py) creates
a SEPARATE `asyncio.Task` (`self.main()`, ASGI lifespan protocol driver)
via a bare `loop.create_task(...)` and keeps no reference to it anywhere
reachable once `startup()` returns — only a local variable. Force-
cancelling THIS module's own uvicorn adapter task only cancels
`Server.serve()`'s own coroutine chain; it has no way to reach that
sibling task, so if ASGI lifespan startup itself hangs, forced
cancellation used to leave it running forever. `_TrackedLifespanOn`
below is uvicorn's own `LifespanOn` with `startup()` overridden to ALSO
keep that task as a reachable attribute (`main_lifespan_task`) — every
other method (`main()`/`shutdown()`/`send()`/`receive()`) is inherited
unchanged from the installed library. `_build_uvicorn_adapter()` wires it
in via `uvicorn.Config.lifespan_class` (the only supported extension
point — `Server._serve()` constructs `self.lifespan = config.
lifespan_class(config)` itself, so `config.load()` must run, with our
override already applied, before `uvicorn.Server(config)` ever serves).
`_settle_uvicorn_lifespan_task()` then explicitly cancels-and-awaits that
task whenever the uvicorn adapter's own `run()` ends, for any reason.
This is pinned to the installed uvicorn 0.52.4 implementation and must be
re-verified against `venv/Lib/site-packages/uvicorn/lifespan/on.py` on
any uvicorn upgrade.

Telegram shutdown (independent-audit MINOR finding): `_default_stop_
telegram()`'s `bot._polling = False` is BEST-EFFORT cooperative
signalling only, never a correctness guarantee — installed pyTelegramBot
API 4.36.1's own `_process_polling()` unconditionally resets `_polling =
True` again on every (re)entry (async_telebot.py, e.g. after
infinity_polling()'s outer loop retries a transient exception), which can
silently overwrite a `False` request, and aiohttp's own request timeout
for an in-flight `get_updates()` call defaults to 5 minutes — nowhere
near bounded by this module's own grace period. The actual, always-
bounded stop mechanism is `_settle_tasks()`'s existing force-cancel-then-
await fallback: pyTelegramBotAPI's `_process_polling()` explicitly
catches `asyncio.CancelledError` inside its own polling loop and returns
cleanly through its own `finally` (closing the aiohttp session); a
cancellation delivered earlier (its own `get_me()` call, or this module's
`setup_bot()` before `infinity_polling()` is even entered) propagates as
an ordinary `CancelledError` instead and settles the same way through
`_settle_tasks()`. `main.shutdown_bot()` (this module's `close_resources`
default) closes the aiohttp session unconditionally either way, as the
final safety net.

Ownership: `_supervise()` is the ONE place that closes shared process-
wide resources (`close_resources`, default `main.shutdown_bot` — bot
session, the shared Qdrant singleton, the shared DB engine), called
exactly once per call, always after every adapter task has fully
settled. The web adapter never closes the DB engine itself in this mode
(`create_app(owns_db_lifecycle=False)`) and never touches Qdrant at all
(no Stage 7A-3 HTTP route exists yet).

Second corrective pass (independent re-audit — two remaining lifecycle
findings):

1. Telegram handler child tasks (independent-audit MAJOR): installed
   pyTelegramBotAPI 4.36.1's `_process_polling()` (venv/Lib/site-packages/
   telebot/async_telebot.py) creates one `asyncio.Task` per batch of
   updates (`self.process_new_updates(updates)`), tracked in a private,
   self-discarding `AsyncTeleBot._pending_tasks` set — `bot.
   infinity_polling()` returning/raising/being cancelled says nothing
   about whether one of these is still alive. "Telegram adapter settled"
   now means polling settled AND every Telegram-owned pending handler
   task settled — see `_settle_telegram_pending_tasks()`'s own docstring.
   The ownership boundary lives inside `_default_run_telegram()` itself
   (a `finally` around the polling call), not the top-level supervisor,
   so `_supervise()` stays ignorant of Telegram internals. `_process_
   polling()`'s own `finally` already closes the aiohttp session before
   `infinity_polling()` returns/raises for any reason, so a still-live
   handler task can no longer safely make further Telegram HTTP calls by
   the time settlement runs — every live task is force-cancelled rather
   than given more time.

2. `_run_uncancellable()` BaseException safety (independent-audit MAJOR):
   the previous implementation ran its protected coroutine directly as a
   bare child task (`asyncio.ensure_future(coro)`) — the exact same class
   of bug `_AdapterBaseExceptionWrapper` already exists to fix for
   adapters, just unfixed here. A SystemExit/KeyboardInterrupt raised by
   `coro` itself (e.g. `close_resources()`) would hit CPython asyncio's
   Task-level special case and re-raise straight out of the event loop's
   own callback dispatch, before `_run_uncancellable()` ever got a chance
   to apply precedence. `_run_protected()`/`_ProtectedOutcome` generalize
   the same containment technique without needing a dedicated wrapper
   BaseException subclass this time: the child task's own top-level frame
   (`_run_protected(coro)`) catches literally every `BaseException` coro
   can raise — ordinary exceptions, SystemExit/KeyboardInterrupt, and a
   CancelledError coro raises on ITSELF (kept strictly distinct from
   `was_cancelled`, which tracks external cancellation of the code
   AWAITING `_run_uncancellable()`) — and turns it into ordinary returned
   data instead. The child task therefore always completes normally from
   the scheduler's point of view, exactly like `_supervised_adapter()`.

3. Unexpected adapter cancellation (independent-audit MINOR): an adapter
   task found ALREADY cancelled in the `done` set from `_supervise()`'s
   very first `asyncio.wait(..., FIRST_COMPLETED)` — i.e. before this
   composition root ever requested that adapter stop — used to be
   silently skipped by `_select_primary_exception()`, letting the service
   return successfully. `done` is captured exactly once, before
   `_settle_tasks()`'s own later force-cancel fallback ever runs, so this
   can only ever fire for a genuine "adapter cancelled itself" case, never
   for ordinary sibling teardown — see `UnifiedServiceAdapterCancelled`.

4. Dependency pins: `_TrackedLifespanOn` mirrors installed uvicorn
   0.52.4's private `LifespanOn.startup()` line-for-line, and
   `_settle_telegram_pending_tasks()` relies on installed pyTelegramBotAPI
   4.36.1's private `_pending_tasks` attribute — requirements.txt now
   pins both exactly (`uvicorn==0.52.4`, `pyTelegramBotAPI==4.36.1`)
   instead of a bounded range, so a future 0.x/4.x release can never
   silently change either private surface out from under this module.
   Re-verify both integration points (and re-pin deliberately) on upgrade.

Third corrective pass (independent re-audit — MAJOR: "adapter settlement
is not transitive to reachable worker/request descendants"):

1. Plain `asyncio.to_thread()` reachable from either adapter, for any call
   touching the shared DB engine (db/engine.py) or the shared Qdrant
   singleton (rag/index.py), has been replaced across the codebase
   (app/identity.py, app/session.py, app/preferences.py,
   app/auth_session.py, app/telegram_link.py, app/github_identity.py,
   app/oauth_transaction.py, rag/query.py, handlers/start.py) with
   `utils.helpers.submit_worker()`/`await_worker()` — see db/engine.py's
   own module docstring for the full rationale (a cancelled Task awaiting
   plain `asyncio.to_thread()` does not stop or await its worker thread,
   so `close_resources()` below could begin disposing the shared
   engine/Qdrant client while one was still using it). This is what makes
   `_settle_telegram_pending_tasks()`'s existing `await asyncio.wait(
   pending)` above, and `_settle_uvicorn_request_tasks()` below, correctly
   imply "this task's resource-sensitive worker descendants have also
   settled" purely as a consequence of `await_worker()`'s own contract
   (utils/helpers.py) — no additional plumbing was needed here for that
   half of the finding. Document ingestion (app/documents.py) and voice
   OGG->WAV conversion (services/stt.py) already used this exact primitive
   before this pass and are unchanged.

2. Uvicorn owns active ASGI request tasks too: `server.serve()`
   returning/raising/being cancelled only means the accept loop and
   `LifespanOn` settled — installed uvicorn 0.52.4's own `Server.
   shutdown()` (venv/Lib/site-packages/uvicorn/server.py) can itself
   cancel surviving `server_state.tasks` without ever awaiting them (its
   `config.timeout_graceful_shutdown` branch), and forced cancellation of
   the OUTER `server.serve()` Task (this module's own `_settle_tasks()`
   fallback) can interrupt `Server._serve()` before its `shutdown()` ever
   runs at all, leaving `server_state.tasks`/`connections` completely
   unaddressed. `_settle_uvicorn_request_tasks()` closes both, called from
   `_build_uvicorn_adapter()`'s own `run()` `finally` (BEFORE `_settle_
   uvicorn_lifespan_task()`, mirroring installed `Server.shutdown()`'s own
   ordering) every time `run()` ends for any reason — see that function's
   own docstring for the exact mechanism.

Clean-shutdown corrective pass (manually reproduced bug: a normal operator
Ctrl+C on `python service_main.py` shut every adapter down correctly and
then ended with `UnifiedServiceAdapterExited` / exit code 1):

`uvicorn.Server.serve()` runs inside `Server.capture_signals()`
(venv/Lib/site-packages/uvicorn/server.py), which REPLACES the process's
SIGINT/SIGTERM handlers for as long as it runs. A real Ctrl+C therefore
never reaches `asyncio.run()`'s own SIGINT handler (which would cancel this
call's task) first — uvicorn's `handle_exit()` consumes it, sets
`server.should_exit` itself, drains, and `serve()` returns NORMALLY. This
composition root's own `_stop_uvicorn` closure was never involved, so
`stopped["uvicorn"]` stayed False and `_supervised_adapter()` correctly-by-
its-old-rules classified a normal return "without being asked" as
`UnifiedServiceAdapterExited`. Only afterwards does `capture_signals()`
restore the previous handlers and re-raise the captured signal, which DOES
reach `asyncio.run()`'s handler and cancels this call's task — but uvicorn's
adapter task completes in that same, uninterrupted event-loop step, so by
the time `_supervise()` observes that cancellation the adapter is already in
`done` with `UnifiedServiceAdapterExited`, and precedence (1) (a genuine
adapter outcome) outranks precedence (2) (external cancellation). The
`CancelledError` that `asyncio.run()` would have turned into a clean
`KeyboardInterrupt` (exit 0) was therefore discarded in favor of a fatal
error (exit 1).

The fix keeps every fail-fast rule as-is and only corrects what counts as
"asked to stop": `_build_uvicorn_adapter()` now also returns a
`shutdown_requested()` predicate over `server.should_exit` — a flag only two
things ever set, this module's own `stop()` and uvicorn's own operator-signal
handler (a lifespan/bind failure raises SystemExit instead and a
`limit_max_requests` exit never sets it, so neither can be mistaken for a
requested shutdown). `run_unified_service()` treats the Uvicorn adapter as
asked-to-stop when EITHER its own `stopped` flag OR that predicate is set.
An adapter returning normally with neither set is still an unexpected exit,
and a genuine adapter exception is still never consulted against either.
The former `__main__` block is now `_main()` so the exit-code mapping
(operator shutdown -> 0, any other unexpected outcome -> 1) is directly
testable.
"""

import asyncio
import contextlib
import dataclasses
import os
import sys
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

import uvicorn
from uvicorn.lifespan.on import LifespanOn

from bot import bot
from main import setup_bot, shutdown_bot
from utils.logging import configure_logging, logger
from web.app import create_app

# Not a guarantee that every in-flight Telegram HTTP call finishes within
# this window (aiohttp's own request timeout defaults to 5 minutes, and
# infinity_polling()'s error-retry backoff can reach 60s — see
# _default_stop_telegram()'s docstring) — it is only the OPPORTUNITY
# given for a cooperative stop before forced cancellation (the actual,
# always-bounded mechanism — see _settle_tasks()) takes over. Avoid
# brittle timing assumptions here; nothing in this module's correctness
# depends on this exact value.
DEFAULT_SHUTDOWN_GRACE_SECONDS = 15.0

_AsyncCallable = Callable[[], Awaitable[None]]
_SyncCallable = Callable[[], None]
_PredicateCallable = Callable[[], bool]


class UnifiedServiceAdapterExited(RuntimeError):
    """
    Raised by run_unified_service() when an adapter task (Telegram polling
    or Uvicorn) terminates — by returning OR by raising — without the
    composition root itself ever having asked it to stop ("the unified
    service must not silently continue indefinitely with only the other
    adapter alive"). Never swallowed into a successful exit.
    """


class UnifiedServiceAdapterCancelled(RuntimeError):
    """
    Raised by run_unified_service() when an adapter task (Telegram polling
    or Uvicorn) is found ALREADY cancelled in the `done` set from
    `_supervise()`'s very first `asyncio.wait(..., FIRST_COMPLETED)` —
    i.e. before this composition root itself ever requested that adapter
    stop. Distinct from `UnifiedServiceAdapterExited` (a normal RETURN
    without being asked) and from ordinary sibling-teardown cancellation
    (`_settle_tasks()`'s own later force-cancel fallback, applied to
    whichever adapter DIDN'T complete first — that set is disjoint from
    the `done` set this is raised from, so normal teardown can never
    trigger this). Never swallowed into a successful exit — see
    `_select_primary_exception()`.
    """


class _AdapterBaseExceptionWrapper(BaseException):
    """
    Wraps a SystemExit/KeyboardInterrupt raised inside an adapter's own
    `run()` coroutine (Section 10) so it can be retrieved through
    ordinary `Task.exception()`/`asyncio.wait()` machinery instead of
    escaping that adapter's own Task as a raw SystemExit/KeyboardInterrupt.

    Verified directly against installed CPython 3.12's own
    asyncio/tasks.py (`Task.__step_run_and_handle_result`): unlike every
    other exception, `except (KeyboardInterrupt, SystemExit) as exc:
    super().set_exception(exc); raise` RE-RAISES immediately, straight
    out of the event loop's own callback dispatch (`_run_once()`) —
    before `asyncio.wait()`'s own waiter Future (what `_supervise()`
    awaits) ever runs that task's done-callback. A raw SystemExit from a
    CHILD task therefore crashes the whole event loop iteration outright;
    `_supervise()` would never get a chance to settle the sibling or run
    close_resources() at all. Wrapping it in a plain BaseException
    subclass here (neither SystemExit nor KeyboardInterrupt) makes it hit
    the GENERIC `except BaseException as exc: super().set_exception(exc)`
    branch instead — no re-raise, so the task completes normally from the
    scheduler's point of view and is observable through the exact same
    asyncio.wait()-based path as any other adapter failure.
    `_select_primary_exception()`/`_settle_tasks()` unwrap this back to
    `.original` before it is ever logged or (from `_supervise()`'s OWN
    top-level `raise`, not a separately-scheduled child task, so no
    further crash-through risk) propagated to the caller — the caller
    always sees the real, unchanged SystemExit/KeyboardInterrupt.
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__()
        self.original = original


def _unwrap_adapter_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, _AdapterBaseExceptionWrapper):
        return exc.original
    return exc


@dataclasses.dataclass(frozen=True)
class _Adapter:
    """One supervised adapter task plus its cooperative stop callable —
    a stable, ordered (Telegram, then Uvicorn) alternative to a bare
    `Set[Task]` so exception retrieval/logging is never dependent on set
    iteration order (independent-audit MINOR finding: the previous
    `next(iter(done)).exception()` was nondeterministic under
    simultaneous adapter failures)."""

    label: str
    task: "asyncio.Task[None]"
    stop: _SyncCallable


async def _supervised_adapter(run: _AsyncCallable, label: str, was_asked_to_stop: Callable[[], bool]) -> None:
    """
    Runs one adapter. A genuine exception from `run()` always propagates
    unchanged. A NORMAL return is only benign if a shutdown was requested
    of this adapter — by the composition root itself, or (Uvicorn only) by
    an operator signal uvicorn's own handler consumed, see this module's
    "Clean-shutdown corrective pass" docstring section (`was_asked_to_stop()`
    is checked AFTER `run()` returns, so it reflects the state at
    completion time, not at call time) — otherwise it is itself an
    unexpected termination (neither adapter may just quietly finish on its
    own).

    SystemExit/KeyboardInterrupt are caught and re-raised wrapped (see
    `_AdapterBaseExceptionWrapper`'s own docstring) — letting either
    escape THIS coroutine unchanged would crash the whole event loop
    outright, bypassing settlement/cleanup entirely, since this coroutine
    is a separately-scheduled Task's own top-level frame.
    """
    try:
        await run()
    except (SystemExit, KeyboardInterrupt) as exc:
        raise _AdapterBaseExceptionWrapper(exc) from exc
    if not was_asked_to_stop():
        raise UnifiedServiceAdapterExited(f"{label} adapter task exited unexpectedly")


# Telegram private-field coupling (installed pyTelegramBotAPI 4.36.1,
# venv/Lib/site-packages/telebot/async_telebot.py) — must be re-reviewed
# on a pyTelegramBotAPI upgrade; see _settle_telegram_pending_tasks()'s
# own docstring and requirements.txt's exact pin.
async def _settle_telegram_pending_tasks(bot_instance=None) -> None:
    """
    Settles every task in `AsyncTeleBot._pending_tasks` — a private,
    self-discarding `set[asyncio.Task]` installed pyTelegramBotAPI
    4.36.1's `_process_polling()` populates with one `process_new_
    updates(updates)` task per batch of updates it receives
    (`task.add_done_callback(self._pending_tasks.discard)` removes a task
    the instant IT completes, but says nothing about tasks still running
    right now). `bot.infinity_polling()` returning, raising, or being
    cancelled is only "polling settled" — NOT "Telegram adapter settled"
    (independent-audit MAJOR, second corrective pass): a handler task
    processing an update can still be alive at that moment, and shared
    Qdrant/DB cleanup must never begin while one is.

    Called from `_default_run_telegram()`'s own `finally`, i.e. always
    AFTER `bot.infinity_polling()` has already returned/raised/been
    cancelled for any reason — `_process_polling()`'s own `finally`
    (async_telebot.py) has therefore already closed the aiohttp session
    by this point, so a still-live handler task can no longer safely make
    further Telegram HTTP calls; every live task here is force-cancelled
    rather than given more time to run. A task that is already `done()`
    (installed `add_done_callback` fires via `call_soon`, not
    synchronously, so a task can finish and still be present in
    `_pending_tasks` for one event-loop iteration) is left alone — only
    its outcome is retrieved.

    Every task's outcome (result or exception) is retrieved here
    unconditionally, whether or not it was force-cancelled, so nothing is
    ever left as an unretrieved Task exception ("Task exception was never
    retrieved"). A handler failure is logged by type only (never its raw
    exception text) and never raised — it does not need to outrank an
    already-selected top-level adapter failure/cancellation, since
    `run_unified_service()`'s own termination is already fully determined
    by the polling coroutine's own outcome and/or the composition root's
    own supervision by the time this runs.
    """
    if bot_instance is None:
        bot_instance = bot
    pending = list(bot_instance._pending_tasks)
    if not pending:
        return
    for task in pending:
        if not task.done():
            task.cancel()
    await asyncio.wait(pending)
    for task in pending:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            logger.debug(
                "Unified service: Telegram handler task ended during adapter settlement | error_type=%s",
                type(exc).__name__,
            )


async def _default_run_telegram() -> None:
    """Reuses main.py's own setup_bot() + the same infinity_polling()
    call/parameters main.py's main() already uses — this module never
    reimplements Telegram startup or polling. The `finally` settles every
    Telegram-owned pending handler task (see
    `_settle_telegram_pending_tasks()`'s own docstring) regardless of how
    `infinity_polling()` ends — normal return, an ordinary exception, or
    cancellation — so "this adapter's task is done" always implies "no
    Telegram-owned task is still alive" by the time it propagates."""
    await setup_bot()
    try:
        await bot.infinity_polling(timeout=10, skip_pending=True)
    finally:
        await _settle_telegram_pending_tasks()


def _default_stop_telegram() -> None:
    """
    Best-effort cooperative signal ONLY — see this module's own docstring
    ("Telegram shutdown") for the full reasoning: installed
    pyTelegramBotAPI 4.36.1's `_process_polling()` can silently overwrite
    this with `_polling = True` again on its own re-entry, so correctness
    never depends on this flag actually being observed. The actual bound
    is `_settle_tasks()`'s force-cancel-then-await fallback, which is
    safe at any phase — `_process_polling()`'s own inner loop explicitly
    catches `asyncio.CancelledError` and returns cleanly through its own
    `finally` (which already closes the aiohttp session — a second
    `close_session()` call from `main.shutdown_bot()` is a documented
    no-op).
    """
    bot._polling = False


# Mirrors installed uvicorn 0.52.4's private LifespanOn.startup()
# (venv/Lib/site-packages/uvicorn/lifespan/on.py) line-for-line, plus one
# extra assignment — must be re-reviewed (and requirements.txt's exact
# `uvicorn==0.52.4` pin re-verified/updated deliberately) on any uvicorn
# upgrade.
class _TrackedLifespanOn(LifespanOn):
    """
    Installed uvicorn 0.52.4's own `LifespanOn`
    (venv/Lib/site-packages/uvicorn/lifespan/on.py) with `startup()`
    overridden to ALSO keep the `main()` task it creates as a reachable
    instance attribute (`main_lifespan_task`) instead of only a local
    variable that goes out of scope once `startup()` returns — see this
    module's own docstring ("Uvicorn's own lifespan child task") for why
    that task is otherwise unreachable from outside once forced
    cancellation needs to settle it. Every other method (`main()`,
    `shutdown()`, `send()`, `receive()`) is inherited UNCHANGED from the
    installed library; only the one extra assignment line is added here.
    """

    def __init__(self, config: "uvicorn.Config") -> None:
        super().__init__(config)
        self.main_lifespan_task: Optional["asyncio.Task[None]"] = None

    async def startup(self) -> None:  # pragma: no cover - exercised via real uvicorn.Server in tests
        self.logger.info("Waiting for application startup.")

        loop = asyncio.get_event_loop()
        self.main_lifespan_task = loop.create_task(self.main())
        startup_event = {"type": "lifespan.startup"}
        await self.receive_queue.put(startup_event)
        await self.startup_event.wait()

        if self.startup_failed or (self.error_occurred and self.config.lifespan == "on"):
            self.logger.error("Application startup failed. Exiting.")
            self.should_exit = True
        else:
            self.logger.info("Application startup complete.")


async def _settle_uvicorn_request_tasks(server: "uvicorn.Server") -> None:
    """
    Settles every task in `server.server_state.tasks` — installed uvicorn
    0.52.4's own per-ASGI-request task set. Every HTTP protocol
    implementation the installed library ships (venv/Lib/site-packages/
    uvicorn/protocols/http/httptools_impl.py's `_start_asgi_task()`, and
    identically in h11_impl.py/zttp_impl.py) follows the exact same
    `self.tasks.add(task)` / `task.add_done_callback(self.tasks.discard)`
    pattern against this SAME shared set `Server.__init__` constructs on
    `self.server_state` — mirrors `_settle_telegram_pending_tasks()`'s
    identical relationship to `AsyncTeleBot._pending_tasks`.

    `server.serve()` returning, raising, or being cancelled is only "the
    accept loop and its own cooperative shutdown path settled" — NOT
    "every request task this server ever spawned has settled" (third
    corrective pass, independent-audit MAJOR — see this module's own
    docstring, "Uvicorn owns active ASGI request tasks too"). Two distinct
    gaps land here identically:

    - Normal cooperative path: installed `Server.shutdown()` (uvicorn/
      server.py) already closes listening sockets, asks every live
      connection to stop accepting further keep-alive requests, and waits
      (bounded by `config.timeout_graceful_shutdown`) for `server_state.
      tasks` to drain — but if that bound is exceeded, it CANCELS the
      survivors (`t.cancel(...)`) and returns immediately WITHOUT ever
      awaiting them. A genuine gap in the installed library itself, not
      one this module introduces.
    - Forced path: `_settle_tasks()`'s own force-cancel fallback (or
      external cancellation reaching `_run_uncancellable()` while this
      adapter's own `run()` is still active) cancels the OUTER `server.
      serve()` Task, interrupting `Server._serve()`'s own coroutine chain
      wherever it currently is — possibly before `Server.shutdown()` (and
      therefore its socket-close/connection-shutdown/task-drain sequence)
      ever ran at all.

    Both are closed here, unconditionally, every time `run()` ends for any
    reason: first make sure no NEW request task can start — closing every
    listening socket this server bound (a harmless no-op, not an error, if
    `Server.shutdown()` already closed them) and asking every still-
    tracked connection to stop serving further keep-alive requests on its
    existing socket (mirrors `Server.shutdown()`'s own `connection.
    shutdown()` loop, equally idempotent against a connection already
    mid-shutdown) — THEN snapshot, cancel, and await every task still in
    `server_state.tasks`, never leaving one merely `.cancel()`-requested
    without ever being awaited. This synchronous prevent-new-work sequence
    (no `await` between closing sockets/connections and snapshotting
    `state.tasks`) is what satisfies "stop new work before snapshotting
    descendants" — nothing else can run on this event loop in between.

    `getattr(server, "server_state", None)` / `getattr(server, "servers",
    [])` guard `Server._serve()` having been cancelled before either
    attribute existed (mirrors `_settle_uvicorn_lifespan_task()`'s own
    defensive style) — `server_state` is always set in `Server.__init__`,
    kept for symmetry; `servers` only exists once `Server.startup()` has
    actually created a listening socket, so if startup never got that far
    there is nothing to close and `server_state.tasks` is necessarily
    still empty.

    Every settled task's outcome is retrieved here unconditionally — the
    same "no unretrieved Task exception" contract `_settle_telegram_
    pending_tasks()` already keeps for Telegram's own handler tasks. A
    request-task failure/cancellation here is expected during shutdown and
    is logged by type only, never raised — `run_unified_service()`'s own
    termination is already fully determined by the uvicorn adapter's own
    outcome by the time this runs.
    """
    state = getattr(server, "server_state", None)
    if state is None:
        return
    for sock_server in getattr(server, "servers", []):
        sock_server.close()
    for connection in list(state.connections):
        connection.shutdown()
    pending = list(state.tasks)
    if not pending:
        return
    for task in pending:
        if not task.done():
            task.cancel()
    await asyncio.wait(pending)
    for task in pending:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            logger.debug(
                "Unified service: Uvicorn request task ended during adapter settlement | error_type=%s",
                type(exc).__name__,
            )


async def _settle_uvicorn_lifespan_task(server: "uvicorn.Server") -> None:
    """
    Settles `_TrackedLifespanOn`'s tracked `main_lifespan_task` whenever
    the uvicorn adapter's own `run()` ends, for any reason (normal
    return, exception, or forced cancellation) — independent-audit MAJOR
    2. `getattr(..., None)` throughout: `server.lifespan` only exists
    once `Server._serve()` has actually started running, and
    `main_lifespan_task` only exists once `_TrackedLifespanOn.startup()`
    has actually run — both windows are safe to no-op through if `run()`
    ended before either happened.

    Cancelling an already-hung lifespan task never raises back here:
    installed uvicorn's `LifespanOn.main()` itself catches
    `BaseException` broadly (including `CancelledError`) and returns
    normally — this function's own `suppress()` is defense-in-depth for
    that, not the primary mechanism.
    """
    lifespan = getattr(server, "lifespan", None)
    task = getattr(lifespan, "main_lifespan_task", None)
    if task is None:
        return
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _build_uvicorn_adapter() -> Tuple[_AsyncCallable, _SyncCallable, _PredicateCallable]:
    """
    Builds ONE uvicorn.Server bound to web.app.create_app(owns_db_
    lifecycle=False) (the web adapter never closes the shared DB engine
    itself in unified mode) and returns (run, stop, shutdown_requested)
    closures scoped to that exact instance. `stop()` sets uvicorn's own
    cooperative `should_exit` flag — see this module's docstring for why
    that, not `Task.cancel()`, is what lets `Server.serve()` run its own
    graceful shutdown path (closing sockets, then the FastAPI lifespan's
    shutdown).

    `shutdown_requested()` reports that same flag. `stop()` is not the only
    thing that sets it: `Server.serve()`'s own `capture_signals()` installs
    a SIGINT/SIGTERM handler that sets it too, so a real operator Ctrl+C is
    consumed by uvicorn (never by this module's own supervisor) — see this
    module's "Clean-shutdown corrective pass" docstring section. Reading
    `should_exit` after `serve()` has returned is what lets the supervisor
    tell "uvicorn exited because a shutdown was requested" apart from
    "uvicorn exited on its own"; it is never set by a lifespan/bind failure
    (those raise SystemExit) or by `limit_max_requests`.

    `config.load()` is called explicitly, BEFORE `config.lifespan_class`
    is overridden to `_TrackedLifespanOn` and BEFORE `uvicorn.Server(...)`
    is constructed: `Server._serve()` would otherwise call `config.
    load()` itself on first `serve()`, which (re)computes `lifespan_class`
    from the plain string-imported default and silently undoes this
    override.
    """
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))
    config = uvicorn.Config(create_app(owns_db_lifecycle=False), host=host, port=port, access_log=False)
    config.load()
    config.lifespan_class = _TrackedLifespanOn
    server = uvicorn.Server(config)

    async def run() -> None:
        try:
            await server.serve()
        finally:
            # Request tasks settle BEFORE the lifespan task, mirroring
            # installed uvicorn's own Server.shutdown() ordering (drains
            # server_state.tasks, THEN calls lifespan.shutdown()) — see
            # _settle_uvicorn_request_tasks()'s own docstring.
            await _settle_uvicorn_request_tasks(server)
            await _settle_uvicorn_lifespan_task(server)

    def stop() -> None:
        server.should_exit = True

    def shutdown_requested() -> bool:
        return server.should_exit

    return run, stop, shutdown_requested


class _ProtectedOutcome:
    """
    Ordinary, inert data describing how `_run_protected()`'s own `coro`
    argument actually finished — success, or `failure` holding the exact
    original exception object `coro` raised (an ordinary `Exception`,
    `SystemExit`, `KeyboardInterrupt`, or a `CancelledError` `coro` raised
    on ITSELF). Generalizes `_AdapterBaseExceptionWrapper`'s containment
    technique (see that class's own docstring for the precise CPython
    asyncio mechanism) to `_run_uncancellable()`'s own protected
    operation — here a plain returned sentinel is enough, since (unlike
    `_supervised_adapter()`) nothing downstream needs this outcome to be
    retrievable through `Task.exception()`/`asyncio.wait()` machinery.
    """

    __slots__ = ("failure",)

    def __init__(self, failure: Optional[BaseException] = None) -> None:
        self.failure = failure


async def _run_protected(coro: Awaitable[None]) -> _ProtectedOutcome:
    """
    Runs `coro` and converts EVERY possible outcome — a normal return, an
    ordinary `Exception`, or a scheduler-special `SystemExit`/
    `KeyboardInterrupt`/self-raised `CancelledError` — into an ordinary
    `_ProtectedOutcome` return value instead of letting any of them
    propagate out of this coroutine's own frame. This is what
    `_run_uncancellable()` schedules as the child task: since this frame
    never lets a BaseException escape it uncaught, the Task driving it
    always completes NORMALLY from the scheduler's point of view — the
    same reasoning `_AdapterBaseExceptionWrapper`'s docstring gives for
    why a raw SystemExit from a child task's own top-level frame would
    otherwise crash the event loop's own callback dispatch before
    `_run_uncancellable()` ever got a chance to apply precedence.
    """
    try:
        await coro
    except BaseException as exc:  # noqa: BLE001 — deliberate, see docstring
        return _ProtectedOutcome(failure=exc)
    return _ProtectedOutcome()


async def _run_uncancellable(coro: Awaitable[None]) -> None:
    """
    Runs `coro` to completion in its own independent Task, immune to any
    number of cancellations delivered to the code awaiting this call.

    `asyncio.shield()` only protects the wrapped Task from being
    cancelled — the coroutine AWAITING `shield()` can still observe a
    `CancelledError` at that await point, exactly as if it had never been
    shielded. The while-loop below is what turns that into real
    immunity: every time this function's own await is interrupted, the
    protected Task itself is completely unaffected, so it just
    re-attaches to the SAME Task and keeps waiting — no matter how many
    times that happens (a second Ctrl+C, a third, ...). This is the
    mechanism `_supervise()` relies on so that settlement/cleanup are
    guaranteed to run to completion before it decides what to propagate,
    even when cancellation arrives DURING settlement itself (independent-
    audit MAJOR 1) — a bare `try/except CancelledError` around a single
    `await` cannot do this, since a cancellation arriving during the
    `except` block's own cleanup work would simply interrupt that too.

    The scheduled child task is `_run_protected(coro)`, not `coro`
    itself (independent-audit MAJOR, second corrective pass — see
    `_ProtectedOutcome`'s own docstring): `coro` raising `SystemExit`/
    `KeyboardInterrupt` directly as a bare child task's own top-level
    frame would otherwise hit CPython asyncio's Task-level special case
    for those two types and re-raise straight out of the event loop's own
    callback dispatch — before this function's `await asyncio.shield(
    task)` below ever got a chance to observe it normally. Wrapping
    means `task` here ALWAYS completes normally; `task.result()` below
    therefore never itself raises.

    `was_cancelled` tracks ONLY external cancellation of the code AWAITING
    this call (observed at the `await asyncio.shield(task)` line) — kept
    strictly distinct from `coro` raising/cancelling on its own, which is
    now ordinary data inside `outcome.failure`. If cancellation was
    observed at least once, `coro`'s own outcome is still retrieved and
    given priority first (so a genuine exception/BaseException from `coro`
    is never left unretrieved and always takes priority), and
    `asyncio.CancelledError` is then raised afterward — external
    cancellation is delayed, never silently discarded, until the
    protected work has genuinely finished.
    """
    task: "asyncio.Task[_ProtectedOutcome]" = asyncio.ensure_future(_run_protected(coro))
    was_cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            was_cancelled = True
    outcome = task.result()
    if outcome.failure is not None:
        raise outcome.failure
    if was_cancelled:
        raise asyncio.CancelledError()
    return None


async def _settle_tasks(
    pending: Set["asyncio.Task[None]"],
    stop_by_task: Dict["asyncio.Task[None]", _SyncCallable],
    grace_seconds: float,
) -> None:
    """
    Cooperative-then-forced settlement of every task in `pending` — the
    core invariant this corrective pass restores: before
    close_resources() begins, every adapter task must be definitively
    settled (done, failed, or cancelled-and-awaited). ALWAYS invoked
    through `_run_uncancellable()` by its only caller (`_supervise()`),
    so this function performs no cancellation handling of its own — it
    can safely assume it will run to completion once started, and every
    `await` below is a plain, uninterrupted one.
    """
    if not pending:
        return
    for task in pending:
        stop_by_task[task]()
    _, still_pending = await asyncio.wait(pending, timeout=grace_seconds)
    for task in still_pending:
        task.cancel()
    if still_pending:
        await asyncio.wait(still_pending)
    for task in pending:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            exc = _unwrap_adapter_exception(exc)
            logger.debug(
                "Unified service: adapter task ended during teardown | error_type=%s", type(exc).__name__
            )


def _select_primary_exception(
    adapters: List[_Adapter], done: Set["asyncio.Task[None]"]
) -> Optional[BaseException]:
    """
    Deterministic primary-failure selection (independent-audit MINOR
    finding — the previous `next(iter(done)).exception()` was
    nondeterministic set-iteration order and could leave a simultaneous
    second failure's exception unretrieved). Stable, documented order:
    Telegram, then Uvicorn (`adapters`' own construction order in
    run_unified_service()). Every completed task's exception is always
    retrieved here — never only the chosen primary's — so nothing is
    ever left as an unretrieved Task exception. A secondary simultaneous
    failure is logged by type only (never its raw exception text, which
    can carry request/token detail) and never replaces the primary.

    A task found in `done` that is ALREADY cancelled (independent-audit
    MINOR finding — second corrective pass) is itself an unexpected
    adapter termination, mapped to `UnifiedServiceAdapterCancelled`
    rather than silently skipped: `done` only ever holds tasks that were
    already complete at `_supervise()`'s very first FIRST_COMPLETED wait,
    strictly BEFORE `_settle_tasks()`'s own later force-cancel fallback
    (applied only to `pending`, the disjoint remainder) could ever have
    run — so a cancelled task here can only mean the adapter cancelled
    itself, never ordinary sibling teardown.
    """
    primary: Optional[BaseException] = None
    for adapter in adapters:
        task = adapter.task
        if task not in done:
            continue
        if task.cancelled():
            exc: BaseException = UnifiedServiceAdapterCancelled(
                f"{adapter.label} adapter task was cancelled unexpectedly"
            )
        else:
            exc = task.exception()
            if exc is None:
                continue
            exc = _unwrap_adapter_exception(exc)
        if primary is None:
            primary = exc
        else:
            logger.warning(
                "Unified service: secondary adapter failure observed alongside primary "
                "(both retrieved; only the primary propagates) | adapter=%s error_type=%s",
                adapter.label, type(exc).__name__,
            )
    return primary


async def _supervise(adapters: List[_Adapter], grace_seconds: float, close_resources: _AsyncCallable) -> None:
    """
    The full supervision algorithm — run_unified_service()'s own body.
    Every phase that MUST complete before the next one begins (settling
    every adapter task, then closing shared resources) runs through
    `_run_uncancellable()`, so no matter where or how many times this
    call's own task is cancelled, settlement and cleanup always finish,
    in order, before this function decides what to propagate.

    Final propagation precedence, most to least specific:
      1. a genuine adapter-task exception (or UnifiedServiceAdapterExited
         for an unexpected normal return) that was determined via `done`
         — covers both an ordinary FIRST_COMPLETED result and the rare
         "cancelled during the initial wait, but a task had in fact
         already raced to completion" case;
      2. external cancellation of this call's own task, observed at any
         point (the initial wait, settlement, or cleanup) before (1) was
         ever determined;
      3. a close_resources() failure — only if NEITHER (1) nor (2) ever
         occurred.
    A close_resources() failure is downgraded to a logged (type-only)
    secondary event, never silently dropped and never replacing, whenever
    (1) or (2) already apply — a primary adapter failure or external
    cancellation always outranks a shared-cleanup failure.
    """
    tasks = [adapter.task for adapter in adapters]
    stop_by_task = {adapter.task: adapter.stop for adapter in adapters}

    cancelled_during_wait = False
    try:
        done, raw_pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        pending = set(raw_pending)
    except asyncio.CancelledError:
        cancelled_during_wait = True
        pending = {t for t in tasks if not t.done()}
        done = set(tasks) - pending

    cancelled_during_settle = False
    try:
        await _run_uncancellable(_settle_tasks(pending, stop_by_task, grace_seconds))
    except asyncio.CancelledError:
        cancelled_during_settle = True

    primary_exc = _select_primary_exception(adapters, done)

    close_exc: Optional[BaseException] = None
    cancelled_during_close = False
    try:
        await _run_uncancellable(close_resources())
    except asyncio.CancelledError:
        cancelled_during_close = True
    except BaseException as exc:
        close_exc = exc

    external_cancel = cancelled_during_wait or cancelled_during_settle or cancelled_during_close

    if primary_exc is not None:
        if close_exc is not None:
            logger.warning(
                "Unified service: shared cleanup failed after a primary adapter failure — "
                "primary failure preserved | error_type=%s", type(close_exc).__name__,
            )
        raise primary_exc
    if external_cancel:
        if close_exc is not None:
            logger.warning(
                "Unified service: shared cleanup failed during external cancellation — "
                "cancellation preserved | error_type=%s", type(close_exc).__name__,
            )
        raise asyncio.CancelledError()
    if close_exc is not None:
        raise close_exc


async def run_unified_service(
    *,
    run_telegram: Optional[_AsyncCallable] = None,
    run_uvicorn: Optional[_AsyncCallable] = None,
    stop_telegram: Optional[_SyncCallable] = None,
    stop_uvicorn: Optional[_SyncCallable] = None,
    close_resources: Optional[_AsyncCallable] = None,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
) -> None:
    """
    Runs the Telegram and Uvicorn adapters concurrently in this one
    process until either one terminates unexpectedly or this call itself
    is cancelled (any number of times), then ALWAYS closes shared
    process resources exactly once before returning or raising — see
    `_supervise()`'s own docstring for the exact algorithm and
    propagation precedence. Every parameter defaults to the real
    production adapter (main.setup_bot()+bot.infinity_polling(), a real
    uvicorn.Server over web.app.create_app(owns_db_lifecycle=False),
    main.shutdown_bot()) — tests inject fakes for every one of them
    instead, so no test in this suite opens a real socket, polls real
    Telegram, or touches a real Qdrant/DB.

    Raises whatever exception caused termination (never converts an
    unexpected exit into a successful return), or
    UnifiedServiceAdapterExited if an adapter simply returned without
    being asked to (the Uvicorn adapter counts as asked once its own
    operator-signal handler requested shutdown, not only via `stop_uvicorn`
    — see this module's "Clean-shutdown corrective pass" docstring
    section). Propagates asyncio.CancelledError unchanged on external
    cancellation, after gracefully stopping both adapters first.
    """
    if run_telegram is None:
        run_telegram = _default_run_telegram
    if stop_telegram is None:
        stop_telegram = _default_stop_telegram
    # An injected `run_uvicorn` never signals an operator shutdown on its
    # own — only the default adapter's real uvicorn.Server can (see
    # _build_uvicorn_adapter()'s docstring), so its predicate is adopted
    # only together with that server's own `run`.
    uvicorn_shutdown_requested: _PredicateCallable = lambda: False
    if run_uvicorn is None or stop_uvicorn is None:
        default_run, default_stop, default_shutdown_requested = _build_uvicorn_adapter()
        if run_uvicorn is None:
            run_uvicorn = default_run
            uvicorn_shutdown_requested = default_shutdown_requested
        if stop_uvicorn is None:
            stop_uvicorn = default_stop
    if close_resources is None:
        close_resources = shutdown_bot

    # Local mutable "did the composition root itself ask this adapter to
    # stop" flags — see _supervised_adapter()'s docstring. Set the instant
    # a stop is requested (not merely after the adapter notices it), so a
    # concurrent cancellation during settlement itself still leaves the
    # flag correctly set for whichever adapter(s) were actually signalled.
    stopped = {"telegram": False, "uvicorn": False}

    def _stop_telegram() -> None:
        stopped["telegram"] = True
        stop_telegram()

    def _stop_uvicorn() -> None:
        stopped["uvicorn"] = True
        stop_uvicorn()

    telegram_task = asyncio.create_task(
        _supervised_adapter(run_telegram, "telegram", lambda: stopped["telegram"]), name="unified-telegram"
    )
    uvicorn_task = asyncio.create_task(
        _supervised_adapter(
            run_uvicorn, "uvicorn", lambda: stopped["uvicorn"] or uvicorn_shutdown_requested()
        ),
        name="unified-uvicorn",
    )
    adapters = [
        _Adapter(label="telegram", task=telegram_task, stop=_stop_telegram),
        _Adapter(label="uvicorn", task=uvicorn_task, stop=_stop_uvicorn),
    ]

    await _supervise(adapters, shutdown_grace_seconds, close_resources)


def _main(run_service: Optional[_AsyncCallable] = None) -> int:
    """
    The process's top-level exception-to-exit-code mapping — the former
    `__main__` block, extracted unchanged so it is directly testable.
    Returns the process exit code instead of calling `sys.exit()` itself.

    - Normal return, or `KeyboardInterrupt` (what `asyncio.run()` turns a
      cancelled main task into after an operator Ctrl+C — see this module's
      "Clean-shutdown corrective pass" docstring section) -> 0.
    - Any other `Exception` (including `UnifiedServiceAdapterExited`/
      `UnifiedServiceAdapterCancelled`) -> 1.
    - `SystemExit` is not caught here and propagates with its own code.
    """
    if run_service is None:
        run_service = run_unified_service
    try:
        logger.info("=" * 60)
        logger.info("Personal Python Tutor Bot - Unified Service Starting")
        logger.info("=" * 60)
        asyncio.run(run_service())
    except KeyboardInterrupt:
        logger.info("Unified service stopped by user (Ctrl+C)")
        return 0
    except Exception as e:
        # Mirrors main.py's own top-level handler: an exception here can
        # originate from a live Telegram/provider call somewhere in either
        # adapter — never log raw exception text or a traceback.
        logger.error("Unified service: fatal error | error_type=%s", type(e).__name__)
        return 1
    return 0


if __name__ == "__main__":
    # Real application startup/composition root — same "exactly once,
    # never from a reusable library module" rule utils/logging.py
    # documents for main.py's own call.
    configure_logging()
    sys.exit(_main())
