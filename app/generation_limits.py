"""
Generation admission control (Stage 7A-1, fail-fast corrective pass).

A small, process-local concurrency gate in front of text-generation
provider calls (services/text_llm.py) — shared by every adapter that goes
through app/text_chat.py's run_text_chat()/execute_admitted_text_chat()
(Telegram today, via app/tutor.py's delegation; a future authenticated
HTTP adapter later), so no single canonical user — and no single process —
can drive an unbounded number of concurrent, expensive provider calls.
Deliberately NOT a general-purpose framework: two fixed caps
(config.TEXT_GENERATION_MAX_PER_USER, config.TEXT_GENERATION_MAX_GLOBAL),
no Redis, no PostgreSQL table/migration — in-process only, exactly the
scope Stage 7A-1 asks for. Never stores message/history content or any
other user data — only a canonical UUID key and small integer counters.

Fail-fast semantics: a second concurrent request for the SAME canonical
user, or a request arriving when the process-wide cap is already
exhausted, is REJECTED IMMEDIATELY with GenerationBusyError — never
queued, never made to wait. An unbounded per-user wait queue is not a
safe admission control for an authenticated web API (an attacker or a
merely impatient client can pile up an arbitrary number of waiters, each
eventually consuming a provider call the moment its predecessor
finishes); a fixed, bounded "busy right now" rejection is. A rejected
request never touches the provider/RAG path and never occupies (or even
attempts) a global slot — the per-user check and the global check are
evaluated together, atomically, before either is granted.

Opaque permit ownership (Stage 7A-1 corrective pass): a successful
acquire() no longer just flips a counter keyed by user_id — it returns a
unique GenerationPermit object, tracked internally (by object IDENTITY,
never by user_id alone) in this controller's OWN `_active_permits`
registry. release()/assert_permit_active() only ever mutate/inspect
counters after confirming the presented permit is a CURRENTLY ACTIVE
entry in THIS controller's own registry:
  - a fabricated permit (constructed directly, never returned by any
    acquire()) is not a key in `_active_permits` at all;
  - a permit issued by a DIFFERENT GenerationAdmissionController instance
    lives only in THAT OTHER controller's own registry, never this one's;
  - a permit already released was already popped from the registry by its
    first release.
In every one of those cases, release()/assert_permit_active() raise
GenerationPermitError (a fixed, internal-invariant-violation message —
never a UUID or other dynamic detail) WITHOUT touching `_per_user_active`
or `_global_active` at all — misuse can never corrupt admission
bookkeeping or be used to forge extra capacity.

Pinned protected use (Stage 7A-1 SECOND corrective pass): a bare
"validate, then separately trust the caller to use it promptly" contract
(the original assert_permit_active(), still kept below as a pure,
side-effect-free inspection primitive) leaves a genuine TOCTOU window
open: nothing stops a permit from being release()'d — by any thread, for
any reason, including caller misuse or a bug elsewhere — in the gap
between a caller validating it and that caller actually finishing the
protected provider work the validation was supposed to cover, which lets
a second, unrelated admission legitimately occupy the capacity the first
caller's still-in-flight work is actually relying on. begin_use()/
end_use()/use_permit() close this: begin_use() validates a permit exactly
like assert_permit_active() AND, atomically under the SAME lock
acquisition, increments a per-permit `_pinned_use_counts` entry; release()
now REFUSES (GenerationPermitError, no counters touched) whenever that
count is above zero. A permit therefore cannot be released — by anyone,
from any thread — while its protected work is genuinely still in flight,
which is exactly what makes "a new admission occupies capacity while a
stale permit still reaches protected work" impossible: capacity can never
be freed out from under work that is still relying on it. This is what
lets app/tutor.py's Telegram adapter hold ONE permit across an entire
snapshot -> generate -> commit transaction (see app/text_chat.py's
execute_admitted_text_chat(), which wraps its own provider dispatch in
use_permit()) while still being provably safe against a caller presenting
the wrong permit for the wrong user, AND against a permit being
invalidated/reassigned mid-flight — see
tests/test_stage7a1_generation_limits.py's real-two-thread race
reproduction.

Concurrency model: the entire admission decision (and every permit
issue/release) is a single, SHORT, PURELY SYNCHRONOUS state transition
(dict/int bookkeeping only — no `await` anywhere inside it) guarded by a
plain `threading.Lock`. This is deliberately NOT an `asyncio.Lock`/
`asyncio.Semaphore`: those bind their internal waiting machinery to
whichever asyncio event loop is running at the time they are first
awaited, which is exactly the kind of state that must never be allowed to
persist across event loop boundaries (see
tests/test_stage7a1_generation_limits.py's cross-event-loop and
cross-OS-thread proofs). `threading.Lock` has no relationship to asyncio
at all — acquiring/releasing it is a plain, non-awaited, near-instant
operation — so it is trivially safe to reuse from any number of different
event loops or OS threads, in any order, including the module-level
singleton below being reused test-to-test under pytest-asyncio's
per-test-function event loop model. Because every acquire()/release() is
atomic under this one lock, and every release() removes a per-user entry
the instant it would otherwise become idle, there is no window in which a
released entry could be confused with, or interfere with, a subsequent
caller's fresh acquire for the same user.

Callers must acquire exactly ONCE per top-level generation request (see
app/text_chat.py's run_text_chat()/execute_admitted_text_chat(), and
app/tutor.py's route_text_request() for the Telegram admitted-transaction
variant) — never once per individual provider call. A mode that
internally makes more than one provider call must still be wrapped by a
single outer acquire(), never a nested one.
"""

import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Dict, Iterator, Optional

import config
from utils.logging import logger

__all__ = [
    "GenerationBusyError",
    "GenerationPermitError",
    "GenerationPermit",
    "GenerationAdmissionController",
    "generation_admission_controller",
]


class GenerationBusyError(RuntimeError):
    """Raised IMMEDIATELY — never after any wait — when either the calling
    canonical user already has an active text generation in flight, or the
    process-wide concurrency cap (config.TEXT_GENERATION_MAX_GLOBAL) is
    already exhausted. A future FastAPI layer maps this to a safe 429
    ("busy, try again shortly") response. Carries a fixed, safe message
    only — never a canonical UUID, prompt, or history content."""

    def __init__(self) -> None:
        super().__init__("Text generation is at capacity right now; please try again shortly")


class GenerationPermitError(RuntimeError):
    """Raised by release()/assert_permit_active() when the presented
    permit is fabricated, already released, or was issued by a different
    GenerationAdmissionController instance (Stage 7A-1 corrective pass) —
    a fixed, internal-invariant-violation message only, never a UUID or
    other dynamic detail. This signals a CALLER BUG (misuse of the permit
    API), never ordinary user-facing behavior — admission counters are
    NEVER touched, in either direction, when this is raised."""

    def __init__(self) -> None:
        super().__init__("Invalid, foreign, or already-released generation permit")


class GenerationPermit:
    """Opaque capability returned by GenerationAdmissionController's
    acquire()/acquire_nowait() — carries no public attributes and is
    meaningful only to the specific controller instance that issued it
    (tracked purely by this object's own identity in that controller's
    private `_active_permits` registry — see release()/
    assert_permit_active()). Never construct one directly outside this
    module: a permit built by calling `GenerationPermit()` yourself is
    indistinguishable, by design, from any other object no controller has
    ever issued — every controller correctly treats it as fabricated."""

    __slots__ = ()


class GenerationAdmissionController:
    """Process-local, fail-fast admission gate — see this module's own
    docstring for the full concurrency/cancellation/cleanup/permit-
    ownership contract."""

    def __init__(self, *, max_per_user: int, max_global: int) -> None:
        # bool is a subclass of int in Python (`isinstance(True, int)` is
        # True) — `type(x) is int` is required to reject it explicitly,
        # `isinstance` alone would silently accept True/False as 1/0.
        if type(max_per_user) is not int or type(max_global) is not int:
            raise ValueError("max_per_user and max_global must be plain int values")
        if max_per_user <= 0 or max_global <= 0:
            raise ValueError("max_per_user and max_global must be positive")

        self._max_per_user = max_per_user
        self._max_global = max_global
        # Guards ONLY the synchronous state transitions below — never held
        # across an `await`, and never itself awaited. See this module's
        # own docstring for why this is threading.Lock, not asyncio.Lock.
        self._lock = threading.Lock()
        self._per_user_active: Dict[uuid.UUID, int] = {}
        self._global_active = 0
        # The single source of truth for permit ownership: maps each
        # currently-outstanding GenerationPermit THIS controller itself
        # issued to the user_id it was issued for. release()/
        # assert_permit_active() check ONLY against this dict — never
        # against `_per_user_active` directly — so a fabricated, foreign,
        # or already-released permit can never be mistaken for genuine.
        self._active_permits: Dict[GenerationPermit, uuid.UUID] = {}
        # Stage 7A-1 second corrective pass: per-permit "protected use" pin
        # count -- see begin_use()/end_use()/use_permit() below. A permit
        # with a pin count above zero cannot be released by anyone (see
        # release() below), which is what closes the validate-then-use
        # TOCTOU a bare inspection-only assert_permit_active() left open.
        self._pinned_use_counts: Dict[GenerationPermit, int] = {}

    def _try_acquire(self, user_id: uuid.UUID) -> Optional[GenerationPermit]:
        with self._lock:
            current = self._per_user_active.get(user_id, 0)
            if current >= self._max_per_user:
                return None
            if self._global_active >= self._max_global:
                return None
            self._per_user_active[user_id] = current + 1
            self._global_active += 1
            permit = GenerationPermit()
            self._active_permits[permit] = user_id
            return permit

    def acquire_nowait(self, user_id: uuid.UUID) -> GenerationPermit:
        """Synchronous, manual acquire: admits immediately if both the
        per-user and global caps have room and returns a fresh
        GenerationPermit, or raises GenerationBusyError immediately
        otherwise — never queues, never waits. The caller owns the
        returned permit's lifecycle and MUST eventually present it back to
        release() (or assert_permit_active() to merely verify it) exactly
        once. `acquire()` below is a thin async-context-manager
        convenience wrapper around this for the common single-scope case;
        this lower-level method exists for callers (Telegram's own
        admitted transaction in app/tutor.py; this module's own tests)
        that must hold a permit across a longer, explicitly managed
        span."""
        permit = self._try_acquire(user_id)
        if permit is None:
            logger.info("generation admission rejected | user_id=%s", user_id)
            raise GenerationBusyError()
        return permit

    def release(self, permit: GenerationPermit) -> None:
        """Releases a permit previously returned by acquire_nowait()/
        acquire(). Raises GenerationPermitError — WITHOUT changing any
        counter — if `permit` is not a currently active permit issued by
        THIS controller (fabricated, already released, or issued by a
        different controller instance), OR if `permit` currently has an
        active begin_use()/use_permit() pin (Stage 7A-1 second corrective
        pass): a permit's protected provider work must never have its
        capacity freed out from under it, regardless of which thread calls
        release() or when — see this module's own docstring ("Pinned
        protected use")."""
        with self._lock:
            owner = self._active_permits.get(permit)
            if owner is None:
                raise GenerationPermitError()
            if self._pinned_use_counts.get(permit, 0) > 0:
                raise GenerationPermitError()
            del self._active_permits[permit]
            current = self._per_user_active.get(owner, 0)
            if current <= 1:
                self._per_user_active.pop(owner, None)
            else:
                self._per_user_active[owner] = current - 1
            if self._global_active > 0:
                self._global_active -= 1

    def assert_permit_active(self, permit: GenerationPermit, *, user_id: uuid.UUID) -> None:
        """Raises GenerationPermitError if `permit` is not a currently
        active permit issued by THIS controller for EXACTLY `user_id` —
        never changes any counter either way (pure inspection, no pinning).
        Kept as a standalone, side-effect-free check for any caller that
        genuinely only needs a point-in-time inspection; a caller that is
        about to perform PROTECTED PROVIDER WORK under `permit` (e.g.
        app.text_chat.execute_admitted_text_chat()) must use begin_use()/
        use_permit() instead — a bare inspection here does not, by itself,
        stop the permit from being concurrently released out from under
        that work (see this module's own docstring, "Pinned protected
        use", for the TOCTOU this distinction exists to close)."""
        with self._lock:
            owner = self._active_permits.get(permit)
        if owner is None or owner != user_id:
            raise GenerationPermitError()

    def begin_use(self, permit: GenerationPermit, *, user_id: uuid.UUID) -> None:
        """Validates `permit` exactly like assert_permit_active() (raises
        GenerationPermitError, without touching any counter, for a
        fabricated/foreign/already-released permit or one that does not
        cover EXACTLY `user_id`) AND, atomically within the SAME lock
        acquisition, marks it as genuinely in use for protected provider
        work by incrementing its `_pinned_use_counts` entry. While that
        count is above zero, release() refuses to release `permit` (see
        release() above) — this is what closes the validate-then-use
        TOCTOU a bare inspection-only check left open (Stage 7A-1 second
        corrective pass). Must be paired with an eventual end_use() call
        for the SAME permit; use_permit() below does this automatically,
        including on exception/cancellation."""
        with self._lock:
            owner = self._active_permits.get(permit)
            if owner is None or owner != user_id:
                raise GenerationPermitError()
            self._pinned_use_counts[permit] = self._pinned_use_counts.get(permit, 0) + 1

    def end_use(self, permit: GenerationPermit) -> None:
        """Ends one begin_use()/use_permit() pin on `permit`. Never touches
        admission counters itself — only release() does that, and only
        once every pin on `permit` has ended. Raises GenerationPermitError
        — without touching anything — if `permit` currently has no active
        pin (a caller bug: end_use() called without a matching, still-open
        begin_use())."""
        with self._lock:
            count = self._pinned_use_counts.get(permit, 0)
            if count <= 0:
                raise GenerationPermitError()
            if count == 1:
                del self._pinned_use_counts[permit]
            else:
                self._pinned_use_counts[permit] = count - 1

    @contextmanager
    def use_permit(self, permit: GenerationPermit, *, user_id: uuid.UUID) -> Iterator[None]:
        """Synchronous context manager bracketing a permit's protected
        provider work (Stage 7A-1 second corrective pass) — see
        begin_use()/end_use() above. Deliberately a PLAIN, non-async
        context manager: entering/exiting it only ever performs near-
        instant, purely synchronous, lock-protected bookkeeping (never an
        `await`), so wrapping it around awaited code — e.g. `with
        controller.use_permit(permit, user_id=user_id): await
        provider_call()` — never holds `self._lock` across that await;
        only begin_use()'s and end_use()'s own brief, independent lock
        acquisitions ever do. Ends the pin on every exit — success,
        exception, or the caller's own cancellation unwinding through it —
        exactly like a `finally` block would."""
        self.begin_use(permit, user_id=user_id)
        try:
            yield
        finally:
            self.end_use(permit)

    def registry_size(self) -> int:
        """Diagnostic/test-only: number of canonical users currently
        holding an active generation permit. Never used by production call
        sites; never grows for a user once their generation ends."""
        with self._lock:
            return len(self._per_user_active)

    def global_active_count(self) -> int:
        """Diagnostic/test-only: current global active-generation count."""
        with self._lock:
            return self._global_active

    @asynccontextmanager
    async def acquire(self, user_id: uuid.UUID) -> AsyncIterator[GenerationPermit]:
        """Async context manager: admits immediately if both the per-user
        and global caps have room, yielding the fresh GenerationPermit —
        or raises GenerationBusyError immediately otherwise — never
        queues, never waits. Releases the SAME permit exactly once on
        exit (success, exception, or cancellation). See this module's own
        docstring for the full contract."""
        permit = self.acquire_nowait(user_id)
        try:
            yield permit
        finally:
            self.release(permit)


# Module-level singleton (same convention as services.anthropic_client.
# anthropic_client / app.session.user_sessions) — construction is pure
# in-memory bookkeeping, no I/O, so it is safe to build at import time.
generation_admission_controller = GenerationAdmissionController(
    max_per_user=config.TEXT_GENERATION_MAX_PER_USER,
    max_global=config.TEXT_GENERATION_MAX_GLOBAL,
)
