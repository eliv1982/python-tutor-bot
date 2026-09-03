"""
Adapter-independent, stateless text-chat core (Stage 7A-1, corrective pass).

The one place plain-chat/RAG text generation actually happens, shared by
every adapter: today only the Telegram adapter (app/tutor.py's
route_text_request() delegates here — see that module's own docstring),
later a future authenticated HTTP adapter that will call run_text_chat()
directly. Deliberately stateless and adapter-agnostic:

- Takes an explicit, already-resolved canonical `user_id: uuid.UUID` (used
  ONLY for private-RAG ownership scoping and generation admission control
  — never resolved here; a caller must already have authenticated/resolved
  it, exactly like every other app/*.py module in this codebase). Strictly
  validated as a genuine `uuid.UUID` INSTANCE — an int, bool, arbitrary
  object, or even a syntactically valid UUID string is rejected before
  admission/provider are ever touched (Stage 7A-1 corrective pass: string
  coercion is a transport-layer concern for a future HTTP adapter, never
  performed here).
- Takes an explicit `message` and an explicit, BOUNDED `history` — never
  reads or writes app.session.user_sessions, never reads cookies or web
  sessions, never determines identity itself. This is what lets a future
  FastAPI route call this directly without ever touching Telegram's own
  ephemeral conversation state, and what lets tests prove Telegram's
  history and a hypothetical web caller's history can never bleed into
  each other for the same canonical UUID (see
  tests/test_stage7a1_text_chat_core.py's isolation proof).
- Takes an explicit `mode` — validated against config.BotMode.ALL, the
  SINGLE canonical allowlist (never a second, locally-invented list that
  could drift from it). Anything else (a typo, None, an unrecognized
  string) is rejected outright, fail-closed — never silently treated as
  "plain chat by default". Only the confirmed-valid, already-canonical
  mode value is ever logged; a rejected raw value never reaches a log
  line.
- Always adds its OWN trusted, server-owned tutor system prompt for the
  plain-chat path — never accepts a client-supplied system prompt (the RAG
  path's own system prompt lives in rag/query.py, unchanged by this
  module, and is likewise never client-suppliable). A "system"-role entry
  anywhere in caller-supplied `history` is rejected by validation, never
  silently accepted as a second system message.
- Has no image-generation capability at all (no import of
  services.image_generation anywhere in this module) — image-intent
  detection/generation stays exclusively Telegram-adapter-level routing in
  app/tutor.py, which runs BEFORE ever delegating here.
- Never persists conversation history anywhere (no PostgreSQL table, no
  process-global state keyed by user_id beyond the admission-control
  registry in app/generation_limits.py, which stores no message content).
- Never shares a mutable object with its caller in either direction:
  validated history is immediately converted into an immutable snapshot
  (a tuple of frozen `_HistoryMessage` records — never the caller's own
  list/dicts), and every provider/RAG call is handed BRAND NEW plain
  dicts built fresh from that snapshot, never a reused/aliased dict or
  list. A provider that mutates the list/dicts it was handed (accidentally
  or adversarially) can therefore never corrupt the caller's own `history`
  argument or this module's own snapshot (Stage 7A-1 corrective pass —
  see tests/test_stage7a1_text_chat_core.py's mutation-isolation proof).
  TextChatResult is itself an immutable (frozen) dataclass, so a caller
  can never mutate a returned result's fields either.
- Validates the ACTUAL provider/RAG result before ever returning or
  wrapping it: only a genuine, non-empty, non-whitespace-only `str` is a
  successful result. `None`, a list, a dict, or an empty/whitespace-only
  string is treated as a generation failure (a sanitized
  TextChatGenerationError), never allowed to propagate outward as a raw
  `TypeError`/`AttributeError` from downstream code that assumed a string
  (Stage 7A-1 corrective pass).

Returns an adapter-neutral TextChatResult on success, or raises one of
this module's own application-layer exceptions on failure — never a raw
provider exception, and never Telegram/HTTP-specific data.

Exception taxonomy — four distinct, non-overlapping application-layer
exceptions, so a future FastAPI layer can map each to its own HTTP status
without inspecting exception text:
  - TextChatValidationError  -> 422 (bad request shape/size)
  - GenerationBusyError      -> 429 (admission-control rejection — see
                                 app/generation_limits.py; NEVER caught or
                                 rewrapped here)
  - TextChatTimeoutError     -> 504/503 (provider call exceeded
                                 config.TEXT_GENERATION_TIMEOUT_SECONDS)
  - TextChatGenerationError  -> 502/503 (any other provider/RAG failure,
                                 including an invalid/empty provider
                                 result)
`asyncio.CancelledError` is a BaseException and is never caught by any
`except Exception`/`except <one of the above>` clause anywhere in this
module — it always propagates unmodified, exactly like every other
awaited call in this codebase. Every application exception raised here
uses `from None` (see execute_admitted_text_chat()'s own except clauses),
deliberately severing the exception chain so a future HTTP/logging layer
that introspects `__cause__`/`__context__` (e.g.
`traceback.format_exception`) can never surface the original provider
exception's own text/body — only this module's own fixed, safe message is
ever reachable that way.

Public surface (Stage 7A-1 corrective pass — split for Telegram
linearizability, see app/tutor.py's route_text_request()):
  - run_text_chat(): the self-contained, single-call convenience entry
    point (validates scalars, validates+normalizes history, acquires its
    own admission permit, executes, releases) — the one a future
    stateless HTTP adapter will call directly.
  - validate_scalar_request(): validates ONLY user_id/message/mode (no
    history) — lets a caller fail fast on bad scalar input BEFORE it ever
    requests an admission permit, and therefore before it ever needs to
    snapshot any adapter-owned history.
  - validate_and_normalize_history(): validates a caller-supplied history
    list and converts it into this module's own immutable snapshot type.
  - execute_admitted_text_chat(): runs the actual generation dispatch
    given already-validated scalars/history and an ALREADY-ACQUIRED
    admission permit — verifies the permit (app.generation_limits.
    GenerationAdmissionController.assert_permit_active()) but neither
    acquires nor releases it; the caller owns that lifecycle. This is
    what lets Telegram's adapter hold ONE permit across its own
    history-snapshot -> generate -> atomic-commit transaction (see
    app/tutor.py) instead of run_text_chat()'s own acquire-execute-release
    happening as a separate, narrower scope that a second concurrent
    request could slip in around.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import config
from app.generation_limits import (
    GenerationBusyError,
    GenerationPermit,
    generation_admission_controller,
)
from config import BotMode
from services import text_llm
from utils.logging import logger

__all__ = [
    "TextChatValidationError",
    "TextChatTimeoutError",
    "TextChatGenerationError",
    "GenerationBusyError",
    "TextChatResult",
    "TUTOR_SYSTEM_PROMPT",
    "run_text_chat",
    "validate_scalar_request",
    "validate_and_normalize_history",
    "execute_admitted_text_chat",
    "generation_admission_controller",
]

_ALLOWED_HISTORY_ROLES = frozenset({"user", "assistant"})
_ALLOWED_HISTORY_KEYS = frozenset({"role", "content"})
_CANONICAL_MODES = frozenset(BotMode.ALL)

# The SAME trusted tutor persona previously inlined in app/tutor.py's plain-
# chat branch (Stage 7A-1: moved here, unchanged, so Telegram's delegated
# call produces byte-identical output to before this refactor). Never
# client-suppliable — this is the only system prompt the plain-chat path
# ever sends.
TUTOR_SYSTEM_PROMPT = (
    "Ты — персональный тьютор по Python. Отвечай на русском, кратко и по делу. "
    "Не используй разметку markdown — только обычный текст. Примеры кода пиши с отступом, "
    "без ** и без обратных кавычек. Объясняй концепции и лучшие практики."
)


class TextChatValidationError(ValueError):
    """Raised for a structurally or size-invalid request (Stage 7A-1) — a
    non-UUID user_id, an unsupported/unrecognized mode, an unsupported
    history role, a non-plain-dict history entry, a history entry with
    extra/missing keys, non-string content, or the message/history bounds
    (config.TEXT_CHAT_MAX_MESSAGE_LENGTH / TEXT_CHAT_MAX_HISTORY_
    MESSAGES / TEXT_CHAT_MAX_HISTORY_TOTAL_CHARS) exceeded. A future
    FastAPI layer maps this to a safe HTTP 422 — never silently accepted.
    Carries only a fixed, safe message; never echoes client-supplied
    content."""


class TextChatTimeoutError(RuntimeError):
    """Raised specifically when the underlying provider call did not
    complete within config.TEXT_GENERATION_TIMEOUT_SECONDS (Stage 7A-1)
    — wraps services.text_llm.TextGenerationTimeoutError exclusively
    (itself now raised uniformly for an asyncio.wait_for timeout AND
    every official native timeout exception of the installed OpenAI/
    Anthropic SDKs — see services/text_llm.py's own docstring); every
    OTHER provider/RAG failure becomes TextChatGenerationError instead,
    never this. Distinct on purpose so a future HTTP layer can map a
    timeout to a gateway/service-timeout status rather than a generic
    upstream-failure one. Fixed, safe message only."""


class TextChatGenerationError(RuntimeError):
    """Raised when the underlying provider generation call fails for any
    reason OTHER than a timeout (a provider SDK exception, an
    AnthropicResponseError, a RAG retrieval/indexing failure), OR when the
    provider/RAG call "succeeded" but returned something other than a
    genuine, non-empty, non-whitespace-only string (Stage 7A-1 corrective
    pass — see this module's own docstring). Deliberately a fixed, safe
    message; never the original provider exception text, and never the
    invalid result's own value/type spelled out (Stage 1D privacy-safe
    logging convention). Distinct from GenerationBusyError
    (admission-control rejection, never wrapped here — see
    app/generation_limits.py) and from TextChatTimeoutError (see that
    class's own docstring), so a future HTTP layer can map all three to
    different status codes."""


@dataclass(frozen=True)
class TextChatResult:
    """Adapter-neutral outcome of a successful run_text_chat()/
    execute_admitted_text_chat() call — no Telegram types, no HTTP types.
    A frozen dataclass: a caller can never mutate `text`/`mode` after the
    fact (attempting to raises dataclasses.FrozenInstanceError)."""
    text: str
    mode: str


@dataclass(frozen=True)
class _HistoryMessage:
    """One immutable, already-validated history entry — the ONLY form
    this module's own internal snapshot ever stores. Never the caller's
    own dict object; see _normalize_history()."""
    role: str
    content: str


def _validate_user_id(user_id: Any) -> None:
    # Deliberately `isinstance`, not `type() is`: a genuine uuid.UUID
    # instance is accepted regardless of whether it is exactly `uuid.UUID`
    # or an application-defined subclass, but an int/bool/str/arbitrary
    # object is not — bool is not a uuid.UUID subclass, so no separate
    # bool special-case is needed here (unlike the int constructor checks
    # in app/generation_limits.py). A syntactically valid UUID *string*
    # (e.g. str(uuid.uuid4())) is deliberately rejected too: coercing
    # transport-layer input into a uuid.UUID is a future HTTP adapter's
    # own job, never this module's.
    if not isinstance(user_id, uuid.UUID):
        raise TextChatValidationError("user_id must be a uuid.UUID instance")


def _validate_mode(mode: Any) -> None:
    # Stage 7A-1 second corrective pass: `type(mode) is not str` is checked
    # BEFORE the `_CANONICAL_MODES` membership test, deliberately never
    # `isinstance` (which a str SUBCLASS would still pass) and deliberately
    # ordered first. A bare `mode not in _CANONICAL_MODES` alone is not
    # enough: frozenset membership is decided by `__hash__`/`__eq__`, so a
    # crafted object whose `__eq__`/`__hash__` mimic a canonical string
    # (e.g. compares equal to and hashes like "text") would satisfy it
    # without genuinely BEING one — and an unhashable object would blow up
    # the `in` check itself with a raw, unsanitized TypeError instead of
    # this module's own validation exception. Requiring a genuine `str`
    # first closes both: it can never be satisfied by `__eq__`/`__hash__`
    # spoofing, and the membership check is only ever attempted against a
    # value Python can already hash safely.
    if type(mode) is not str or mode not in _CANONICAL_MODES:
        raise TextChatValidationError("mode must be one of the canonical BotMode values")


def _validate_message(message: Any) -> None:
    if not isinstance(message, str):
        raise TextChatValidationError("message must be a string")
    if not message.strip():
        raise TextChatValidationError("message must not be empty or whitespace-only")
    if len(message) > config.TEXT_CHAT_MAX_MESSAGE_LENGTH:
        raise TextChatValidationError("message exceeds the maximum allowed length")


def _validate_history(history: Any) -> None:
    if not isinstance(history, list):
        raise TextChatValidationError("history must be a list")
    if len(history) > config.TEXT_CHAT_MAX_HISTORY_MESSAGES:
        raise TextChatValidationError("history exceeds the maximum allowed number of messages")

    total_chars = 0
    for entry in history:
        # `type(entry) is dict`, not `isinstance` — a dict SUBCLASS (or a
        # proxy) is rejected outright (Stage 7A-1 corrective pass): only a
        # genuine plain dict, whose .keys()/[] behavior cannot have been
        # overridden, is trusted here.
        if type(entry) is not dict:
            raise TextChatValidationError("history entry must be a plain dict with 'role' and 'content'")
        if set(entry.keys()) != _ALLOWED_HISTORY_KEYS:
            raise TextChatValidationError("history entry must have exactly the 'role' and 'content' keys")
        role = entry["role"]
        # Stage 7A-1 second corrective pass: same rationale as
        # _validate_mode() above — a genuine `str` is required BEFORE the
        # `_ALLOWED_HISTORY_ROLES` membership test, closing both an
        # equality/hash-spoofing role object and an unhashable one raising
        # a raw TypeError instead of TextChatValidationError.
        if type(role) is not str or role not in _ALLOWED_HISTORY_ROLES:
            raise TextChatValidationError("history entry has an unsupported role")
        content = entry["content"]
        if not isinstance(content, str):
            raise TextChatValidationError("history entry content must be a string")
        total_chars += len(content)

    if total_chars > config.TEXT_CHAT_MAX_HISTORY_TOTAL_CHARS:
        raise TextChatValidationError("history exceeds the maximum allowed combined content length")


def _normalize_history(history: List[Dict[str, Any]]) -> Tuple[_HistoryMessage, ...]:
    """Converts an already-validated `history` list into this module's own
    immutable snapshot — a tuple of frozen _HistoryMessage records built
    from FRESH string values, never referencing the caller's own dict
    objects. Mutating the caller's original list/dicts afterward (or
    mutating anything downstream code does with a list built FROM this
    snapshot — see _fresh_messages()) can therefore never change this
    snapshot's own content."""
    return tuple(_HistoryMessage(role=entry["role"], content=entry["content"]) for entry in history)


def _fresh_messages(snapshot: Tuple[_HistoryMessage, ...]) -> List[Dict[str, str]]:
    """Builds a BRAND NEW list of BRAND NEW plain dicts from `snapshot`,
    every time it is called — never a cached/reused list or dict object.
    This is what is actually handed to a provider/RAG call: if that call
    mutates the list it received (append/remove) or a dict inside it
    (reassigns a key), neither the caller's original history argument nor
    this module's own `snapshot` is affected, since nothing here is the
    same object."""
    return [{"role": m.role, "content": m.content} for m in snapshot]


def validate_scalar_request(*, user_id: uuid.UUID, message: str, mode: str) -> None:
    """Validates ONLY `user_id`/`message`/`mode` — deliberately NOT
    `history` (Stage 7A-1 corrective pass). Lets a caller (see
    app/tutor.py's route_text_request()) fail fast on bad scalar input
    BEFORE it ever requests an admission permit, and therefore before it
    ever needs to snapshot any adapter-owned history — a permit is a
    scarce, contended resource that a malformed request must never
    occupy, even briefly. Raises TextChatValidationError. Never logs the
    raw `mode` value: validation either passes silently or raises before
    any logging happens."""
    _validate_user_id(user_id)
    _validate_mode(mode)
    _validate_message(message)


def validate_and_normalize_history(history: List[Dict[str, Any]]) -> Tuple[_HistoryMessage, ...]:
    """Validates a caller-supplied `history` list (see _validate_history()'s
    own docstring for the exact fail-closed shape/bounds contract) and
    converts it into this module's own immutable snapshot type. Raises
    TextChatValidationError."""
    _validate_history(history)
    return _normalize_history(history)


async def execute_admitted_text_chat(
    *,
    user_id: uuid.UUID,
    message: str,
    history_snapshot: Tuple[_HistoryMessage, ...],
    mode: str,
    permit: GenerationPermit,
) -> TextChatResult:
    """
    Runs the actual plain-chat/RAG generation dispatch, given
    already-validated `user_id`/`message`/`mode` and an already-validated-
    and-normalized `history_snapshot`, assuming the caller ALREADY HOLDS a
    valid admission permit for `user_id` (see
    app.generation_limits.GenerationAdmissionController.acquire()/
    acquire_nowait()). Neither acquires nor releases that permit — the
    caller owns that lifecycle, which is exactly what lets app/tutor.py's
    Telegram adapter hold ONE permit across its own history-snapshot ->
    generate -> atomic-commit transaction instead of a narrower
    acquire-execute-release scope a second concurrent request could slip
    in around. Stage 7A-1 second corrective pass: the entire dispatch below
    runs inside generation_admission_controller.use_permit(permit,
    user_id=user_id) rather than a bare, point-in-time
    assert_permit_active() check — this PINS the permit for the whole
    provider-call window, so it cannot be released (by anyone, from any
    thread) while this function's own protected work is still relying on
    it, closing the validate-then-use TOCTOU a bare inspection-only check
    would otherwise leave open (see app/generation_limits.py's own
    docstring, "Pinned protected use").

    Validates the ACTUAL provider/RAG result before ever returning it:
    only a genuine, non-empty, non-whitespace-only `str` is treated as
    success — see this module's own docstring.

    Raises:
        GenerationPermitError: `permit` does not genuinely cover
            `user_id` on this exact controller (a caller bug — never
            ordinary user-facing behavior).
        TextChatTimeoutError: the provider call did not complete within
            config.TEXT_GENERATION_TIMEOUT_SECONDS.
        TextChatGenerationError: the provider/RAG call failed for any
            other reason, or returned an invalid/empty result.
    """
    # Exception-chain secrecy (verified empirically, not merely assumed —
    # see services/text_llm.py's own docstring for the full proof):
    # `raise TextChatTimeoutError(...) from None` / `raise
    # TextChatGenerationError(...) from None`, if raised from *inside* the
    # except clauses below, would each still leave `__context__` pointing
    # at the original provider exception — `from None` only sets
    # `__cause__` and `__suppress_context__` (which merely hides that
    # chain from the STANDARD `traceback` module's default formatting, not
    # from direct `__context__` introspection by a future HTTP/logging
    # layer). Fixed here by only recording a fixed OUTCOME MARKER (a
    # string, never the original exception object) inside each except
    # clause, and raising the actual application exception AFTER the
    # try/except AND after use_permit()'s own `with` block has exited — so
    # no exception is being handled at the point of the raise, giving both
    # `__cause__` and `__context__` as None with no special `from` clause
    # needed.
    outcome_error: Optional[str] = None
    response_text: Optional[str] = None

    with generation_admission_controller.use_permit(permit, user_id=user_id):
        logger.debug(
            "text_chat core dispatch | user_id=%s, mode=%s, history_len=%s, message_len=%s",
            user_id, mode, len(history_snapshot), len(message),
        )

        try:
            if mode == BotMode.RAG:
                from rag.query import query_knowledge_base

                raw_result = await query_knowledge_base(message, str(user_id), _fresh_messages(history_snapshot))
            else:
                messages = (
                    [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}]
                    + _fresh_messages(history_snapshot)
                    + [{"role": "user", "content": message}]
                )
                raw_result = await text_llm.generate_text_response(messages)

            if not isinstance(raw_result, str) or not raw_result.strip():
                # Never log the invalid value/type's actual content — only
                # that it happened and its Python type name (safe, fixed
                # vocabulary, never provider/user data).
                logger.error(
                    "text_chat core generation returned an invalid result | user_id=%s, mode=%s, result_type=%s",
                    user_id, mode, type(raw_result).__name__,
                )
                outcome_error = "generation"
            else:
                response_text = raw_result
        except GenerationBusyError:
            # Defense-in-depth only: neither branch above touches admission
            # control, so this should be unreachable in practice. Kept so a
            # future refactor that accidentally introduced a nested acquire()
            # inside the dispatch above would fail loudly (GenerationBusyError
            # propagating unwrapped) rather than being silently rewrapped as a
            # generic TextChatGenerationError by the broad `except Exception`
            # below.
            raise
        except text_llm.TextGenerationTimeoutError:
            logger.error(
                "text_chat core generation timed out | user_id=%s, mode=%s",
                user_id, mode,
            )
            outcome_error = "timeout"
        except Exception as e:
            # Wraps services.text_llm.generate_text_response() (both
            # providers) and rag.query.query_knowledge_base() for any
            # failure other than a timeout — never log or raise raw
            # provider exception text (Stage 1D privacy guarantee). Only
            # `type(e).__name__` (a safe string) is ever extracted from `e`
            # — `e` itself is never assigned to anything that outlives this
            # except block.
            logger.error(
                "text_chat core generation failed | user_id=%s, mode=%s, error_type=%s",
                user_id, mode, type(e).__name__,
            )
            outcome_error = "generation"
    # use_permit()'s own `with` block has now exited -- the permit's pin is
    # released (though the permit itself is still held by the caller, per
    # this function's own contract). No exception is being handled below
    # this point, so raising here (see the two `if outcome_error == ...`
    # branches) keeps both __cause__ and __context__ as None.

    if outcome_error == "timeout":
        raise TextChatTimeoutError("Text generation timed out")
    if outcome_error == "generation":
        raise TextChatGenerationError("Text generation failed")

    logger.info(
        "text_chat core done | user_id=%s, mode=%s, response_len=%s",
        user_id, mode, len(response_text),
    )
    return TextChatResult(text=response_text, mode=mode)


async def run_text_chat(
    *,
    user_id: uuid.UUID,
    message: str,
    history: List[Dict[str, Any]],
    mode: str,
) -> TextChatResult:
    """
    Generate a tutor reply for `message`, given an explicit, already-
    bounded `history` — the self-contained, single-call convenience entry
    point (validates scalars, validates+normalizes history, acquires its
    own admission permit around the whole per-mode dispatch — including
    RAG's own possible retrieval-then-fallback pair of provider calls
    (rag/query.py) — executes, then releases). The one a future stateless
    HTTP adapter will call directly; Telegram's own admitted transaction
    (app/tutor.py's route_text_request()) instead calls
    validate_scalar_request()/validate_and_normalize_history()/
    execute_admitted_text_chat() directly so ONE permit can also cover its
    own history snapshot and atomic history commit — see this module's
    own docstring.

    Args:
        user_id: Canonical internal user UUID — used only for private-RAG
            ownership scoping (rag.query.query_knowledge_base) and
            per-user generation admission control. Never re-derived here;
            must be a genuine uuid.UUID instance.
        message: The current user message (bounded — see
            config.TEXT_CHAT_MAX_MESSAGE_LENGTH).
        history: Explicit prior turns, role in {"user", "assistant"} only,
            string content only (bounded — see config.TEXT_CHAT_MAX_
            HISTORY_MESSAGES / TEXT_CHAT_MAX_HISTORY_TOTAL_CHARS). Never
            mutated by this function, and never shared by reference with
            what any provider/RAG call actually receives.
        mode: A canonical config.BotMode.ALL value. BotMode.RAG routes
            through rag.query.query_knowledge_base(); every other
            canonical mode routes through the plain-chat path with this
            module's own trusted system prompt.

    Returns:
        TextChatResult on success.

    Raises:
        TextChatValidationError: user_id/message/mode/history is
            structurally or size invalid.
        GenerationBusyError: the process-wide or per-user concurrency cap
            is already exhausted (app/generation_limits.py) — raised
            immediately, before this function ever touches the provider/RAG
            path; never caught or rewrapped here, so a future HTTP layer
            can map it distinctly (429).
        TextChatTimeoutError: the provider call did not complete within
            config.TEXT_GENERATION_TIMEOUT_SECONDS.
        TextChatGenerationError: the provider/RAG call failed for any other
            reason, or returned an invalid/empty result.
    """
    validate_scalar_request(user_id=user_id, message=message, mode=mode)
    history_snapshot = validate_and_normalize_history(history)

    async with generation_admission_controller.acquire(user_id) as permit:
        return await execute_admitted_text_chat(
            user_id=user_id,
            message=message,
            history_snapshot=history_snapshot,
            mode=mode,
            permit=permit,
        )
