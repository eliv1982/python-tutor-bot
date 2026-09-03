"""
Stage 7A-1 corrective-pass regression tests: exception-CHAIN secrecy.

The prior pass's report claimed `raise ... from None` prevents a future
HTTP handler or logger from reaching the original provider exception. That
claim was checked and found INCOMPLETE: `from None` sets `__cause__` to
None and sets `__suppress_context__` (which only hides the chain from the
STANDARD `traceback` module's default rendering), but Python's IMPLICIT
exception chaining independently sets `__context__` to whatever exception
is currently being handled at the point of the `raise` — `from None` does
NOT clear `__context__` itself. Verified empirically (see the module
docstrings of services/text_llm.py and app/text_chat.py) before any
production code was touched.

Fixed by raising the application exception AFTER the `except` block (and,
in app/text_chat.py, after the admission-control `async with` block) has
already exited — at that point no exception is being handled, so
`__context__` is never populated at all, with no special `from` clause
needed. This file proves BOTH the ORIGINAL defect (a control/regression
proof against a bare `from None` wrapper, run inline, never touching
production code) and the FIX (against the actual production exception
types), so a future reader can see exactly what was wrong and why the fix
works.
"""

import asyncio
import traceback
import uuid

import pytest

import app.text_chat as text_chat
import config
from app.generation_limits import GenerationAdmissionController, GenerationBusyError
from app.text_chat import (
    TextChatGenerationError,
    TextChatTimeoutError,
    TextChatValidationError,
    run_text_chat,
)
from config import BotMode
from services.anthropic_client import anthropic_client
from services.openai_client import openai_client
from services.text_llm import TextGenerationTimeoutError, generate_text_response


def _assert_fully_severed(exc: BaseException, *, forbidden_sentinels=()) -> None:
    """The complete invariant required for a safe application/service
    wrapper exception (Stage 7A-1 corrective pass, Section 3): both direct
    chain attributes are None, no sentinel appears in str/repr/formatted
    traceback/instance attributes, and a recursive chain walk terminates
    at this exception itself (never reaching anything further)."""
    assert exc.__cause__ is None, f"__cause__ leaked a chained exception: {exc.__cause__!r}"
    assert exc.__context__ is None, f"__context__ leaked a chained exception: {exc.__context__!r}"

    text_repr = str(exc)
    full_repr = repr(exc)
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    for sentinel in forbidden_sentinels:
        assert sentinel not in text_repr, "sentinel leaked via str(exc)"
        assert sentinel not in full_repr, "sentinel leaked via repr(exc)"
        assert sentinel not in formatted, "sentinel leaked via traceback.format_exception"

    for attr_name, attr_value in vars(exc).items():
        assert not isinstance(attr_value, BaseException), (
            f"exception attribute {attr_name!r} holds a chained exception object: {attr_value!r}"
        )
        for sentinel in forbidden_sentinels:
            assert sentinel not in str(attr_value), f"sentinel leaked via exception attribute {attr_name!r}"

    # Recursive chain traversal: since __cause__/__context__ are already
    # asserted None above, this is an independent, explicit proof that the
    # chain truly terminates at `exc` itself rather than relying solely on
    # the two direct attribute checks.
    seen = []
    node = exc
    while node is not None:
        seen.append(node)
        node = node.__cause__ or node.__context__
    assert seen == [exc], f"exception chain unexpectedly extends beyond the wrapper itself: {seen!r}"


# ============================================================================
# A. Control proof: the ORIGINAL (defective) pattern really does leak
#    __context__, run entirely inline against toy types -- never against
#    production code. Proves the detection mechanism itself is real before
#    trusting the "fixed" proofs below.
# ============================================================================


def test_control_bare_from_none_inside_except_still_leaks_context():
    class _Original(RuntimeError):
        pass

    class _Wrapper(RuntimeError):
        pass

    def _defective():
        try:
            raise _Original("RAW_SENTINEL_control_leak")
        except _Original:
            raise _Wrapper("safe message") from None  # the ORIGINAL (defective) pattern

    with pytest.raises(_Wrapper) as excinfo:
        _defective()

    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is not None  # <- the defect: still leaks
    assert isinstance(excinfo.value.__context__, _Original)
    assert "RAW_SENTINEL_control_leak" in str(excinfo.value.__context__)


def test_control_raising_after_except_block_fully_severs_context():
    """Same scenario, but raised AFTER the except block exits -- the fix
    pattern applied to production code below."""
    class _Original(RuntimeError):
        pass

    class _Wrapper(RuntimeError):
        pass

    def _fixed():
        to_raise = False
        try:
            raise _Original("RAW_SENTINEL_control_fixed")
        except _Original:
            to_raise = True
        if to_raise:
            raise _Wrapper("safe message")

    with pytest.raises(_Wrapper) as excinfo:
        _fixed()

    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


# ============================================================================
# B. services.text_llm.TextGenerationTimeoutError
# ============================================================================


async def _hang_forever(*args, **kwargs):
    never = asyncio.Event()
    await never.wait()


async def test_text_generation_timeout_error_is_fully_severed(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(anthropic_client.client.messages, "create", _hang_forever)

    with pytest.raises(TextGenerationTimeoutError) as excinfo:
        await generate_text_response([{"role": "user", "content": "hi"}])

    _assert_fully_severed(excinfo.value)
    assert str(excinfo.value) == "Text generation timed out"


async def test_text_generation_timeout_error_is_fully_severed_for_sdk_native_openai_timeout(monkeypatch):
    """Same proof, but the timeout arrives as the installed OpenAI SDK's
    OWN native openai.APITimeoutError (Stage 7A-1 corrective pass) rather
    than via a hang + asyncio.wait_for -- the raw SDK exception's own
    request/message must never survive into __cause__/__context__ either."""
    import httpx2
    import openai as openai_sdk

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")

    async def raise_native_timeout(*args, **kwargs):
        raise openai_sdk.APITimeoutError(request=request)

    monkeypatch.setattr(openai_client.client.chat.completions, "create", raise_native_timeout)

    with pytest.raises(TextGenerationTimeoutError) as excinfo:
        await generate_text_response([{"role": "user", "content": "hi"}])

    _assert_fully_severed(excinfo.value)
    assert str(excinfo.value) == "Text generation timed out"


# ============================================================================
# C. app.text_chat.TextChatTimeoutError — plain and RAG.
# ============================================================================


async def test_text_chat_timeout_error_is_fully_severed_plain(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(openai_client.client.chat.completions, "create", _hang_forever)

    with pytest.raises(TextChatTimeoutError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.TEXT)

    _assert_fully_severed(excinfo.value)
    assert str(excinfo.value) == "Text generation timed out"
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


async def test_text_chat_timeout_error_is_fully_severed_rag(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)

    async def timing_out_rag(query, requesting_user_uuid, conversation_history=None):
        raise TextGenerationTimeoutError("Text generation timed out")

    monkeypatch.setattr("rag.query.query_knowledge_base", timing_out_rag)

    with pytest.raises(TextChatTimeoutError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.RAG)

    _assert_fully_severed(excinfo.value)
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


# ============================================================================
# D. app.text_chat.TextChatGenerationError — the case that actually wraps
#    raw provider exceptions carrying arbitrary text (the real leak risk).
# ============================================================================


async def test_text_chat_generation_error_is_fully_severed_plain(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)

    sentinel = f"RAW_PROVIDER_SENTINEL_{uuid.uuid4().hex}"

    async def failing_generate(messages, max_tokens=None):
        raise RuntimeError(f"provider said: {sentinel} (request_id=abc123, prompt echoed: {sentinel})")

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", failing_generate)

    with pytest.raises(TextChatGenerationError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.TEXT)

    _assert_fully_severed(excinfo.value, forbidden_sentinels=(sentinel,))
    assert str(excinfo.value) == "Text generation failed"
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


async def test_text_chat_generation_error_is_fully_severed_rag(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)

    sentinel = f"RAW_RAG_SENTINEL_{uuid.uuid4().hex}"

    async def failing_rag(query, requesting_user_uuid, conversation_history=None):
        raise RuntimeError(f"qdrant blew up: {sentinel}")

    monkeypatch.setattr("rag.query.query_knowledge_base", failing_rag)

    with pytest.raises(TextChatGenerationError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.RAG)

    _assert_fully_severed(excinfo.value, forbidden_sentinels=(sentinel,))
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


async def test_text_chat_generation_error_from_invalid_provider_result_is_fully_severed(monkeypatch):
    """Stage 7A-1 corrective pass: an invalid provider RESULT (not an
    exception at all -- here `None`) must also become a fully-severed
    TextChatGenerationError, with the permit released, exactly like a
    genuine provider exception would."""
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)

    async def fake_generate(messages, max_tokens=None):
        return None

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    with pytest.raises(TextChatGenerationError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.TEXT)

    _assert_fully_severed(excinfo.value)
    assert str(excinfo.value) == "Text generation failed"
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


# ============================================================================
# E. Regressions: GenerationBusyError / TextChatValidationError never wrap
#    a provider exception in the first place.
# ============================================================================


async def test_generation_busy_error_has_no_chain():
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    user_id = uuid.uuid4()

    holding = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with controller.acquire(user_id):
            holding.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await holding.wait()

    with pytest.raises(GenerationBusyError) as excinfo:
        async with controller.acquire(user_id):
            pass

    _assert_fully_severed(excinfo.value)

    release.set()
    await asyncio.wait_for(task, timeout=5)


def test_text_chat_validation_error_has_no_chain():
    with pytest.raises(TextChatValidationError) as excinfo:
        text_chat._validate_message("a" * (config.TEXT_CHAT_MAX_MESSAGE_LENGTH + 1))

    _assert_fully_severed(excinfo.value)


# ============================================================================
# F. Re-verified regressions (Section 5's "additionally re-check" list) —
#    proving the exception-chain restructuring did not change any of these
#    behaviors.
# ============================================================================


async def test_timeout_still_releases_admission_slot_and_next_request_succeeds(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(openai_client.client.chat.completions, "create", _hang_forever)

    user_id = uuid.uuid4()
    with pytest.raises(TextChatTimeoutError):
        await run_text_chat(user_id=user_id, message="hi", history=[], mode=BotMode.TEXT)

    assert controller.registry_size() == 0

    async def fake_generate(messages, max_tokens=None):
        return "ok"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)
    result = await run_text_chat(user_id=user_id, message="hi again", history=[], mode=BotMode.TEXT)
    assert result.text == "ok"


async def test_provider_failure_still_releases_admission_slot(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)

    async def failing_generate(messages, max_tokens=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", failing_generate)

    user_id = uuid.uuid4()
    with pytest.raises(TextChatGenerationError):
        await run_text_chat(user_id=user_id, message="hi", history=[], mode=BotMode.TEXT)

    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


async def test_cancellation_still_propagates_unwrapped_not_as_generation_error(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)

    started = asyncio.Event()

    async def hanging_generate(messages, max_tokens=None):
        started.set()
        never = asyncio.Event()
        await never.wait()

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", hanging_generate)

    user_id = uuid.uuid4()
    task = asyncio.create_task(run_text_chat(user_id=user_id, message="hi", history=[], mode=BotMode.TEXT))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert controller.registry_size() == 0


async def test_rag_timeout_still_makes_exactly_one_provider_call(monkeypatch):
    from types import SimpleNamespace

    import rag.query as rag_query
    from rag.index import SCOPE_PRIVATE

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)

    call_count = {"n": 0}

    async def hang_and_count(*args, **kwargs):
        call_count["n"] += 1
        never = asyncio.Event()
        await never.wait()

    monkeypatch.setattr(openai_client.client.chat.completions, "create", hang_and_count)

    fake_doc = SimpleNamespace(
        metadata={
            "source": "notes.txt", "document_id": "upload:" + "b" * 32, "chunk_index": 0,
            "scope": SCOPE_PRIVATE, "owner_user_uuid": str(uuid.uuid4()),
        },
        page_content="some retrieved passage",
    )
    monkeypatch.setattr(
        rag_query, "_validated_similarity_search",
        lambda query, requesting_user_uuid, k: [(fake_doc, 0.1)],
    )

    with pytest.raises(TextGenerationTimeoutError):
        await rag_query.query_knowledge_base("a question", str(uuid.uuid4()))

    assert call_count["n"] == 1


async def test_telegram_history_still_empty_after_timeout_provider_failure_and_cancellation(monkeypatch):
    import app.tutor as tutor
    from app.session import user_sessions

    async def _no_image_intent(text, history):
        return {"needs_generation": False}

    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    # Timeout
    async def timing_out(messages, max_tokens=None):
        raise TextGenerationTimeoutError("Text generation timed out")

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", timing_out)
    user_a = uuid.uuid4()
    response = await tutor.route_text_request(user_a, "will time out")
    assert response["error"] == "TextChatTimeoutError"
    assert user_sessions.get_history(user_a) == []

    # Provider failure
    async def failing(messages, max_tokens=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", failing)
    user_b = uuid.uuid4()
    response = await tutor.route_text_request(user_b, "will fail")
    assert response["error"] == "TextChatGenerationError"
    assert user_sessions.get_history(user_b) == []

    # Cancellation
    started = asyncio.Event()

    async def hanging(messages, max_tokens=None):
        started.set()
        never = asyncio.Event()
        await never.wait()

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", hanging)
    user_c = uuid.uuid4()
    task = asyncio.create_task(tutor.route_text_request(user_c, "will be cancelled"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert user_sessions.get_history(user_c) == []

    user_sessions.sessions.pop(user_a, None)
    user_sessions.sessions.pop(user_b, None)
    user_sessions.sessions.pop(user_c, None)
