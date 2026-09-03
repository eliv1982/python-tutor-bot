"""
Stage 7A-1 regression tests: services.text_llm's centralized generation
timeout (config.TEXT_GENERATION_TIMEOUT_SECONDS).

Scope covered:
- plain Anthropic path times out and raises TextGenerationTimeoutError;
- plain OpenAI path times out and raises TextGenerationTimeoutError;
- the RAG answer-generation path (rag.query._generate_rag_response) times
  out the same way;
- a timed-out provider call is genuinely CANCELLED, not merely abandoned
  (proven by observing the hanging call's own cancellation, not by timing);
- app.generation_limits permits are released after a timeout;
- a subsequent request (same user) succeeds cleanly afterward.

Every test monkeypatches config.TEXT_GENERATION_TIMEOUT_SECONDS to a small
value so the suite runs fast and deterministically; the "hang" itself is a
provider call awaiting an asyncio.Event that is never set, so only
asyncio.wait_for's own cancellation can ever end it — never a race against
a real sleep duration.

Section A2 covers the OTHER timeout source (Stage 7A-1 corrective pass):
the installed OpenAI/Anthropic SDK's OWN native timeout exception classes
(`openai.APITimeoutError`, `anthropic.APITimeoutError` — confirmed real by
reading the installed `openai._exceptions`/`anthropic._exceptions` modules
directly, never guessed) raised DIRECTLY by the (mocked) SDK call, never
via a hang + asyncio.wait_for's own timeout — proving services/text_llm.py
normalizes this path too, not just asyncio.TimeoutError.
"""

import asyncio
import uuid

import anthropic
import httpx2
import openai
import pytest

import app.text_chat as text_chat
import config
import services.text_llm as text_llm
from app.generation_limits import GenerationAdmissionController
from app.text_chat import TextChatTimeoutError, run_text_chat
from config import BotMode
from services.anthropic_client import anthropic_client
from services.openai_client import openai_client
from services.text_llm import TextGenerationTimeoutError


def _openai_native_timeout_error() -> "openai.APITimeoutError":
    """A genuine openai.APITimeoutError instance — its real constructor
    requires an httpx2.Request (confirmed by reading the installed
    openai._exceptions module: `class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx2.Request) -> None: ...`)."""
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APITimeoutError(request=request)


def _anthropic_native_timeout_error() -> "anthropic.APITimeoutError":
    """A genuine anthropic.APITimeoutError instance — same shape as
    openai.APITimeoutError above, confirmed against the installed
    anthropic._exceptions module."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APITimeoutError(request=request)


async def _hang_forever(*args, **kwargs):
    """Never completes on its own -- only cancellation (asyncio.wait_for's
    own timeout handling) can end it. Proves the timeout genuinely cancels
    the underlying provider call rather than merely racing a sleep."""
    never = asyncio.Event()
    try:
        await never.wait()
    except asyncio.CancelledError:
        raise


# ============================================================================
# A. Plain-chat path, both providers.
# ============================================================================


async def test_plain_anthropic_path_times_out(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(anthropic_client.client.messages, "create", _hang_forever)

    with pytest.raises(TextGenerationTimeoutError):
        await text_llm.generate_text_response([{"role": "user", "content": "hi"}])


async def test_plain_openai_path_times_out(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(openai_client.client.chat.completions, "create", _hang_forever)

    with pytest.raises(TextGenerationTimeoutError):
        await text_llm.generate_text_response([{"role": "user", "content": "hi"}])


async def test_timeout_cancels_the_underlying_provider_call_not_just_abandons_it(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)

    was_cancelled = {"flag": False}

    async def hang_and_record_cancellation(*args, **kwargs):
        never = asyncio.Event()
        try:
            await never.wait()
        except asyncio.CancelledError:
            was_cancelled["flag"] = True
            raise

    monkeypatch.setattr(openai_client.client.chat.completions, "create", hang_and_record_cancellation)

    with pytest.raises(TextGenerationTimeoutError):
        await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    assert was_cancelled["flag"] is True


async def test_timeout_exception_never_carries_raw_provider_text(monkeypatch, caplog):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(openai_client.client.chat.completions, "create", _hang_forever)

    import logging

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TextGenerationTimeoutError) as excinfo:
            await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    assert str(excinfo.value) == "Text generation timed out"


# ============================================================================
# A2. SDK-native timeout classes (Stage 7A-1 corrective pass): the
#    installed SDK's OWN official timeout exception, raised DIRECTLY by
#    the (mocked) SDK call -- never via a hang + asyncio.wait_for's own
#    timeout -- proving services/text_llm.py normalizes this path too.
# ============================================================================


async def test_plain_openai_native_sdk_timeout_raises_text_generation_timeout_error(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")

    async def raise_native_timeout(*args, **kwargs):
        raise _openai_native_timeout_error()

    monkeypatch.setattr(openai_client.client.chat.completions, "create", raise_native_timeout)

    with pytest.raises(TextGenerationTimeoutError) as excinfo:
        await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    assert str(excinfo.value) == "Text generation timed out"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


async def test_plain_anthropic_native_sdk_timeout_raises_text_generation_timeout_error(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")

    async def raise_native_timeout(*args, **kwargs):
        raise _anthropic_native_timeout_error()

    monkeypatch.setattr(anthropic_client.client.messages, "create", raise_native_timeout)

    with pytest.raises(TextGenerationTimeoutError) as excinfo:
        await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    assert str(excinfo.value) == "Text generation timed out"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


async def test_native_sdk_timeout_exception_never_carries_raw_sdk_text(monkeypatch, caplog):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")

    async def raise_native_timeout(*args, **kwargs):
        raise _openai_native_timeout_error()

    monkeypatch.setattr(openai_client.client.chat.completions, "create", raise_native_timeout)

    import logging

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TextGenerationTimeoutError) as excinfo:
            await text_llm.generate_text_response([{"role": "user", "content": "hi"}])

    assert str(excinfo.value) == "Text generation timed out"
    # The SDK's own message text ("Request timed out.") must never appear
    # raw in logs.
    assert "Request timed out" not in caplog.text


async def test_rag_openai_native_sdk_timeout_does_not_trigger_a_hidden_second_provider_call(monkeypatch):
    """Same one-attempt proof as test_rag_timeout_does_not_trigger_a_hidden_
    second_provider_call above, but the timeout arrives as the SDK's own
    native exception class rather than via a hang."""
    from types import SimpleNamespace

    import rag.query as rag_query
    from rag.index import SCOPE_PRIVATE

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")

    call_count = {"n": 0}

    async def raise_native_timeout(*args, **kwargs):
        call_count["n"] += 1
        raise _openai_native_timeout_error()

    monkeypatch.setattr(openai_client.client.chat.completions, "create", raise_native_timeout)

    fake_doc = SimpleNamespace(
        metadata={
            "source": "notes.txt", "document_id": "upload:" + "c" * 32, "chunk_index": 0,
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

    assert call_count["n"] == 1, "an SDK-native timeout on the primary call must not be followed by a fallback retry"


async def test_text_chat_core_plain_openai_native_sdk_timeout_raises_text_chat_timeout_error(monkeypatch):
    fresh_controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", fresh_controller)
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")

    async def raise_native_timeout(*args, **kwargs):
        raise _openai_native_timeout_error()

    monkeypatch.setattr(openai_client.client.chat.completions, "create", raise_native_timeout)

    user_id = uuid.uuid4()
    with pytest.raises(TextChatTimeoutError) as excinfo:
        await run_text_chat(user_id=user_id, message="hi", history=[], mode=BotMode.TEXT)

    assert str(excinfo.value) == "Text generation timed out"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert fresh_controller.registry_size() == 0
    assert fresh_controller.global_active_count() == 0

    # Subsequent request (same user) succeeds cleanly -- the permit was
    # genuinely released.
    async def fake_generate(messages, max_tokens=None):
        return "ok"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)
    result = await run_text_chat(user_id=user_id, message="hi again", history=[], mode=BotMode.TEXT)
    assert result.text == "ok"


async def test_text_chat_core_plain_anthropic_native_sdk_timeout_raises_text_chat_timeout_error(monkeypatch):
    fresh_controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", fresh_controller)
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")

    async def raise_native_timeout(*args, **kwargs):
        raise _anthropic_native_timeout_error()

    monkeypatch.setattr(anthropic_client.client.messages, "create", raise_native_timeout)

    with pytest.raises(TextChatTimeoutError):
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.TEXT)

    assert fresh_controller.registry_size() == 0
    assert fresh_controller.global_active_count() == 0


async def test_text_chat_core_rag_openai_native_sdk_timeout_raises_text_chat_timeout_error(monkeypatch):
    from types import SimpleNamespace

    import rag.query as rag_query
    from rag.index import SCOPE_PRIVATE

    fresh_controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", fresh_controller)
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")

    async def raise_native_timeout(*args, **kwargs):
        raise _openai_native_timeout_error()

    monkeypatch.setattr(openai_client.client.chat.completions, "create", raise_native_timeout)

    fake_doc = SimpleNamespace(
        metadata={
            "source": "notes.txt", "document_id": "upload:" + "d" * 32, "chunk_index": 0,
            "scope": SCOPE_PRIVATE, "owner_user_uuid": str(uuid.uuid4()),
        },
        page_content="some retrieved passage",
    )
    monkeypatch.setattr(
        rag_query, "_validated_similarity_search",
        lambda query, requesting_user_uuid, k: [(fake_doc, 0.1)],
    )

    with pytest.raises(TextChatTimeoutError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.RAG)

    assert str(excinfo.value) == "Text generation timed out"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert fresh_controller.registry_size() == 0


# ============================================================================
# B. RAG answer-generation path.
# ============================================================================


async def test_rag_answer_generation_path_times_out(monkeypatch):
    from rag.query import _generate_rag_response

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(openai_client.client.chat.completions, "create", _hang_forever)

    with pytest.raises(TextGenerationTimeoutError):
        await _generate_rag_response(query="what is a list", context="[some context]")


async def test_rag_timeout_does_not_trigger_a_hidden_second_provider_call(monkeypatch):
    """Stage 7A-1 corrective pass: query_knowledge_base()'s own broad
    except-Exception fallback (rag/query.py) must NOT catch a timeout on
    the PRIMARY answer-generation call and silently retry through
    _fallback_response() — that would be a hidden second provider call,
    and could turn a genuine timeout into a late "success". Proven by
    counting real calls to the provider client: retrieval finds a result
    (so the primary path, not the "no results" fallback, is taken), the
    primary generation call hangs and times out, and the provider client
    must have been invoked EXACTLY ONCE by the time query_knowledge_base()
    itself raises -- never a second time for _fallback_response()."""
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
            "source": "notes.txt", "document_id": "upload:" + "a" * 32, "chunk_index": 0,
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

    assert call_count["n"] == 1, "the timed-out primary call must not have been followed by a fallback retry"


# ============================================================================
# C. Permit release + subsequent request, through the full text-chat core.
# ============================================================================


async def test_permit_is_released_after_timeout_and_next_request_succeeds(monkeypatch):
    fresh_controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", fresh_controller)

    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(anthropic_client.client.messages, "create", _hang_forever)

    user_id = uuid.uuid4()

    with pytest.raises(TextChatTimeoutError):
        await run_text_chat(user_id=user_id, message="hi", history=[], mode=BotMode.TEXT)

    assert fresh_controller.registry_size() == 0
    assert fresh_controller.global_active_count() == 0

    # A normal (non-hanging) provider response now succeeds for the SAME
    # user -- proving the timed-out request's permits were genuinely
    # released, not stuck.
    async def fake_generate(messages, max_tokens=None):
        return "ok"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)
    result = await run_text_chat(user_id=user_id, message="hi again", history=[], mode=BotMode.TEXT)
    assert result.text == "ok"
    assert fresh_controller.registry_size() == 0
