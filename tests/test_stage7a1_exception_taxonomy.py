"""
Stage 7A-1 corrective-pass regression tests: the explicit, non-overlapping
application-layer exception taxonomy exposed by app.text_chat.run_text_chat()
— see that module's own docstring for the full contract a future FastAPI
layer will rely on:
  - TextChatValidationError  -> future HTTP 422
  - GenerationBusyError      -> future HTTP 429 (never caught/rewrapped)
  - TextChatTimeoutError     -> future HTTP gateway/service timeout
  - TextChatGenerationError  -> future HTTP upstream/service error
  - asyncio.CancelledError   -> always propagates unmodified

Each test proves BOTH the correct exception type is raised AND that no
raw provider sentinel text, canonical UUID, or history content ever
appears in the raised exception's own message.
"""

import asyncio
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
from secrecy_helpers import assert_no_secret_leak


def _fresh_controller(monkeypatch, *, max_per_user=1, max_global=4) -> GenerationAdmissionController:
    controller = GenerationAdmissionController(max_per_user=max_per_user, max_global=max_global)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)
    return controller


# ============================================================================
# A. TextChatValidationError — invalid history/message.
# ============================================================================


async def test_invalid_message_raises_text_chat_validation_error():
    message = "a" * (config.TEXT_CHAT_MAX_MESSAGE_LENGTH + 1)
    with pytest.raises(TextChatValidationError):
        await run_text_chat(user_id=uuid.uuid4(), message=message, history=[], mode=BotMode.TEXT)


async def test_invalid_history_role_raises_text_chat_validation_error():
    with pytest.raises(TextChatValidationError):
        await run_text_chat(
            user_id=uuid.uuid4(), message="hi", history=[{"role": "system", "content": "x"}], mode=BotMode.TEXT
        )


# ============================================================================
# B. GenerationBusyError — same-user and global — never rewrapped.
# ============================================================================


async def test_same_user_busy_raises_generation_busy_error_unwrapped(monkeypatch):
    _fresh_controller(monkeypatch, max_per_user=1, max_global=4)

    holding = asyncio.Event()
    release = asyncio.Event()

    async def hanging_generate(messages, max_tokens=None):
        holding.set()
        never = asyncio.Event()
        await never.wait()

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", hanging_generate)

    user_id = uuid.uuid4()
    task = asyncio.create_task(run_text_chat(user_id=user_id, message="first", history=[], mode=BotMode.TEXT))
    await holding.wait()

    with pytest.raises(GenerationBusyError) as excinfo:
        await run_text_chat(user_id=user_id, message="second", history=[], mode=BotMode.TEXT)

    # Never rewrapped as TextChatGenerationError/TextChatTimeoutError.
    assert type(excinfo.value) is GenerationBusyError
    assert str(user_id) not in str(excinfo.value)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_global_busy_raises_generation_busy_error_unwrapped(monkeypatch):
    controller = _fresh_controller(monkeypatch, max_per_user=1, max_global=2)

    holding_count = {"n": 0}
    all_holding = asyncio.Event()
    release = asyncio.Event()

    async def hanging_generate(messages, max_tokens=None):
        holding_count["n"] += 1
        if holding_count["n"] == 2:
            all_holding.set()
        await release.wait()
        return "reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", hanging_generate)

    user_a, user_b, user_c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    task_a = asyncio.create_task(run_text_chat(user_id=user_a, message="a", history=[], mode=BotMode.TEXT))
    task_b = asyncio.create_task(run_text_chat(user_id=user_b, message="b", history=[], mode=BotMode.TEXT))
    await asyncio.wait_for(all_holding.wait(), timeout=5)

    with pytest.raises(GenerationBusyError) as excinfo:
        await run_text_chat(user_id=user_c, message="c", history=[], mode=BotMode.TEXT)

    assert type(excinfo.value) is GenerationBusyError
    assert str(user_c) not in str(excinfo.value)
    assert controller.global_active_count() == 2  # the rejected third attempt never touched it

    release.set()
    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5)


# ============================================================================
# C. TextChatTimeoutError — plain (both providers) and RAG.
# ============================================================================


async def _hang_forever(*args, **kwargs):
    never = asyncio.Event()
    await never.wait()


async def test_plain_timeout_raises_text_chat_timeout_error(monkeypatch):
    from services.anthropic_client import anthropic_client

    _fresh_controller(monkeypatch)
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(anthropic_client.client.messages, "create", _hang_forever)

    with pytest.raises(TextChatTimeoutError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.TEXT)

    assert str(excinfo.value) == "Text generation timed out"
    # Chain fully severed (Stage 7A-1 corrective pass) -- see
    # tests/test_stage7a1_exception_chain_secrecy.py for the full proof
    # this holds for every raise site, including __context__, which a bare
    # `from None` alone does NOT clear.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


async def test_plain_openai_timeout_raises_text_chat_timeout_error(monkeypatch):
    from services.openai_client import openai_client

    _fresh_controller(monkeypatch)
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "TEXT_GENERATION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(openai_client.client.chat.completions, "create", _hang_forever)

    with pytest.raises(TextChatTimeoutError):
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.TEXT)


async def test_rag_timeout_raises_text_chat_timeout_error(monkeypatch):
    _fresh_controller(monkeypatch)

    async def hanging_query_knowledge_base(query, requesting_user_uuid, conversation_history=None):
        import services.text_llm as text_llm_module

        raise text_llm_module.TextGenerationTimeoutError("Text generation timed out")

    monkeypatch.setattr("rag.query.query_knowledge_base", hanging_query_knowledge_base)

    with pytest.raises(TextChatTimeoutError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.RAG)

    assert str(excinfo.value) == "Text generation timed out"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


# ============================================================================
# D. TextChatGenerationError — any other provider/RAG failure.
# ============================================================================


async def test_provider_exception_raises_text_chat_generation_error_not_timeout(monkeypatch, caplog):
    _fresh_controller(monkeypatch)
    marker = "SECRET_TOKEN_LEAK_MARKER_taxonomy"

    async def failing_generate(messages, max_tokens=None):
        raise RuntimeError(marker)

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", failing_generate)

    import logging

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TextChatGenerationError) as excinfo:
            await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.TEXT)

    assert type(excinfo.value) is TextChatGenerationError
    assert str(excinfo.value) == "Text generation failed"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert_no_secret_leak(marker, str(excinfo.value), caplog=caplog)


async def test_rag_provider_exception_raises_text_chat_generation_error(monkeypatch, caplog):
    _fresh_controller(monkeypatch)
    marker = "SECRET_TOKEN_LEAK_MARKER_rag_taxonomy"

    async def failing_query(query, requesting_user_uuid, conversation_history=None):
        raise RuntimeError(marker)

    monkeypatch.setattr("rag.query.query_knowledge_base", failing_query)

    import logging

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TextChatGenerationError) as excinfo:
            await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.RAG)

    assert_no_secret_leak(marker, str(excinfo.value), caplog=caplog)


# ============================================================================
# E. asyncio.CancelledError propagates unmodified.
# ============================================================================


async def test_caller_cancellation_propagates_as_cancelled_error_not_wrapped(monkeypatch):
    _fresh_controller(monkeypatch)

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
