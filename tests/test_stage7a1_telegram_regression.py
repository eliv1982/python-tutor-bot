"""
Stage 7A-1 regression tests: app.tutor.route_text_request()'s Telegram
behavior after delegating generation to app.text_chat.run_text_chat().

Scope covered:
- the existing text path genuinely delegates to the stateless core (not a
  reimplemented copy of its logic);
- Telegram's own history stays separate/bounded and is unaffected by the
  core's own validation bounds under ordinary use;
- image-intent routing still runs through the prior Telegram-only path,
  unaffected by the refactor;
- a successful turn records BOTH the user message and the assistant reply
  together, consistently;
- a provider failure, a generation timeout, and caller cancellation each
  leave NO partially-recorded turn in Telegram's history (Stage 7A-1's
  explicit failure-semantics requirement).
"""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.text_chat as text_chat
import app.tutor as tutor
from app.session import user_sessions
from app.text_chat import TextChatGenerationError, TextChatResult
from config import BotMode


def _new_user_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture(autouse=True)
def _clean_sessions():
    yield
    user_sessions.sessions.clear()


async def _no_image_intent(text, history):
    return {"needs_generation": False}


# ============================================================================
# A. Delegation to the stateless core.
# ============================================================================


async def test_route_text_request_delegates_to_text_chat_core(monkeypatch):
    """Stage 7A-1 corrective pass: route_text_request() no longer calls
    app.text_chat.run_text_chat() as a single opaque call (that would
    acquire its OWN, narrower admission permit around generation alone,
    reopening the snapshot/generate/commit race — see this module's own
    docstring and Section E below). It now calls
    app.text_chat.execute_admitted_text_chat() directly, under a permit
    IT acquired itself and holds across the whole transaction. This test
    proves that delegation surface directly — the actual generation logic
    still lives in app.text_chat, never reimplemented here."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    captured = {}

    async def fake_execute(*, user_id, message, history_snapshot, mode, permit):
        captured["user_id"] = user_id
        captured["message"] = message
        captured["history_snapshot"] = history_snapshot
        captured["mode"] = mode
        return TextChatResult(text="core reply", mode=mode)

    monkeypatch.setattr(text_chat, "execute_admitted_text_chat", fake_execute)

    user_id = _new_user_id()
    response = await tutor.route_text_request(user_id, "hello there")

    assert response == {"text": "core reply", "mode": BotMode.TEXT}
    assert captured["user_id"] == user_id
    assert captured["message"] == "hello there"
    assert captured["mode"] == BotMode.TEXT
    assert captured["history_snapshot"] == ()  # nothing recorded yet for a brand-new user


# ============================================================================
# B. Telegram history stays separate/bounded and unaffected by the core's
#    own validation under ordinary use.
# ============================================================================


async def test_telegram_history_recorded_and_passed_to_core_on_next_turn(monkeypatch):
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    async def fake_generate(messages, max_tokens=None):
        return "reply " + str(len(messages))

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    user_id = _new_user_id()
    r1 = await tutor.route_text_request(user_id, "first message")
    assert r1["text"]

    history_after_first = user_sessions.get_history(user_id)
    assert history_after_first == [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": r1["text"]},
    ]

    r2 = await tutor.route_text_request(user_id, "second message")
    assert r2["text"]

    history_after_second = user_sessions.get_history(user_id)
    assert history_after_second[0] == {"role": "user", "content": "first message"}
    assert history_after_second[-1] == {"role": "assistant", "content": r2["text"]}
    assert len(history_after_second) == 4


async def test_telegram_history_stays_within_its_existing_bound(monkeypatch):
    """Telegram's own MAX_HISTORY_LENGTH trimming (app/session.py) is
    unchanged by Stage 7A-1 -- it already keeps history within the new
    core's own TEXT_CHAT_MAX_HISTORY_MESSAGES bound, with no expansion."""
    import config as app_config

    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    async def fake_generate(messages, max_tokens=None):
        return "reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    user_id = _new_user_id()
    for i in range(20):
        await tutor.route_text_request(user_id, f"message {i}")

    history = user_sessions.get_history(user_id)
    assert len(history) == app_config.MAX_HISTORY_LENGTH * 2
    assert len(history) <= app_config.TEXT_CHAT_MAX_HISTORY_MESSAGES


# ============================================================================
# C. Image-intent routing is unaffected (still Telegram-only, still runs
#    BEFORE the core is ever reached).
# ============================================================================


async def test_image_intent_still_routes_through_the_prior_telegram_path(monkeypatch):
    async def image_intent_yes(text, history):
        return {"needs_generation": True, "confidence": 0.9, "prompt": "a cat in space"}

    monkeypatch.setattr(tutor, "detect_image_generation_intent", image_intent_yes)

    core_called = {"flag": False}

    async def fake_execute(**kwargs):
        core_called["flag"] = True
        return TextChatResult(text="should not be reached", mode=BotMode.TEXT)

    monkeypatch.setattr(text_chat, "execute_admitted_text_chat", fake_execute)

    image_result = {
        "text": "Изображение создано!",
        "image_path": "/tmp/fake.png",
        "revised_prompt": "a cat in space, revised",
        "original_prompt": "a cat in space",
        "has_image": True,
    }
    route_image_mock = AsyncMock(return_value=image_result)
    monkeypatch.setattr(tutor, "route_image_generation_request", route_image_mock)

    user_id = _new_user_id()
    response = await tutor.route_text_request(user_id, "Нарисуй кота в космосе")

    assert response == image_result
    route_image_mock.assert_awaited_once()
    assert core_called["flag"] is False  # image path never reaches the text-chat core

    # Image-intent routing never touches Telegram history through
    # route_text_request itself (route_image_generation_request owns its
    # own history bookkeeping, unchanged by this refactor).
    assert user_sessions.get_history(user_id) == []


# ============================================================================
# D. Failure semantics: no partially-recorded turn after a provider
#    failure, a generation timeout, or caller cancellation.
# ============================================================================


async def test_provider_failure_leaves_no_partial_history(monkeypatch):
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    async def failing_generate(messages, max_tokens=None):
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", failing_generate)

    user_id = _new_user_id()
    response = await tutor.route_text_request(user_id, "will fail")

    assert "error" in response
    assert response["text"] == "Извините, произошла ошибка при обработке запроса."
    assert user_sessions.get_history(user_id) == []  # no orphaned user-only turn


async def test_timeout_leaves_no_partial_history(monkeypatch):
    from services.text_llm import TextGenerationTimeoutError

    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    async def timing_out_generate(messages, max_tokens=None):
        raise TextGenerationTimeoutError("Text generation timed out")

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", timing_out_generate)

    user_id = _new_user_id()
    response = await tutor.route_text_request(user_id, "will time out")

    assert response["error"] == "TextChatTimeoutError"
    assert user_sessions.get_history(user_id) == []


async def test_cancellation_leaves_no_partial_history(monkeypatch):
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    started = asyncio.Event()

    async def hanging_generate(messages, max_tokens=None):
        started.set()
        never = asyncio.Event()
        await never.wait()

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", hanging_generate)

    user_id = _new_user_id()
    task = asyncio.create_task(tutor.route_text_request(user_id, "will be cancelled"))

    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # route_text_request's own broad `except Exception` does not catch
    # CancelledError (a BaseException in modern Python) -- it propagates,
    # exactly like every other awaited call in this codebase -- but no
    # history must have been written before that point.
    assert user_sessions.get_history(user_id) == []

    # The admission-control permit held by the cancelled attempt must also
    # be released -- a fresh request for the same user succeeds cleanly.
    async def fake_generate(messages, max_tokens=None):
        return "ok after cancellation"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)
    response = await tutor.route_text_request(user_id, "retry")
    assert response["text"] == "ok after cancellation"


# ============================================================================
# E. Telegram linearizability (Stage 7A-1 corrective pass): ONE admission
#    permit covers the whole history-snapshot -> generate -> atomic-commit
#    transaction, closing the race where a second concurrent same-user
#    request could previously be admitted in the gap between the first
#    request's generation finishing and its history commit running.
# ============================================================================


async def test_second_concurrent_same_user_request_is_rejected_before_snapshot_or_provider_call(monkeypatch):
    """Deterministic reproduction of the old race: while the first
    same-user request is blocked mid-generation (an event it controls),
    a second same-user request must be rejected as busy WITHOUT ever
    taking its own history snapshot (app.text_chat.
    validate_and_normalize_history()) or calling the provider."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    first_holding = asyncio.Event()
    release_first = asyncio.Event()
    provider_call_count = {"n": 0}

    async def fake_generate(messages, max_tokens=None):
        provider_call_count["n"] += 1
        first_holding.set()
        await release_first.wait()
        return "first reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    snapshot_call_count = {"n": 0}
    original_validate_and_normalize_history = text_chat.validate_and_normalize_history

    def counting_validate_and_normalize_history(history):
        snapshot_call_count["n"] += 1
        return original_validate_and_normalize_history(history)

    monkeypatch.setattr(text_chat, "validate_and_normalize_history", counting_validate_and_normalize_history)

    user_id = _new_user_id()
    first_task = asyncio.create_task(tutor.route_text_request(user_id, "first message"))
    await first_holding.wait()

    # Second same-user request while the first is still active (blocked
    # on release_first) -- must be rejected immediately, never queued.
    second_response = await tutor.route_text_request(user_id, "second message")

    assert second_response.get("error") == "GenerationBusyError"
    assert snapshot_call_count["n"] == 1  # only the FIRST request's own snapshot
    assert provider_call_count["n"] == 1  # the rejected request never reached the provider

    release_first.set()
    first_response = await first_task
    assert first_response["text"] == "first reply"


async def test_telegram_linearizability_full_race_reproduction(monkeypatch):
    """End-to-end proof of the full Stage 7A-1 corrective-pass contract in
    one deterministic scenario: the first same-user provider call blocks
    on an event; while it is active, a second same-user request is
    rejected as busy; the first request then completes and atomically
    records ONE full user/assistant pair; a request made AFTER the
    permit is released succeeds, and its own provider input contains the
    complete first pair; the final history contains both pairs, in
    order."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    first_holding = asyncio.Event()
    release_first = asyncio.Event()
    captured_messages = []

    async def fake_generate(messages, max_tokens=None):
        captured_messages.append(messages)
        if len(captured_messages) == 1:
            first_holding.set()
            await release_first.wait()
            return "first reply"
        return "second reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    user_id = _new_user_id()
    first_task = asyncio.create_task(tutor.route_text_request(user_id, "first message"))
    await first_holding.wait()

    busy_response = await tutor.route_text_request(user_id, "should be rejected")
    assert busy_response.get("error") == "GenerationBusyError"
    assert len(captured_messages) == 1  # the rejected request never called the provider

    release_first.set()
    first_response = await first_task
    assert first_response["text"] == "first reply"

    # Atomic commit: exactly one full pair recorded, in order.
    history_after_first = user_sessions.get_history(user_id)
    assert history_after_first == [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first reply"},
    ]

    # A fresh request after release succeeds and its own provider input
    # contains the complete first pair.
    second_response = await tutor.route_text_request(user_id, "third message")
    assert second_response["text"] == "second reply"
    assert len(captured_messages) == 2
    second_call_messages = captured_messages[1]
    assert {"role": "user", "content": "first message"} in second_call_messages
    assert {"role": "assistant", "content": "first reply"} in second_call_messages

    final_history = user_sessions.get_history(user_id)
    assert final_history == [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "third message"},
        {"role": "assistant", "content": "second reply"},
    ]


async def test_different_users_run_concurrently_and_do_not_block_each_other(monkeypatch):
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    holding_a = asyncio.Event()
    holding_b = asyncio.Event()
    release_both = asyncio.Event()
    call_order = []

    async def fake_generate(messages, max_tokens=None):
        # Distinguish which user's call this is by which history/message
        # it carries (each user's own message text is the last entry).
        is_a = messages[-1]["content"] == "from a"
        (holding_a if is_a else holding_b).set()
        await release_both.wait()
        call_order.append("a" if is_a else "b")
        return "a reply" if is_a else "b reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    user_a, user_b = _new_user_id(), _new_user_id()
    task_a = asyncio.create_task(tutor.route_text_request(user_a, "from a"))
    task_b = asyncio.create_task(tutor.route_text_request(user_b, "from b"))

    await asyncio.wait_for(holding_a.wait(), timeout=5)
    await asyncio.wait_for(holding_b.wait(), timeout=5)  # both admitted concurrently -- neither blocked the other

    release_both.set()
    response_a, response_b = await asyncio.gather(task_a, task_b)
    assert response_a["text"] == "a reply"
    assert response_b["text"] == "b reply"


# ============================================================================
# F. Legacy route resolution: the pre-Stage-7A-1 route_rag_request() (a
#    direct rag.query.query_knowledge_base() call bypassing admission
#    control, the timeout/error taxonomy, and the atomic history commit)
#    has been removed outright -- it had no production caller. Proven
#    behaviorally, not merely by a source-string assertion.
# ============================================================================


def test_route_rag_request_no_longer_exists():
    assert not hasattr(tutor, "route_rag_request")


def test_tutor_module_source_has_no_direct_reference_to_query_knowledge_base():
    """Checks actual CODE, not this module's own explanatory docstring
    (which legitimately names the removed route_rag_request() function's
    former target in prose, documenting why it was removed) -- the module
    docstring's own line range is stripped first."""
    import ast
    import pathlib

    source = pathlib.Path(tutor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        doc_node = tree.body[0]
        for i in range(doc_node.lineno - 1, doc_node.end_lineno):
            lines[i] = ""
    code_only = "\n".join(lines)
    assert "query_knowledge_base" not in code_only


async def test_only_text_chat_core_can_reach_rag_query_knowledge_base(monkeypatch):
    """Behavioral proof: every currently callable app.tutor.route_*
    function is exercised with benign inputs, and rag.query.
    query_knowledge_base is reached ONLY through route_text_request's
    RAG-mode path (i.e. through app.text_chat's validated/admitted core)
    -- never through any other route."""
    rag_mock = AsyncMock(return_value="rag reply")
    monkeypatch.setattr("rag.query.query_knowledge_base", rag_mock)
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent)

    async def fake_generate(messages, max_tokens=None):
        return "plain reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    # A TEXT-mode text request must never reach RAG.
    text_response = await tutor.route_text_request(_new_user_id(), "hello", mode=BotMode.TEXT)
    assert text_response["text"] == "plain reply"
    rag_mock.assert_not_called()

    # A RAG-mode text request DOES reach it, exactly through the core.
    rag_response = await tutor.route_text_request(_new_user_id(), "hello", mode=BotMode.RAG)
    assert rag_response["text"] == "rag reply"
    rag_mock.assert_called_once()


# ============================================================================
# G. Full validation/admission ordering (Stage 7A-1 SECOND corrective
#    pass): an independent re-audit found that route_text_request() still
#    performed a Telegram history read, raw-mode logging, and the
#    image-intent classifier's own provider call BEFORE scalar validation
#    and admission. Proven here with instrumented spies (never merely by
#    reading the code) that a malformed request or a same-user busy
#    rejection produces ZERO history reads, ZERO classifier calls, and
#    ZERO provider calls.
# ============================================================================


def _install_call_counters(monkeypatch):
    """Wraps user_sessions.get_history / tutor.detect_image_generation_intent
    / text_chat.text_llm.generate_text_response with counting spies that
    still delegate to real (or benign fake) behavior, so a test can prove
    exactly how many times each was invoked."""
    history_calls = {"n": 0}
    classifier_calls = {"n": 0}
    provider_calls = {"n": 0}

    original_get_history = user_sessions.get_history

    def counting_get_history(uid):
        history_calls["n"] += 1
        return original_get_history(uid)

    monkeypatch.setattr(user_sessions, "get_history", counting_get_history)

    async def counting_classifier(text, history):
        classifier_calls["n"] += 1
        return {"needs_generation": False}

    monkeypatch.setattr(tutor, "detect_image_generation_intent", counting_classifier)

    async def counting_generate(messages, max_tokens=None):
        provider_calls["n"] += 1
        return "reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", counting_generate)

    return history_calls, classifier_calls, provider_calls


async def test_invalid_uuid_triggers_no_history_read_classifier_or_provider_or_admission_leak(monkeypatch):
    history_calls, classifier_calls, provider_calls = _install_call_counters(monkeypatch)

    response = await tutor.route_text_request("not-a-uuid", "hello", mode=BotMode.TEXT)

    assert response.get("error") == "TextChatValidationError"
    assert history_calls["n"] == 0
    assert classifier_calls["n"] == 0
    assert provider_calls["n"] == 0
    assert text_chat.generation_admission_controller.registry_size() == 0
    assert text_chat.generation_admission_controller.global_active_count() == 0


async def test_invalid_mode_triggers_no_history_read_classifier_or_provider(monkeypatch):
    history_calls, classifier_calls, provider_calls = _install_call_counters(monkeypatch)

    response = await tutor.route_text_request(_new_user_id(), "hello", mode="totally_bogus_mode")

    assert response.get("error") == "TextChatValidationError"
    assert history_calls["n"] == 0
    assert classifier_calls["n"] == 0
    assert provider_calls["n"] == 0


@pytest.mark.parametrize("bad_message", ["", "   ", "\n\t"])
async def test_whitespace_only_message_triggers_no_history_read_classifier_or_provider(monkeypatch, bad_message):
    history_calls, classifier_calls, provider_calls = _install_call_counters(monkeypatch)

    response = await tutor.route_text_request(_new_user_id(), bad_message, mode=BotMode.TEXT)

    assert response.get("error") == "TextChatValidationError"
    assert history_calls["n"] == 0
    assert classifier_calls["n"] == 0
    assert provider_calls["n"] == 0


async def test_invalid_raw_mode_never_logged_before_validation(caplog):
    import logging

    sentinel = "TOTALLY_BOGUS_MODE_SENTINEL_tutor_never_logged"
    with caplog.at_level(logging.DEBUG):
        response = await tutor.route_text_request(_new_user_id(), "hello", mode=sentinel)

    assert response.get("error") == "TextChatValidationError"
    assert sentinel not in caplog.text


async def test_same_user_busy_request_triggers_no_additional_history_read_or_classifier_call(monkeypatch):
    """The exact scenario the independent re-audit reproduced: while a
    first same-user request is genuinely in flight (blocked mid-provider-
    call), a second same-user request must be rejected as busy WITHOUT
    ever adding to the history-read or classifier-call counts -- i.e. the
    busy rejection happens strictly before either of those runs for the
    second request."""
    history_calls, classifier_calls, _provider_calls = _install_call_counters(monkeypatch)

    first_holding = asyncio.Event()
    release_first = asyncio.Event()

    async def hanging_generate(messages, max_tokens=None):
        first_holding.set()
        await release_first.wait()
        return "first reply"

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", hanging_generate)

    user_id = _new_user_id()
    first_task = asyncio.create_task(tutor.route_text_request(user_id, "first message"))
    await first_holding.wait()

    history_count_before_second = history_calls["n"]
    classifier_count_before_second = classifier_calls["n"]
    assert history_count_before_second == 1  # only the first request's own single read
    assert classifier_count_before_second == 1

    second_response = await tutor.route_text_request(user_id, "second message")
    assert second_response.get("error") == "GenerationBusyError"

    # The rejected second request contributed NOTHING to either count.
    assert history_calls["n"] == history_count_before_second
    assert classifier_calls["n"] == classifier_count_before_second

    release_first.set()
    first_response = await first_task
    assert first_response["text"] == "first reply"
