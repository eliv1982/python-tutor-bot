"""
Stage 5B regression tests: tutoring orchestration (app/tutor.py) and
conversation-state ownership (app/session.py) as an explicit,
Telegram-independent application boundary.

Covers:
- app.tutor.route_text_request() is callable with plain int/str values,
  no telebot message/update objects anywhere in the call.
- Stage 5A finding: UserSession.get_history() used to return the LIVE
  list, so a text turn added via add_message() before the provider
  request was built showed up once through that mutation and once more
  via the explicit trailing message — duplicating the current turn on
  every call after the first. Fixed in app/session.py by returning a
  snapshot copy. This file proves the fix directly (unit-level) and
  through the real route_text_request() call path (integration-level),
  for both TEXT and RAG modes.
- Existing Text/RAG routing selection remains correct.
- Provider selection / no-silent-fallback remains intact through the
  moved orchestration.
- Conversation state (history/mode/voice/pending-image) remains
  per-user isolated after moving out of utils/helpers.py.

All Telegram/OpenAI/Anthropic/Qdrant boundaries are mocked or use the
existing deterministic fakes. No network access is performed by this
module.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.tutor as tutor
from app.session import UserSession, user_sessions
from config import BotMode


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Reset the shared in-memory session store around each test (same
    convention as tests/test_stage1c_access_control.py etc.)."""
    user_sessions.sessions.clear()
    yield
    user_sessions.sessions.clear()


def _no_image_intent():
    return AsyncMock(return_value={"needs_generation": False, "confidence": 0.0})


# ---------------------------------------------------------------------------
# A. Callable without Telegram message/update objects
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_route_text_request_callable_with_plain_values_no_telegram_objects(monkeypatch):
    """route_text_request() takes only a plain int user_id and a plain str
    — no telebot.types.Message/CallbackQuery, no bot instance — and
    returns a plain dict, matching the Stage 5B application-boundary
    contract."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())
    monkeypatch.setattr(tutor.text_llm, "generate_text_response", AsyncMock(return_value="Привет! Это ответ тьютора."))

    user_id = 12345  # a bare int — not derived from any Telegram object
    result = await tutor.route_text_request(user_id, "Что такое список в Python?")

    assert isinstance(result, dict)
    assert result["text"] == "Привет! Это ответ тьютора."
    assert result["mode"] == BotMode.TEXT


# ---------------------------------------------------------------------------
# B. Duplicate-current-turn fix (Stage 5A finding)
# ---------------------------------------------------------------------------

def test_get_history_returns_a_snapshot_not_the_live_list():
    """Unit-level proof of the fix in app/session.py: a caller that grabs
    get_history() and then calls add_message() must not see that new
    message silently appear in the snapshot it already holds."""
    session = UserSession()
    user_id = 1
    session.add_message(user_id, "user", "turn one")
    session.add_message(user_id, "assistant", "reply one")

    snapshot = session.get_history(user_id)
    assert len(snapshot) == 2

    session.add_message(user_id, "user", "turn two")

    assert len(snapshot) == 2, "the earlier snapshot must not grow when new messages are added afterward"
    assert snapshot[-1]["content"] == "reply one"


@pytest.mark.asyncio
async def test_current_user_turn_appears_exactly_once_on_subsequent_calls(monkeypatch):
    """Integration-level proof, TEXT mode: on the second (and any later)
    call for the same user, the provider request must contain the current
    turn's text exactly once. Before the fix, UserSession.get_history()
    returned the live list, so add_message(user, text) — called before the
    messages list was built — made the current turn appear once via that
    mutation and once more via the explicit trailing
    {"role": "user", "content": text} append."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())
    captured_messages = []

    async def fake_generate(messages, **kwargs):
        captured_messages.append(messages)
        return f"response {len(captured_messages)}"

    monkeypatch.setattr(tutor.text_llm, "generate_text_response", fake_generate)

    user_id = 555
    await tutor.route_text_request(user_id, "first turn")
    await tutor.route_text_request(user_id, "second turn")

    # The bug only manifests from the SECOND call onward (see app/session.py
    # docstring) — assert on it explicitly rather than only checking counts.
    second_call_messages = captured_messages[1]
    occurrences = sum(
        1 for m in second_call_messages
        if m.get("role") == "user" and m.get("content") == "second turn"
    )
    assert occurrences == 1, f"expected 'second turn' exactly once, found it {occurrences} times: {second_call_messages}"

    # And the immediately preceding turn must still be present exactly
    # once too — the fix must not have dropped legitimate history.
    first_turn_occurrences = sum(
        1 for m in second_call_messages
        if m.get("role") == "user" and m.get("content") == "first turn"
    )
    assert first_turn_occurrences == 1


@pytest.mark.asyncio
async def test_current_user_turn_appears_exactly_once_in_rag_mode(monkeypatch):
    """Same regression, RAG mode: route_text_request() passes `history` into
    query_knowledge_base() before add_message() may have mutated it."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())
    await user_sessions.set_mode(777, BotMode.RAG)

    captured_history = []

    async def fake_query_knowledge_base(query, requesting_user_uuid, conversation_history=None):
        captured_history.append(list(conversation_history or []))
        return f"rag response for {query}"

    import rag.query as rag_query
    monkeypatch.setattr(rag_query, "query_knowledge_base", fake_query_knowledge_base)
    # route_text_request() imports query_knowledge_base lazily from
    # rag.query inside the function body, so patch it at the source module.

    await tutor.route_text_request(777, "first rag turn")
    await tutor.route_text_request(777, "second rag turn")

    second_call_history = captured_history[1]
    occurrences = sum(1 for m in second_call_history if m.get("content") == "second rag turn")
    assert occurrences == 0, (
        "the CURRENT turn must not already be inside the history snapshot "
        f"handed to query_knowledge_base(): {second_call_history}"
    )


# ---------------------------------------------------------------------------
# C. Text/RAG routing selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_mode_calls_text_llm_not_rag(monkeypatch):
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())
    text_llm_mock = AsyncMock(return_value="plain answer")
    monkeypatch.setattr(tutor.text_llm, "generate_text_response", text_llm_mock)

    import rag.query as rag_query
    rag_mock = AsyncMock(return_value="should not be called")
    monkeypatch.setattr(rag_query, "query_knowledge_base", rag_mock)

    result = await tutor.route_text_request(1, "hello", mode=BotMode.TEXT)

    text_llm_mock.assert_awaited_once()
    rag_mock.assert_not_called()
    assert result["text"] == "plain answer"


@pytest.mark.asyncio
async def test_rag_mode_calls_query_knowledge_base_not_text_llm(monkeypatch):
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())
    text_llm_mock = AsyncMock(return_value="should not be called")
    monkeypatch.setattr(tutor.text_llm, "generate_text_response", text_llm_mock)

    import rag.query as rag_query
    rag_mock = AsyncMock(return_value="rag answer")
    monkeypatch.setattr(rag_query, "query_knowledge_base", rag_mock)

    result = await tutor.route_text_request(1, "hello", mode=BotMode.RAG)

    rag_mock.assert_awaited_once()
    text_llm_mock.assert_not_called()
    assert result["text"] == "rag answer"


# ---------------------------------------------------------------------------
# D. Provider selection / no silent fallback, preserved through the move
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_selected_provider_failure_never_silently_falls_back(monkeypatch):
    """The exhaustive provider-dispatch/no-fallback contract itself is
    covered by tests/test_stage2a_text_llm_provider.py; this test only
    proves that guarantee still holds THROUGH the moved orchestration
    (app.tutor.route_text_request -> services.text_llm.generate_text_response)
    rather than re-deriving it from scratch."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())

    from services.anthropic_client import anthropic_client
    from services.openai_client import openai_client
    import config

    monkeypatch.setattr(config, "LLM_PROVIDER", config.LLMProvider.ANTHROPIC)
    anthropic_mock = AsyncMock(side_effect=RuntimeError("anthropic is down"))
    openai_mock = AsyncMock(return_value="should never be reached")
    monkeypatch.setattr(anthropic_client, "generate_text_response", anthropic_mock)
    monkeypatch.setattr(openai_client, "generate_text_response", openai_mock)

    result = await tutor.route_text_request(1, "hello", mode=BotMode.TEXT)

    anthropic_mock.assert_awaited_once()
    openai_mock.assert_not_called()
    assert result.get("error") == "RuntimeError"


# ---------------------------------------------------------------------------
# E. Conversation-state ownership: isolation + equivalence after the move
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_users_remain_isolated_across_history_mode_voice_and_pending_image():
    session = UserSession()
    session.add_message(1, "user", "user one's message")
    await session.set_mode(1, BotMode.RAG)
    await session.set_voice(1, "nova")
    session.set_pending_image(1, "data:image/png;base64,AAA")

    session.add_message(2, "user", "user two's message")
    await session.set_mode(2, BotMode.VOICE)

    assert session.get_history(2) == [{"role": "user", "content": "user two's message"}]
    assert await session.get_mode(2) == BotMode.VOICE
    assert session.get_pending_image(2) is None  # never set for user 2

    # User 1's state is untouched by user 2's activity.
    assert await session.get_mode(1) == BotMode.RAG
    assert await session.get_voice(1) == "nova"
    assert session.get_pending_image(1) == "data:image/png;base64,AAA"
    assert len(session.get_history(1)) == 1


def test_clear_history_and_clear_pending_image_are_per_user():
    session = UserSession()
    session.add_message(1, "user", "hello")
    session.add_message(2, "user", "hello")
    session.set_pending_image(1, "img-1")
    session.set_pending_image(2, "img-2")

    session.clear_history(1)
    session.clear_pending_image(1)

    assert session.get_history(1) == []
    assert session.get_pending_image(1) is None
    assert session.get_history(2) == [{"role": "user", "content": "hello"}]
    assert session.get_pending_image(2) == "img-2"


def test_history_length_cap_still_enforced_after_the_move():
    from config import MAX_HISTORY_LENGTH

    session = UserSession()
    for i in range(MAX_HISTORY_LENGTH * 2 + 5):
        session.add_message(1, "user", f"message {i}")

    assert len(session.get_history(1)) == MAX_HISTORY_LENGTH * 2
