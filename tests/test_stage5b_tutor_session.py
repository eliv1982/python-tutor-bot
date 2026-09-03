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

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.text_chat as text_chat
import app.tutor as tutor
from app.session import UserSession, user_sessions
from config import BotMode

# Stage 7A-1: route_text_request() now delegates plain-chat/RAG generation
# to app.text_chat, which itself calls services.text_llm via app.text_chat's
# OWN `text_llm` reference — tutor.py no longer imports text_llm directly,
# so every test below that used to patch `tutor.text_llm.generate_text_response`
# now patches `text_chat.text_llm.generate_text_response` instead. See
# tests/test_stage7a1_telegram_regression.py for the dedicated delegation
# proof.
#
# Stage 7A-1 corrective pass: app.text_chat now strictly validates that
# route_text_request()'s user_id is a genuine uuid.UUID instance (an int
# is rejected before admission/provider are ever touched — see
# tests/test_stage7a1_text_chat_core.py's input-contract proofs). Every
# test below that goes THROUGH route_text_request() (and therefore through
# that validation) now uses a real uuid.uuid4() rather than a bare int —
# preserving each test's original intent ("callable with a plain scalar
# value, not a Telegram object") while satisfying the stricter contract.
# Tests that exercise app.session.UserSession directly (Section E below)
# are unaffected: UserSession itself is not part of Stage 7A-1's validated
# core boundary and still accepts any hashable key.


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
    """route_text_request() takes only a plain uuid.UUID user_id and a
    plain str — no telebot.types.Message/CallbackQuery, no bot instance —
    and returns a plain dict, matching the Stage 5B application-boundary
    contract."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())
    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", AsyncMock(return_value="Привет! Это ответ тьютора."))

    user_id = uuid.uuid4()  # a plain scalar value — not derived from any Telegram object
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

    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", fake_generate)

    user_id = uuid.uuid4()
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
    user_id = uuid.uuid4()
    await user_sessions.set_mode(user_id, BotMode.RAG)

    captured_history = []

    async def fake_query_knowledge_base(query, requesting_user_uuid, conversation_history=None):
        captured_history.append(list(conversation_history or []))
        return f"rag response for {query}"

    import rag.query as rag_query
    monkeypatch.setattr(rag_query, "query_knowledge_base", fake_query_knowledge_base)
    # app.text_chat.execute_admitted_text_chat() imports query_knowledge_base
    # lazily from rag.query inside its own RAG branch, so patch it at the
    # source module.

    await tutor.route_text_request(user_id, "first rag turn")
    await tutor.route_text_request(user_id, "second rag turn")

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
    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", text_llm_mock)

    import rag.query as rag_query
    rag_mock = AsyncMock(return_value="should not be called")
    monkeypatch.setattr(rag_query, "query_knowledge_base", rag_mock)

    result = await tutor.route_text_request(uuid.uuid4(), "hello", mode=BotMode.TEXT)

    text_llm_mock.assert_awaited_once()
    rag_mock.assert_not_called()
    assert result["text"] == "plain answer"


@pytest.mark.asyncio
async def test_rag_mode_calls_query_knowledge_base_not_text_llm(monkeypatch):
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())
    text_llm_mock = AsyncMock(return_value="should not be called")
    monkeypatch.setattr(text_chat.text_llm, "generate_text_response", text_llm_mock)

    import rag.query as rag_query
    rag_mock = AsyncMock(return_value="rag answer")
    monkeypatch.setattr(rag_query, "query_knowledge_base", rag_mock)

    result = await tutor.route_text_request(uuid.uuid4(), "hello", mode=BotMode.RAG)

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
    (app.tutor.route_text_request -> app.text_chat.execute_admitted_text_chat
    -> services.text_llm.generate_text_response) rather than re-deriving it
    from scratch. Stage 7A-1: app.text_chat.execute_admitted_text_chat()
    now wraps ANY provider failure into its own TextChatGenerationError (a
    stable, fixed-message application-layer exception for future HTTP
    mapping — see app/text_chat.py) before it reaches route_text_request()'s
    own broad except-clause, so the reported error TYPE changed from the raw
    provider exception's own class name to that wrapper's — the no-
    fallback-between-providers guarantee this test is actually about is
    unaffected and still asserted below via openai_mock.assert_not_called()."""
    monkeypatch.setattr(tutor, "detect_image_generation_intent", _no_image_intent())

    from services.anthropic_client import anthropic_client
    from services.openai_client import openai_client
    import config

    monkeypatch.setattr(config, "LLM_PROVIDER", config.LLMProvider.ANTHROPIC)
    anthropic_mock = AsyncMock(side_effect=RuntimeError("anthropic is down"))
    openai_mock = AsyncMock(return_value="should never be reached")
    monkeypatch.setattr(anthropic_client, "generate_text_response", anthropic_mock)
    monkeypatch.setattr(openai_client, "generate_text_response", openai_mock)

    result = await tutor.route_text_request(uuid.uuid4(), "hello", mode=BotMode.TEXT)

    anthropic_mock.assert_awaited_once()
    openai_mock.assert_not_called()
    assert result.get("error") == "TextChatGenerationError"


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


# ---------------------------------------------------------------------------
# F. Atomic history exchange to concurrent OS-thread observers (Stage 7A-1
#    second corrective pass): add_exchange() must publish both halves of a
#    user/assistant pair under one synchronization boundary, so a
#    concurrent get_history() call from a genuinely different OS thread
#    can never observe exactly one half of an in-progress exchange.
# ---------------------------------------------------------------------------

import threading


def test_add_exchange_is_atomic_to_a_concurrent_real_thread_observer():
    """Deterministic (no sleep, no timing guess) two-real-thread proof: a
    WRITER thread calls add_exchange() many times in a tight loop; a
    READER thread concurrently calls get_history() in a tight loop and
    records every observed history length. Since add_exchange() only ever
    appends message pairs (never a lone message) starting from an empty
    history, a genuinely atomic exchange means the observed length must
    ALWAYS be even -- an odd length would mean the reader observed exactly
    one half of an in-progress add_exchange(), which is precisely the bug
    this correction closes. Runs enough iterations across real OS threads
    (not asyncio tasks) that the GIL will readily interleave them if the
    two appends are not genuinely synchronized."""
    session = UserSession()
    user_id = uuid.uuid4()
    iterations = 20000

    stop = threading.Event()
    odd_lengths_observed = []

    def writer():
        for i in range(iterations):
            session.add_exchange(user_id, f"user {i}", f"assistant {i}")
        stop.set()

    def reader():
        while not stop.is_set():
            length = len(session.get_history(user_id))
            if length % 2 != 0:
                odd_lengths_observed.append(length)

    t_writer = threading.Thread(target=writer)
    t_reader = threading.Thread(target=reader)
    t_writer.start()
    t_reader.start()
    t_writer.join(timeout=30)
    t_reader.join(timeout=30)

    assert not t_writer.is_alive(), "writer thread did not finish in time"
    assert not t_reader.is_alive(), "reader thread did not finish in time"
    assert odd_lengths_observed == [], (
        f"observed {len(odd_lengths_observed)} odd-length history snapshot(s) -- "
        "a concurrent reader saw exactly one half of an in-progress add_exchange()"
    )


def test_add_exchange_forces_a_stall_between_the_two_halves_and_reader_still_blocks():
    """Fully deterministic single-shot proof (no reliance on GIL scheduling
    luck): a custom list subclass is pre-seeded as the user's history so
    that add_exchange()'s FIRST internal append can be made to stall
    (block on a threading.Event) WHILE add_exchange() still holds
    UserSession._lock. A concurrent get_history() call, from a genuinely
    different real OS thread, must then itself block on that same lock --
    proven by showing it has NOT completed after the stall has been held
    open for a controlled window -- and, once the stall is released, must
    return the COMPLETE pair, never just the first half."""

    class _StallingList(list):
        def __init__(self):
            super().__init__()
            self.first_append_reached = threading.Event()
            self.allow_continue = threading.Event()

        def append(self, item):
            super().append(item)
            if len(self) == 1:
                self.first_append_reached.set()
                self.allow_continue.wait(timeout=10)

    session = UserSession()
    user_id = uuid.uuid4()
    stalling_list = _StallingList()
    session.sessions[user_id] = stalling_list

    writer_done = threading.Event()

    def writer():
        session.add_exchange(user_id, "user turn", "assistant turn")
        writer_done.set()

    t_writer = threading.Thread(target=writer)
    t_writer.start()

    assert stalling_list.first_append_reached.wait(timeout=10), "writer never reached the stall point"

    # The writer is now stalled INSIDE add_exchange(), between the two
    # appends, while still holding session._lock (append() runs under the
    # lock -- see app/session.py). A concurrent get_history() call must
    # therefore itself block on that lock and NOT complete while the stall
    # is held open.
    reader_done = threading.Event()
    reader_result = {}

    def reader():
        reader_result["history"] = session.get_history(user_id)
        reader_done.set()

    t_reader = threading.Thread(target=reader)
    t_reader.start()

    # The reader must NOT complete while the writer is deliberately
    # stalled mid-exchange -- this is the actual proof of exclusion, not a
    # timing guess: we control exactly how long the stall lasts.
    assert not reader_done.wait(timeout=1), (
        "get_history() completed while add_exchange() was stalled between its two "
        "appends -- history mutation is not genuinely exclusive"
    )

    # Release the stall: the writer's second append (and the rest of
    # add_exchange()) now proceeds, releasing the lock.
    stalling_list.allow_continue.set()
    assert writer_done.wait(timeout=10)
    assert reader_done.wait(timeout=10)
    t_writer.join(timeout=10)
    t_reader.join(timeout=10)

    # The reader, once unblocked, observed the COMPLETE pair -- never just
    # the first half.
    assert reader_result["history"] == [
        {"role": "user", "content": "user turn"},
        {"role": "assistant", "content": "assistant turn"},
    ]


def test_concurrent_add_exchange_calls_for_the_same_user_never_interleave():
    """Two writer threads concurrently call add_exchange() for the SAME
    user many times each. If the two appends of one call could interleave
    with another thread's own add_exchange() calls, the resulting history
    could contain a user/assistant pair with mismatched indices (e.g. two
    consecutive "user" entries). Proven by checking that every consecutive
    pair in the final (necessarily length-capped, since both threads
    together vastly exceed MAX_HISTORY_LENGTH) history is a genuine
    (user, assistant) pair with the SAME tag+numeric suffix -- i.e. every
    exchange landed as a whole, never split and reassembled with another
    thread's halves."""
    from config import MAX_HISTORY_LENGTH

    session = UserSession()
    user_id = uuid.uuid4()
    iterations = 5000

    def writer(tag):
        for i in range(iterations):
            session.add_exchange(user_id, f"{tag}-user-{i}", f"{tag}-assistant-{i}")

    t1 = threading.Thread(target=writer, args=("a",))
    t2 = threading.Thread(target=writer, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=60)
    t2.join(timeout=60)
    assert not t1.is_alive() and not t2.is_alive()

    history = session.get_history(user_id)
    # Both threads together appended far more than the cap allows -- the
    # final history must be exactly the (even) capped length.
    assert len(history) == MAX_HISTORY_LENGTH * 2
    for j in range(0, len(history), 2):
        user_entry = history[j]
        assistant_entry = history[j + 1]
        assert user_entry["role"] == "user"
        assert assistant_entry["role"] == "assistant"
        user_tag, _, user_i = user_entry["content"].rpartition("-user-")
        assistant_tag, _, assistant_i = assistant_entry["content"].rpartition("-assistant-")
        assert user_tag == assistant_tag
        assert user_i == assistant_i
