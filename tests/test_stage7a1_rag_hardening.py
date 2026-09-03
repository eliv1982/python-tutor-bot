"""
Stage 7A-1 SECOND corrective-pass regression tests: rag.query's raw
provider-output validation and per-attempt message isolation.

An independent re-audit found that rag/query.py validated only the FINAL,
already-decorated RAG result at app.text_chat's boundary -- never the RAW
provider output immediately after each of its own two possible provider
calls (_generate_rag_response()'s primary answer-generation call,
_fallback_response()'s own separate call). A malformed raw result (None,
a list, a dict, or an empty/whitespace-only string) could therefore:
  - be silently decorated into an apparently-successful response (the
    "no results" fallback branch interpolates it directly into an
    f-string warning message); or
  - raise a raw AttributeError/TypeError from a later string operation
    (.rstrip()) on the PRIMARY branch, which query_knowledge_base()'s own
    broad `except Exception` then caught and silently retried through
    _fallback_response() -- a hidden SECOND provider call turning a
    genuine failure into a late "success".

Also: RAG reused the SAME message dict objects across its own primary and
fallback provider attempts (built once from `conversation_history[-6:]`
and simply `.extend()`-ed into each attempt's own `messages` list) -- a
primary provider fake that mutates the dicts it receives could therefore
poison what the fallback attempt subsequently received, even though
caller-level aliasing (app.text_chat's own _fresh_messages()) was already
fixed.

Scope covered:
- raw provider output validation immediately after each of RAG's two
  possible provider calls, for both branches ("no results" fallback, and
  the primary/has-results branch), for None/list/dict/empty/
  whitespace-only;
- malformed PRIMARY output never triggers a hidden second (fallback)
  provider call -- exactly one attempt;
- malformed output surfaces, end-to-end through app.text_chat, as a fully
  severed TextChatGenerationError;
- each RAG provider attempt gets its own fresh, caller/attempt-
  independent message dictionaries -- a mutating primary fake cannot
  influence what a subsequent fallback attempt receives.
"""

import uuid
from types import SimpleNamespace

import pytest

import app.text_chat as text_chat
import rag.query as rag_query
from app.generation_limits import GenerationAdmissionController
from app.text_chat import TextChatGenerationError, run_text_chat
from config import BotMode
from rag.index import SCOPE_PRIVATE
from rag.query import RagGenerationOutputError


def _fake_doc(tag="a"):
    return SimpleNamespace(
        metadata={
            "source": "notes.txt",
            "document_id": "upload:" + tag * 32,
            "chunk_index": 0,
            "scope": SCOPE_PRIVATE,
            "owner_user_uuid": str(uuid.uuid4()),
        },
        page_content="some retrieved passage",
    )


_MALFORMED_RAW_RESULTS = [None, [], {}, ["a", "b"], {"text": "x"}, 42, "", "   ", "\n\t\n"]


# ============================================================================
# A. "No retrieval results" branch: query_knowledge_base() calls
#    _fallback_response() directly. Malformed raw output there must never
#    be decorated into an apparently-successful response.
# ============================================================================


@pytest.mark.parametrize("bad_result", _MALFORMED_RAW_RESULTS)
async def test_no_results_branch_malformed_fallback_output_is_rejected_not_decorated(monkeypatch, bad_result):
    monkeypatch.setattr(rag_query, "_validated_similarity_search", lambda query, requesting_user_uuid, k: [])

    async def fake_generate(messages, max_tokens=None):
        return bad_result

    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", fake_generate)

    with pytest.raises(RagGenerationOutputError):
        await rag_query.query_knowledge_base("a question", str(uuid.uuid4()))


async def test_no_results_branch_valid_fallback_output_is_still_decorated_normally(monkeypatch):
    """Regression: a genuinely valid fallback result is still decorated
    with the knowledge-base warning exactly as before."""
    monkeypatch.setattr(rag_query, "_validated_similarity_search", lambda query, requesting_user_uuid, k: [])

    async def fake_generate(messages, max_tokens=None):
        return "a real fallback answer"

    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", fake_generate)

    response = await rag_query.query_knowledge_base("a question", str(uuid.uuid4()))
    assert "a real fallback answer" in response
    assert "База знаний не содержит информации" in response


# ============================================================================
# B. "Has results" (primary) branch: malformed raw output on the PRIMARY
#    answer-generation call must be rejected IMMEDIATELY -- before
#    .rstrip()/source-decoration -- and must NEVER trigger a hidden
#    second (fallback) provider call.
# ============================================================================


@pytest.mark.parametrize("bad_result", _MALFORMED_RAW_RESULTS)
async def test_primary_branch_malformed_output_is_rejected_with_exactly_one_provider_call(monkeypatch, bad_result):
    call_count = {"n": 0}

    monkeypatch.setattr(
        rag_query, "_validated_similarity_search",
        lambda query, requesting_user_uuid, k: [(_fake_doc(), 0.1)],
    )

    async def fake_generate(messages, max_tokens=None):
        call_count["n"] += 1
        return bad_result

    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", fake_generate)

    with pytest.raises(RagGenerationOutputError):
        await rag_query.query_knowledge_base("a question", str(uuid.uuid4()))

    assert call_count["n"] == 1, "malformed primary output must never trigger a hidden second (fallback) provider call"


async def test_primary_branch_valid_output_is_still_decorated_with_sources_normally(monkeypatch):
    monkeypatch.setattr(
        rag_query, "_validated_similarity_search",
        lambda query, requesting_user_uuid, k: [(_fake_doc(), 0.1)],
    )

    async def fake_generate(messages, max_tokens=None):
        return "a real grounded answer"

    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", fake_generate)

    response = await rag_query.query_knowledge_base("a question", str(uuid.uuid4()))
    assert response.startswith("a real grounded answer")
    assert "Источник(и):" in response


# ============================================================================
# C. End-to-end through app.text_chat: malformed RAG output becomes a
#    fully-severed TextChatGenerationError, exactly like any other RAG
#    failure.
# ============================================================================


async def test_malformed_rag_output_surfaces_as_text_chat_generation_error_end_to_end(monkeypatch):
    controller = GenerationAdmissionController(max_per_user=1, max_global=4)
    monkeypatch.setattr(text_chat, "generation_admission_controller", controller)

    monkeypatch.setattr(
        rag_query, "_validated_similarity_search",
        lambda query, requesting_user_uuid, k: [(_fake_doc(), 0.1)],
    )

    async def fake_generate(messages, max_tokens=None):
        return None

    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", fake_generate)

    with pytest.raises(TextChatGenerationError) as excinfo:
        await run_text_chat(user_id=uuid.uuid4(), message="hi", history=[], mode=BotMode.RAG)

    assert str(excinfo.value) == "Text generation failed"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert controller.registry_size() == 0
    assert controller.global_active_count() == 0


# ============================================================================
# D. Attempt isolation: a mutating primary provider fake cannot influence
#    what a subsequent, legitimate fallback attempt receives.
# ============================================================================


async def test_primary_mutation_does_not_leak_into_fallback_messages(monkeypatch):
    """The primary attempt mutates the message list/dicts it was actually
    handed (append an injected entry, change an existing entry's role),
    then fails with an ordinary provider exception (a genuine failure,
    NOT malformed output) that legitimately triggers
    query_knowledge_base()'s own broad-except fallback. The fallback
    attempt must receive a PRISTINE reconstruction of the original
    conversation_history -- no injected entry, no mutated role."""
    monkeypatch.setattr(
        rag_query, "_validated_similarity_search",
        lambda query, requesting_user_uuid, k: [(_fake_doc(), 0.1)],
    )

    captured_calls = []

    async def mutating_then_fallback_generate(messages, max_tokens=None):
        captured_calls.append(messages)
        if len(captured_calls) == 1:
            # Mutate the list AND a dict inside it -- this must never
            # reach the second (fallback) attempt's own messages.
            messages.append({"role": "user", "content": "INJECTED"})
            if len(messages) > 1:
                messages[1]["role"] = "hacked"
                messages[1]["content"] = "MUTATED"
            raise RuntimeError("simulated primary provider failure")
        return "fallback reply"

    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", mutating_then_fallback_generate)

    conversation_history = [{"role": "assistant", "content": "original reference answer"}]
    expected_snapshot = [dict(e) for e in conversation_history]

    response = await rag_query.query_knowledge_base("a question", str(uuid.uuid4()), conversation_history)

    assert "fallback reply" in response
    assert len(captured_calls) == 2

    fallback_messages = captured_calls[1]
    assert "INJECTED" not in [m.get("content") for m in fallback_messages]
    assert "hacked" not in [m.get("role") for m in fallback_messages]
    assert "MUTATED" not in [m.get("content") for m in fallback_messages]

    # The fallback attempt's own history-derived entries are a pristine
    # reconstruction of the original conversation_history.
    history_entries = [m for m in fallback_messages if m["content"] == "original reference answer"]
    assert history_entries == [{"role": "assistant", "content": "original reference answer"}]

    # conversation_history itself (the caller's own object) was never
    # mutated either.
    assert conversation_history == expected_snapshot


async def test_each_rag_attempt_gets_distinct_dict_objects_not_shared_across_attempts(monkeypatch):
    """More direct proof than the mutation test above: capture the actual
    dict OBJECT IDENTITIES each attempt receives for the history-derived
    entries, and confirm the primary and fallback attempts never share a
    single dict object."""
    monkeypatch.setattr(
        rag_query, "_validated_similarity_search",
        lambda query, requesting_user_uuid, k: [(_fake_doc(), 0.1)],
    )

    captured_calls = []
    call_count = {"n": 0}

    async def failing_then_ok(messages, max_tokens=None):
        captured_calls.append(messages)
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure")
        return "ok"

    monkeypatch.setattr(rag_query.text_llm, "generate_text_response", failing_then_ok)

    conversation_history = [{"role": "user", "content": "shared history entry"}]
    response = await rag_query.query_knowledge_base("a question", str(uuid.uuid4()), conversation_history)

    assert response == "⚠️ База знаний не содержит информации по этому вопросу.\n\nok"
    assert len(captured_calls) == 2

    primary_history_dicts = [m for m in captured_calls[0] if m["content"] == "shared history entry"]
    fallback_history_dicts = [m for m in captured_calls[1] if m["content"] == "shared history entry"]
    assert len(primary_history_dicts) == 1
    assert len(fallback_history_dicts) == 1
    assert primary_history_dicts[0] is not fallback_history_dicts[0]
    assert primary_history_dicts[0] is not conversation_history[0]
    assert fallback_history_dicts[0] is not conversation_history[0]
