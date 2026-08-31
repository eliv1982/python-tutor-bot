"""
Stage 5C regression tests: Qdrant private ownership payload/filter using
the canonical internal UUID (`owner_user_uuid`) — most of this boundary is
already exhaustively covered by the retargeted tests/test_stage3a_*.py and
tests/test_stage3b_*.py suites (now UUID-based). This module adds the
specific NEW Stage 5C proofs those don't already cover: the fresh
versioned collection itself, and an explicit "no mixed integer/string
owner contract" regression (a legacy int-owner payload is never accepted
as a valid filter match for a UUID request, and vice versa).

Real local-persistent Qdrant (not mocked), deterministic local fake
embeddings (tests/rag_fakes.py) — no real OpenAI/Qdrant network calls.
"""

import uuid

import pytest
from langchain_core.documents import Document
from qdrant_client.http.models import PointStruct

import rag.constants as rag_constants
from rag.identity import point_id, sha256_hex
from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings


def _doc(text, document_id, chunk_index, source="test.md", **extra):
    meta = {"document_id": document_id, "chunk_index": chunk_index, "source": source}
    meta.update(extra)
    return Document(page_content=text, metadata=meta)


@pytest.fixture
def vi(tmp_path):
    index = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage5c_ownership_test",
    )
    yield index
    index.close()


# ---------------------------------------------------------------------------
# A. Fresh versioned collection name
# ---------------------------------------------------------------------------

def test_production_collection_name_is_the_bumped_uuid_versioned_name():
    """rag_constants.QDRANT_COLLECTION_NAME (the production default every
    real caller uses when it doesn't pass an explicit collection_name=)
    must be the NEW, bumped name — never the old int-owner collection's
    name — so production never mixes UUID-owned points into the old
    collection that may still carry legacy integer-owned points."""
    assert rag_constants.QDRANT_COLLECTION_NAME != "python_tutor_knowledge_base"
    assert "uuid" in rag_constants.QDRANT_COLLECTION_NAME.lower()


# ---------------------------------------------------------------------------
# B. No mixed integer/string owner contract
# ---------------------------------------------------------------------------

def test_legacy_integer_owner_payload_never_matches_a_uuid_requester(vi):
    """A point carrying the OLD int-typed `owner_user_id` field (never
    written by any current code path, but could exist in a not-yet-
    migrated/legacy collection) must never be returned to ANY UUID-based
    `requesting_user_uuid` — there is no code path that maps an int owner
    to a UUID requester, by construction (_SAFE_PAYLOAD_FIELDS only
    recognizes `owner_user_uuid`)."""
    doc_id = "upload:legacy_int_owner"
    pid = point_id(doc_id, 0)
    content = "Legacy point still carrying the old integer owner field."
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(content),
            payload={
                "text": content,
                "document_id": doc_id,
                "chunk_index": 0,
                "content_sha256": sha256_hex(content.encode("utf-8")),
                "source": "legacy.txt",
                "scope": "private",
                "owner_user_id": 123456789,  # the OLD field — never owner_user_uuid
            },
        )],
    )

    some_requester = str(uuid.uuid4())
    results = vi.similarity_search_with_score(content, requesting_user_uuid=some_requester, k=5)
    assert all(doc.metadata.get("document_id") != doc_id for doc, _ in results)


def test_safe_payload_never_writes_the_legacy_integer_owner_field(vi):
    """add_documents()/_safe_payload() must never write `owner_user_id`
    into a Qdrant payload under any circumstances — only
    `owner_user_uuid` is a recognized ownership field going forward."""
    owner = str(uuid.uuid4())
    vi.add_documents([_doc("private content", "upload:new_owner", 0, owner_user_uuid=owner)])

    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert len(records) == 1
    assert "owner_user_id" not in records[0].payload
    assert records[0].payload["owner_user_uuid"] == owner


def test_visibility_filter_rejects_a_raw_telegram_style_integer():
    """A caller that accidentally still passes a raw Telegram integer
    (the pre-Stage-5C identity) instead of the resolved canonical UUID
    string must be rejected outright, never silently coerced or treated
    as a valid (if unusual) owner value."""
    with pytest.raises(ValueError):
        VectorIndex._visibility_filter(123456789)
    with pytest.raises(ValueError):
        VectorIndex._visibility_filter(True)


# ---------------------------------------------------------------------------
# C. UUID private payload + filter, references shared, A/B isolation —
# focused smoke coverage (exhaustive isolation proofs already live in
# tests/test_stage3a_multiuser_isolation.py and
# tests/test_stage3b_rebuild_ownership.py, both retargeted to UUID
# ownership for Stage 5C).
# ---------------------------------------------------------------------------

def test_uuid_private_payload_and_filter_isolate_two_users_and_share_reference(vi):
    owner_a, owner_b = str(uuid.uuid4()), str(uuid.uuid4())
    vi.add_documents([_doc("A's private notes on decorators.", "upload:a", 0, owner_user_uuid=owner_a, source="a.txt")])
    vi.add_documents([_doc("B's private notes on generators.", "upload:b", 0, owner_user_uuid=owner_b, source="b.txt")])
    vi.add_documents([_doc("Shared reference: list comprehensions explained.", "ref:shared", 0, source="ref.md")])

    results_a = vi.similarity_search_with_score("A's private notes on decorators.", requesting_user_uuid=owner_a, k=5)
    assert any(d.metadata.get("source") == "a.txt" for d, _ in results_a)
    assert all(d.metadata.get("source") != "b.txt" for d, _ in results_a)

    results_b = vi.similarity_search_with_score("A's private notes on decorators.", requesting_user_uuid=owner_b, k=5)
    assert all(d.metadata.get("source") != "a.txt" for d, _ in results_b)

    for requester in (owner_a, owner_b):
        ref_results = vi.similarity_search_with_score("Shared reference: list comprehensions explained.", requesting_user_uuid=requester, k=5)
        assert any(d.metadata.get("source") == "ref.md" for d, _ in ref_results)
