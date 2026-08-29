"""
Stage 3A regression tests: multi-user Qdrant isolation + sidecar ownership
persistence.

Security invariant under test: for every RAG request from Telegram user U,
the retrievable corpus is exactly (1) shared reference documents and (2)
private managed documents owned by U — never any other user's private
documents, and never via a silent/default "search everything" path.

All VectorIndex instances here are real local-persistent Qdrant (not
mocked) against tmp_path, with a deterministic local fake embeddings double
(tests/rag_fakes.py) — no real OpenAI/Qdrant network calls anywhere in this
module, matching the established Stage 2B test convention.
"""

import json

import pytest
from langchain_core.documents import Document
from qdrant_client.http.models import PointStruct

from rag.identity import point_id, sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.sidecar import parse_sidecar_bytes, sidecar_path_for
from rag_fakes import DeterministicFakeEmbeddings


@pytest.fixture
def index_factory(tmp_path):
    """Same pattern as tests/test_stage2b_qdrant_vector_index.py: real
    local-persistent VectorIndex instances against tmp_path, each closed
    automatically at teardown."""
    created = []

    def _make(collection_name="test_collection", embeddings=None, path=None):
        vi = VectorIndex(
            persist_directory=path or (tmp_path / "qdrant"),
            embeddings=embeddings or DeterministicFakeEmbeddings(),
            collection_name=collection_name,
        )
        created.append(vi)
        return vi

    yield _make
    for vi in created:
        try:
            vi.close()
        except Exception:
            pass


def _doc(text, document_id, chunk_index, source="test.md", **extra):
    meta = {"document_id": document_id, "chunk_index": chunk_index, "source": source}
    meta.update(extra)
    return Document(page_content=text, metadata=meta)


# ===========================================================================
# A. Qdrant payload: scope + owner_user_id
# ===========================================================================

def test_reference_chunk_has_scope_reference_and_no_owner(index_factory):
    vi = index_factory()
    vi.add_documents([_doc("shared reference content", "ref:guide", 0)])

    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert len(records) == 1
    assert records[0].payload["scope"] == "reference"
    assert "owner_user_id" not in records[0].payload


def test_private_chunk_has_scope_private_and_correct_owner(index_factory):
    vi = index_factory()
    vi.add_documents([_doc("my private notes", "upload:aaa", 0, owner_user_id=555)])

    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert len(records) == 1
    assert records[0].payload["scope"] == "private"
    assert records[0].payload["owner_user_id"] == 555


def test_owner_and_scope_survive_reconciliation_reindex(index_factory, tmp_path):
    vi = index_factory()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("version one content", encoding="utf-8")

    status, count = vi.reconcile_document("upload:reconcile_owner", file_path, owner_user_id=777)
    assert status == "reindexed"
    assert count == 1
    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert records[0].payload["scope"] == "private"
    assert records[0].payload["owner_user_id"] == 777

    # Reconciling the SAME (unchanged) content must not disturb ownership.
    status2, _ = vi.reconcile_document("upload:reconcile_owner", file_path, owner_user_id=777)
    assert status2 == "unchanged"
    records2, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert records2[0].payload["owner_user_id"] == 777

    # A genuine content change (full re-embed) must also preserve ownership.
    file_path.write_text("version two, completely different content", encoding="utf-8")
    status3, _ = vi.reconcile_document("upload:reconcile_owner", file_path, owner_user_id=777)
    assert status3 == "reindexed"
    records3, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert records3[0].payload["scope"] == "private"
    assert records3[0].payload["owner_user_id"] == 777


def test_two_users_uploading_identical_content_remain_independently_attributed(index_factory):
    vi = index_factory()
    shared_text = "identical shared wording across two independent uploads"
    vi.add_documents([_doc(shared_text, "upload:userA_doc", 0, owner_user_id=1001)])
    vi.add_documents([_doc(shared_text, "upload:userB_doc", 0, owner_user_id=2002)])

    results_a = vi.similarity_search_with_score(shared_text, requesting_user_id=1001, k=5)
    results_b = vi.similarity_search_with_score(shared_text, requesting_user_id=2002, k=5)

    assert len(results_a) == 1
    assert results_a[0][0].metadata["owner_user_id"] == 1001
    assert len(results_b) == 1
    assert results_b[0][0].metadata["owner_user_id"] == 2002


# ===========================================================================
# B. End-to-end retrieval isolation — the core security invariant.
# Deliberately uses a query that is VERBATIM IDENTICAL to the other user's
# private content, so an unfiltered search would rank it a perfect
# (score≈1.0) match — proving a leak here cannot be explained away as
# merely "wasn't ranked highly enough".
# ===========================================================================

def test_no_cross_user_retrieval_even_at_perfect_similarity(index_factory):
    vi = index_factory()
    secret_content = "Confidential quarterly revenue figures for Project Falcon are stored here."
    vi.add_documents([_doc(secret_content, "upload:userA_secret", 0, owner_user_id=1001, source="A_private.txt")])
    vi.add_documents([_doc("Public onboarding guide for new hires.", "ref:onboarding", 0, source="onboarding.md")])

    # User B queries with the EXACT text of user A's private document —
    # the hardest possible case to leak.
    results_b = vi.similarity_search_with_score(secret_content, requesting_user_id=2002, k=5)
    assert all(doc.metadata.get("owner_user_id") != 1001 for doc, _ in results_b)
    assert all(doc.metadata.get("source") != "A_private.txt" for doc, _ in results_b)

    # User A can retrieve their own document with the same query.
    results_a = vi.similarity_search_with_score(secret_content, requesting_user_id=1001, k=5)
    assert any(doc.metadata.get("source") == "A_private.txt" for doc, _ in results_a)

    # Both users can retrieve the shared reference content.
    results_a_ref = vi.similarity_search_with_score("Public onboarding guide for new hires.", requesting_user_id=1001, k=5)
    results_b_ref = vi.similarity_search_with_score("Public onboarding guide for new hires.", requesting_user_id=2002, k=5)
    assert any(doc.metadata.get("source") == "onboarding.md" for doc, _ in results_a_ref)
    assert any(doc.metadata.get("source") == "onboarding.md" for doc, _ in results_b_ref)


@pytest.mark.asyncio
async def test_end_to_end_telegram_upload_and_rag_query_isolates_between_users(monkeypatch, tmp_path):
    """Full-stack proof: Telegram document upload (handlers/document_upload.py)
    -> router (services/router.py) -> rag.query.query_knowledge_base() ->
    VectorIndex, for two DIFFERENT Telegram users, never leaks one user's
    private upload into the other's RAG answer — proven through the real
    production call chain, not just VectorIndex directly."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import handlers.document_upload as document_upload
    import rag.query as rag_query
    from services.openai_client import openai_client

    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="e2e_isolation_test",
    )
    monkeypatch.setattr(document_upload, "get_vector_index", lambda: vi)
    monkeypatch.setattr(rag_query, "get_vector_index", lambda: vi)
    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())

    secret_text = "User A's confidential exam answers: the capital of France quiz key is Paris."

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/secret.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=secret_text.encode("utf-8")))
    message_a = SimpleNamespace(
        from_user=SimpleNamespace(id=11111), chat=SimpleNamespace(id=11111),
        document=SimpleNamespace(file_name="secret.txt", mime_type="text/plain", file_id="fidA", file_size=len(secret_text)),
    )

    try:
        await document_upload.process_document_upload(message_a, message_a.document)
        assert vi.get_stats(requesting_user_id=11111)["total_documents"] == 1

        monkeypatch.setattr(
            openai_client.client.chat.completions, "create",
            AsyncMock(return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="No matching info found."))],
                usage=None,
            )),
        )

        # A different Telegram user queries with the exact text of A's
        # private document — must never see it (and, since the collection
        # holds nothing else, must fall back to the no-results path).
        answer_for_b = await rag_query.query_knowledge_base(secret_text, 22222)
        assert "secret.txt" not in answer_for_b
        assert vi.get_stats(requesting_user_id=22222)["total_documents"] == 0

        # The owner retrieves their own upload successfully.
        answer_for_a = await rag_query.query_knowledge_base(secret_text, 11111)
        assert "secret.txt" in answer_for_a
    finally:
        vi.close()


# ===========================================================================
# C. API enforcement — no accidental unfiltered path through the
# VectorIndex/Qdrant access layer.
# ===========================================================================

def test_similarity_search_omitted_requesting_user_id_raises(index_factory):
    vi = index_factory()
    with pytest.raises(TypeError):
        vi.similarity_search_with_score("query")  # missing required keyword-only arg


@pytest.mark.parametrize("bad_value", [None, True, False, "1001", 0, -5, 3.5])
def test_similarity_search_rejects_invalid_requesting_user_id(index_factory, bad_value):
    vi = index_factory()
    with pytest.raises(ValueError):
        vi.similarity_search_with_score("query", requesting_user_id=bad_value, k=1)


@pytest.mark.parametrize("bad_value", [None, True, False, "1001", 0, -5, 3.5])
def test_get_stats_rejects_invalid_requesting_user_id(index_factory, bad_value):
    vi = index_factory()
    with pytest.raises(ValueError):
        vi.get_stats(requesting_user_id=bad_value)


@pytest.mark.asyncio
async def test_query_knowledge_base_requires_requesting_user_id():
    import inspect

    import rag.query as rag_query

    sig = inspect.signature(rag_query.query_knowledge_base)
    assert "requesting_user_id" in sig.parameters
    assert sig.parameters["requesting_user_id"].default is inspect.Parameter.empty


# ===========================================================================
# D. Stats privacy — a user's stats never reveal another user's private
# document count.
# ===========================================================================

def test_stats_excludes_other_users_private_documents(index_factory):
    vi = index_factory()
    vi.add_documents([_doc("shared ref content", "ref:doc1", 0)])
    vi.add_documents([_doc("user A private content", "upload:a1", 0, owner_user_id=1001)])
    vi.add_documents([_doc("user B private content one", "upload:b1", 0, owner_user_id=2002)])
    vi.add_documents([_doc("user B private content two", "upload:b2", 0, owner_user_id=2002)])

    stats_a = vi.get_stats(requesting_user_id=1001)
    stats_b = vi.get_stats(requesting_user_id=2002)

    # User A sees: 1 reference + 1 own private = 2 — never B's 2 private docs.
    assert stats_a["total_documents"] == 2
    # User B sees: 1 reference + 2 own private = 3.
    assert stats_b["total_documents"] == 3


# ===========================================================================
# E. Legacy sidecar (no recorded owner) is never silently promoted to
# shared/reference, and never silently exposed — rebuild fails closed.
# ===========================================================================

def test_parse_sidecar_bytes_represents_legacy_v1_sidecar_as_unowned():
    legacy = {
        "schema_version": 1,
        "document_id": "upload:" + "b" * 32,
        "display_name": "legacy.txt",
        "stored_name": "b" * 32 + ".txt",
        "content_sha256": "c" * 64,
    }
    parsed = parse_sidecar_bytes(json.dumps(legacy).encode("utf-8"))
    assert parsed["owner_user_id"] is None


def test_rebuild_plan_skips_legacy_sidecar_with_no_owner(tmp_path, monkeypatch):
    import scripts.rebuild_qdrant as rebuild

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uploads_dir.mkdir()

    import rag.loader as rag_loader
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", uploads_dir)

    stem = "a" * 32
    physical = uploads_dir / f"{stem}.txt"
    content = b"legacy upload content with no recorded owner"
    physical.write_bytes(content)
    # Hand-crafted v1 (pre-Stage-3A) sidecar: structurally valid, but no
    # owner_user_id field at all — exactly what a real pre-Stage-3A upload
    # left on disk.
    legacy_sidecar = {
        "schema_version": 1,
        "document_id": upload_document_id(stem),
        "display_name": "legacy.txt",
        "stored_name": physical.name,
        "content_sha256": sha256_hex(content),
    }
    sidecar_path_for(physical).write_text(json.dumps(legacy_sidecar), encoding="utf-8")

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(plan.upload_documents) == 0
    assert "missing_owner" in plan.skipped_upload_reasons
    # Never silently promoted into the plan under any guise.
    assert upload_document_id(stem) not in {d.document_id for d in plan.all_documents}


# ===========================================================================
# F. Pre-Stage-3A Qdrant payload compatibility — acceptance-blocker
# regression tests.
#
# Existing Stage 2 Qdrant points were created before `scope`/`owner_user_id`
# existed at all. reconcile_document()'s "unchanged" classification used to
# key off content_sha256 identity alone, so a pre-Stage-3A reference point
# with matching id/hash but no `scope` field would be accepted as current
# forever and stay permanently excluded from the Stage 3A visibility filter
# (_visibility_filter() requires `scope` to match). This section proves the
# fix: currency now also requires matching Stage 3A visibility metadata.
# ===========================================================================

def _seed_legacy_point(vi, document_id, content, extra_payload):
    """Directly upserts one hand-built Qdrant point bypassing
    VectorIndex._safe_payload() entirely — simulates a point exactly as a
    pre-Stage-3A (or otherwise stale/malformed) reconciliation left it,
    with whatever `extra_payload` visibility fields (or lack thereof) a
    real legacy point would have."""
    pid = point_id(document_id, 0)
    payload = {
        "text": content,
        "document_id": document_id,
        "chunk_index": 0,
        "content_sha256": sha256_hex(content.encode("utf-8")),
        "source": "legacy.txt",
    }
    payload.update(extra_payload)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(id=pid, vector=DeterministicFakeEmbeddings()._vector_for(content), payload=payload)],
    )


def test_scopeless_reference_point_not_accepted_as_unchanged_and_converges(index_factory, tmp_path):
    """Property 1: a pre-Stage-3A reference point with matching id/hash but
    NO scope field at all must not be classified 'unchanged' — it must
    converge (full reindex) to scope='reference', no private owner."""
    vi = index_factory()
    file_path = tmp_path / "legacy_guide.md"
    content = "Legacy pre-Stage-3A reference guide content."
    file_path.write_text(content, encoding="utf-8")
    doc_id = "ref:legacy_guide"

    _seed_legacy_point(vi, doc_id, content, extra_payload={})  # no scope, no owner_user_id
    before, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert "scope" not in before[0].payload

    status, count = vi.reconcile_document(doc_id, file_path)
    assert status != "unchanged"
    assert status == "reindexed"
    assert count == 1

    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert len(records) == 1
    assert records[0].payload["scope"] == "reference"
    assert "owner_user_id" not in records[0].payload

    # A subsequent call against the now-current, correctly-scoped point IS
    # a genuine no-op — this is not a permanent forced-reindex loop.
    status2, _ = vi.reconcile_document(doc_id, file_path)
    assert status2 == "unchanged"


def test_upgraded_reference_document_visible_after_convergence(index_factory, tmp_path):
    """Property 2: once a scope-less legacy reference point converges, it
    is visible through normal user-filtered retrieval and counted in
    visible stats for an arbitrary requesting user — it must never remain
    hidden merely because content/hash never changed."""
    vi = index_factory()
    file_path = tmp_path / "legacy_guide2.md"
    content = "Onboarding steps for the legacy pre-Stage-3A knowledge base."
    file_path.write_text(content, encoding="utf-8")
    doc_id = "ref:legacy_guide2"

    _seed_legacy_point(vi, doc_id, content, extra_payload={})
    status, _ = vi.reconcile_document(doc_id, file_path)
    assert status == "reindexed"

    some_user = 424242
    results = vi.similarity_search_with_score(content, requesting_user_id=some_user, k=5)
    assert any(doc.metadata.get("document_id") == doc_id for doc, _ in results)
    assert vi.get_stats(requesting_user_id=some_user)["total_documents"] == 1


def test_already_current_reference_document_keeps_unchanged_fast_path(index_factory, tmp_path):
    """Property 3: a reference document already correctly Stage-3A-scoped
    (matching id/hash/scope) keeps the existing zero-unnecessary-work
    contract — reconciling it again performs zero re-embedding/replacement."""
    fake = DeterministicFakeEmbeddings()
    vi = index_factory(embeddings=fake)
    file_path = tmp_path / "current_guide.md"
    file_path.write_text("Already-current Stage 3A reference content.", encoding="utf-8")
    doc_id = "ref:current_guide"

    status, count = vi.reconcile_document(doc_id, file_path)
    assert status == "reindexed"
    assert count == 1
    embed_calls_after_first = fake.embed_documents_call_count

    status2, count2 = vi.reconcile_document(doc_id, file_path)
    assert status2 == "unchanged"
    assert count2 == 1
    assert fake.embed_documents_call_count == embed_calls_after_first

    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert records[0].payload["scope"] == "reference"


@pytest.mark.parametrize(
    "stale_extra_payload",
    [
        {},
        {"scope": "private", "owner_user_id": 999},
        {"scope": "reference"},
    ],
    ids=["missing_metadata", "wrong_owner", "wrong_scope"],
)
def test_private_document_with_stale_visibility_metadata_converges_to_trusted_owner(
    index_factory, tmp_path, stale_extra_payload
):
    """Property 4: a private document whose existing points carry
    missing/wrong/inconsistent scope or owner_user_id must not be accepted
    as current, and must converge to the TRUSTED expected owner supplied by
    the caller — never inferred/guessed from the stale payload itself (the
    'wrong_owner' case seeds owner_user_id=999 but must converge to 555)."""
    vi = index_factory()
    file_path = tmp_path / "private_doc.txt"
    content = "Private managed upload content pending Stage 3A convergence."
    file_path.write_text(content, encoding="utf-8")
    doc_id = "upload:legacy_private"
    trusted_owner = 555

    _seed_legacy_point(vi, doc_id, content, extra_payload=stale_extra_payload)

    status, count = vi.reconcile_document(doc_id, file_path, owner_user_id=trusted_owner)
    assert status == "reindexed"
    assert count == 1

    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    assert len(records) == 1
    assert records[0].payload["scope"] == "private"
    assert records[0].payload["owner_user_id"] == trusted_owner

    # A subsequent call with the SAME trusted owner is now a genuine no-op.
    status2, _ = vi.reconcile_document(doc_id, file_path, owner_user_id=trusted_owner)
    assert status2 == "unchanged"
