"""
Stage 2B-B regression tests: VectorIndex against REAL local-persistent
Qdrant (not a mocked vector database) — per Stage 2B-B Section T, core
Qdrant behavior must be proven against the actual qdrant-client local
implementation.

All embeddings are a deterministic local fake (tests/rag_fakes.py) — NO
OpenAI call is ever made from this module. Every VectorIndex here is
constructed with an explicit tmp_path persist_directory and a distinct
collection name, so nothing here ever touches the real (gitignored)
data/qdrant. pytest.ini's global `--disable-socket --allow-hosts=127.0.0.1,::1`
policy means any attempt by qdrant-client's local mode to reach a real
network socket beyond loopback would already fail the whole test session
— local Qdrant genuinely needs none, which these tests passing at all is
itself evidence of.
"""

import threading

import pytest
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, PointIdsList

import config
import rag.index as rag_index_module
from rag.identity import point_id
from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings


@pytest.fixture
def index_factory(tmp_path):
    """Builds real local-persistent VectorIndex instances against tmp_path,
    each closed automatically at teardown (deterministic lock release —
    Stage 2B-B Section F requires this be tested, not left to GC)."""
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


# ---------------------------------------------------------------------------
# 1/2/3: collection initializes with the explicit schema, zero provider calls
# ---------------------------------------------------------------------------

def test_collection_initializes_with_explicit_schema_and_zero_embedding_calls(index_factory):
    fake = DeterministicFakeEmbeddings()
    vi = index_factory(embeddings=fake)

    info = vi.client.get_collection(vi.collection_name)
    assert info.config.params.vectors.size == config.EMBEDDING_DIMENSIONS == 1536
    assert info.config.params.vectors.distance == Distance.COSINE

    # Collection creation must never probe embedding dimensions live.
    assert fake.embed_documents_call_count == 0
    assert fake.embed_query_call_count == 0
    assert vi.get_stats(requesting_user_id=1) == {"total_documents": 0, "status": "ok"}


# ---------------------------------------------------------------------------
# 4: local persistence survives close + reopen
# ---------------------------------------------------------------------------

def test_local_persistence_survives_close_and_reopen(tmp_path):
    qdir = tmp_path / "qdrant"
    vi1 = VectorIndex(persist_directory=qdir, embeddings=DeterministicFakeEmbeddings(), collection_name="persist_test")
    vi1.add_documents([_doc("alpha content", "doc1", 0)])
    assert vi1.get_stats(requesting_user_id=1)["total_documents"] == 1
    vi1.close()

    vi2 = VectorIndex(persist_directory=qdir, embeddings=DeterministicFakeEmbeddings(), collection_name="persist_test")
    try:
        assert vi2.get_stats(requesting_user_id=1)["total_documents"] == 1
    finally:
        vi2.close()


# ---------------------------------------------------------------------------
# 5/15: close() releases the persistent-path lock; a concurrent second
# client against the same still-open path is rejected
# ---------------------------------------------------------------------------

def test_close_releases_persistent_path_lock(tmp_path):
    qdir = tmp_path / "qdrant"
    vi1 = VectorIndex(persist_directory=qdir, embeddings=DeterministicFakeEmbeddings(), collection_name="lock_test")
    vi1.close()

    # Must not raise "already accessed by another instance" now that vi1 closed.
    vi2 = VectorIndex(persist_directory=qdir, embeddings=DeterministicFakeEmbeddings(), collection_name="lock_test")
    vi2.close()


def test_second_simultaneous_client_against_same_path_is_rejected(tmp_path):
    qdir = tmp_path / "qdrant"
    vi1 = VectorIndex(persist_directory=qdir, embeddings=DeterministicFakeEmbeddings(), collection_name="reject_test")
    try:
        with pytest.raises(Exception):
            VectorIndex(persist_directory=qdir, embeddings=DeterministicFakeEmbeddings(), collection_name="reject_test")
    finally:
        vi1.close()


# ---------------------------------------------------------------------------
# 6: upsert / count
# ---------------------------------------------------------------------------

def test_add_documents_upserts_and_count_reflects_chunks(index_factory):
    vi = index_factory()
    chunks = [_doc(f"chunk number {i}", "docA", i) for i in range(5)]
    vi.add_documents(chunks)
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 5


# ---------------------------------------------------------------------------
# 7: deterministic point ids
# ---------------------------------------------------------------------------

def test_point_ids_are_deterministic(index_factory):
    vi = index_factory()
    chunks = [_doc("first chunk text", "docB", 0), _doc("second chunk text", "docB", 1)]
    vi.add_documents(chunks)

    expected_ids = {point_id("docB", 0), point_id("docB", 1)}
    assert vi._existing_point_ids("docB") == expected_ids

    # Re-computing point_id() independently for the same (document_id,
    # chunk_index) always yields the same id — the property replacement
    # (Section M) depends on.
    assert point_id("docB", 0) == point_id("docB", 0)


# ---------------------------------------------------------------------------
# 9: payload round trip — only safe fields, never an absolute path
# ---------------------------------------------------------------------------

def test_payload_round_trip_preserves_safe_metadata_only(index_factory):
    vi = index_factory()
    chunk = _doc(
        "round trip content", "docC", 0, source="notes.txt",
        content_sha256="abc123", stored_name="uuidname.txt", page=2,
        file_path="C:\\Users\\someone\\secret_deploy_user\\notes.txt",  # must NOT survive into payload
    )
    vi.add_documents([chunk])

    results = vi.similarity_search_with_score("round trip content", requesting_user_id=1, k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "round trip content"
    assert doc.metadata["source"] == "notes.txt"
    assert doc.metadata["document_id"] == "docC"
    assert doc.metadata["chunk_index"] == 0
    assert doc.metadata["content_sha256"] == "abc123"
    assert doc.metadata["stored_name"] == "uuidname.txt"
    assert doc.metadata["page"] == 2
    assert "file_path" not in doc.metadata
    assert "secret_deploy_user" not in str(doc.metadata)

    # Also verify directly against the raw Qdrant payload, not just the
    # Document conversion — proves the absolute path was never written to
    # Qdrant at all, not merely filtered back out on the way out.
    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
    for record in records:
        assert "file_path" not in record.payload
        assert "secret_deploy_user" not in str(record.payload)


# ---------------------------------------------------------------------------
# 8: query_points nearest-first top-k ordering
# ---------------------------------------------------------------------------

def test_similarity_search_returns_nearest_first(index_factory):
    vi = index_factory()
    chunks = [
        _doc("apples and oranges", "docD", 0),
        _doc("quantum mechanics textbook", "docD", 1),
        _doc("a completely unrelated sentence about weather", "docD", 2),
    ]
    vi.add_documents(chunks)

    # Querying with the EXACT text of one chunk yields the identical
    # deterministic fake vector that chunk was embedded with (same hash
    # function on both sides), so it must rank first — a fully
    # deterministic nearest-match proof, no manual vector math needed.
    results = vi.similarity_search_with_score("quantum mechanics textbook", requesting_user_id=1, k=3)
    assert len(results) == 3
    assert results[0][0].page_content == "quantum mechanics textbook"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)  # nearest-first


# ---------------------------------------------------------------------------
# 10: duplicate display filenames under separate document_ids coexist
# ---------------------------------------------------------------------------

def test_duplicate_display_filenames_coexist_as_separate_documents(index_factory):
    vi = index_factory()
    vi.add_documents([_doc("first upload content", "upload:aaa", 0, source="notes.txt")])
    vi.add_documents([_doc("second upload content", "upload:bbb", 0, source="notes.txt")])

    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 2
    assert len(vi._existing_point_ids("upload:aaa")) == 1
    assert len(vi._existing_point_ids("upload:bbb")) == 1
    assert vi._existing_point_ids("upload:aaa") != vi._existing_point_ids("upload:bbb")


# ---------------------------------------------------------------------------
# 11: repeated identical indexing does not increase count
# ---------------------------------------------------------------------------

def test_repeated_identical_indexing_is_idempotent(index_factory):
    vi = index_factory()
    vi.add_documents([_doc("stable content", "docE", 0)])
    vi.add_documents([_doc("stable content", "docE", 0)])
    vi.add_documents([_doc("stable content", "docE", 0)])
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 1


# ---------------------------------------------------------------------------
# 12: 10 -> 7 chunk replacement removes exactly the stale 3 (Section M/T.12)
# ---------------------------------------------------------------------------

def test_replacing_document_with_fewer_chunks_removes_stale_points(index_factory):
    vi = index_factory()
    original = [_doc(f"original chunk {i}", "docF", i) for i in range(10)]
    vi.add_documents(original)
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 10

    replacement = [_doc(f"replacement chunk {i}", "docF", i) for i in range(7)]
    vi.add_documents(replacement)

    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 7
    remaining_ids = vi._existing_point_ids("docF")
    assert remaining_ids == {point_id("docF", i) for i in range(7)}
    stale_ids = {point_id("docF", i) for i in range(7, 10)}
    assert remaining_ids.isdisjoint(stale_ids)


def test_embedding_failure_leaves_previous_valid_index_untouched(index_factory, monkeypatch):
    """Section M: an embedding/upsert failure must never erase the
    previously valid index — old points are deleted only AFTER a
    successful upsert of the new set, never before."""
    vi = index_factory()
    vi.add_documents([_doc(f"chunk {i}", "docK", i) for i in range(3)])
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 3

    def failing_embed_documents(texts):
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(vi.embeddings, "embed_documents", failing_embed_documents)

    with pytest.raises(RuntimeError):
        vi.add_documents([_doc("new chunk", "docK", 0)])

    # Old points are still fully intact — nothing was deleted first.
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 3
    assert vi._existing_point_ids("docK") == {point_id("docK", i) for i in range(3)}


# ---------------------------------------------------------------------------
# 13: clear_index recreates the collection safely
# ---------------------------------------------------------------------------

def test_clear_index_recreates_collection(index_factory):
    vi = index_factory()
    vi.add_documents([_doc("some content", "docG", 0)])
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 1

    vi.clear_index()
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 0

    # Collection is genuinely usable again afterward — not left absent.
    vi.add_documents([_doc("fresh content", "docH", 0)])
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 1


# ---------------------------------------------------------------------------
# 16: cross-thread access through VectorIndex succeeds under the RLock
# ---------------------------------------------------------------------------

def test_cross_thread_access_succeeds_under_rlock(index_factory):
    vi = index_factory()
    errors = []

    def worker(i):
        try:
            vi.add_documents([_doc(f"thread chunk {i}", f"docThread{i}", 0)])
        except Exception as e:  # pragma: no cover - failure path only
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert not errors
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 8


# ---------------------------------------------------------------------------
# 17: concurrent operations cannot bypass the RLock
# ---------------------------------------------------------------------------

def test_concurrent_search_cannot_bypass_lock_held_by_add_documents(index_factory, monkeypatch):
    vi = index_factory()
    vi.add_documents([_doc("baseline content", "docI", 0)])

    events = []
    events_lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()

    real_upsert = vi.client.upsert

    def blocking_upsert(*args, **kwargs):
        with events_lock:
            events.append("upsert_entered")
        entered.set()
        assert release.wait(timeout=5), "test setup: release was never set"
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(vi.client, "upsert", blocking_upsert)

    worker = threading.Thread(target=lambda: vi.add_documents([_doc("new content", "docJ", 0)]))
    worker.start()
    assert entered.wait(timeout=5), "worker never reached the underlying Qdrant upsert call"

    search_done = threading.Event()

    def do_search():
        vi.similarity_search("baseline content", requesting_user_id=1, k=1)
        with events_lock:
            events.append("search_completed")
        search_done.set()

    searcher = threading.Thread(target=do_search)
    searcher.start()

    # The searcher cannot complete yet — add_documents still holds the lock.
    assert not search_done.wait(timeout=0.3), "concurrent search bypassed the RLock held by add_documents()"
    with events_lock:
        assert "search_completed" not in events

    release.set()
    worker.join(timeout=5)
    searcher.join(timeout=5)
    assert not worker.is_alive()
    assert not searcher.is_alive()

    assert events[0] == "upsert_entered"
    assert events[-1] == "search_completed"


# ---------------------------------------------------------------------------
# Stage 2B-C Blocker 2: reconcile_document() freshness/convergence
# classification — a matching content_sha256 on SOME points is never
# sufficient on its own; the exact expected point-ID SET must match, and a
# previous stale-delete failure must converge for free (zero re-embedding)
# on the next call rather than being stuck forever.
# ---------------------------------------------------------------------------

def _install_fake_loader(monkeypatch, chunk_count_ref, version_ref):
    """Replaces rag.index.document_loader.load_document() (the SAME shared
    singleton rag/index.py itself calls) with a fake that ignores the
    physical file's real text content and instead returns
    `chunk_count_ref["n"]` chunks whose text embeds `version_ref["v"]` (so
    the deterministic fake embeddings produce a different vector/hash scope
    per version) — this lets a test precisely control "how many chunks
    this document currently has" across successive reconcile_document()
    calls without fighting the real RecursiveCharacterTextSplitter's exact
    chunk-boundary arithmetic."""
    def fake_load_document(path, display_name=None, document_id=None, content_sha256=None, stored_name=None, owner_user_id=None):
        extra = {"content_sha256": content_sha256}
        if owner_user_id is not None:
            extra["owner_user_id"] = owner_user_id
        return [
            _doc(f"{version_ref['v']} chunk {i}", document_id, i, **extra)
            for i in range(chunk_count_ref["n"])
        ]
    monkeypatch.setattr(rag_index_module.document_loader, "load_document", fake_load_document)


def test_reconcile_document_exact_current_makes_zero_calls(index_factory, monkeypatch, tmp_path):
    fake = DeterministicFakeEmbeddings()
    vi = index_factory(embeddings=fake)
    file_path = tmp_path / "doc.txt"
    file_path.write_text("v1", encoding="utf-8")

    chunk_count_ref = {"n": 3}
    version_ref = {"v": "v1"}
    _install_fake_loader(monkeypatch, chunk_count_ref, version_ref)

    status, count = vi.reconcile_document("docExact", file_path)
    assert status == "reindexed"
    assert count == 3
    assert fake.embed_documents_call_count == 1

    upsert_calls = {"n": 0}
    delete_calls = {"n": 0}
    real_upsert, real_delete = vi.client.upsert, vi.client.delete

    def counting_upsert(*args, **kwargs):
        upsert_calls["n"] += 1
        return real_upsert(*args, **kwargs)

    def counting_delete(*args, **kwargs):
        delete_calls["n"] += 1
        return real_delete(*args, **kwargs)

    monkeypatch.setattr(vi.client, "upsert", counting_upsert)
    monkeypatch.setattr(vi.client, "delete", counting_delete)

    # Exact same content/chunk-count again — must be a complete no-op.
    status2, count2 = vi.reconcile_document("docExact", file_path)
    assert status2 == "unchanged"
    assert count2 == 3
    assert fake.embed_documents_call_count == 1  # NOT incremented
    assert fake.embed_query_call_count == 0
    assert upsert_calls["n"] == 0
    assert delete_calls["n"] == 0
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 3


def test_reconcile_document_converges_after_stale_delete_failure(index_factory, monkeypatch, tmp_path):
    """Reproduces Codex's exact scenario: a 10 -> 7 chunk replacement
    upserts the new 7 successfully, but the stale-delete of the old
    trailing 3 fails. The failure must surface (10 points temporarily
    remain — never silently "successful"), but the NEXT reconcile call
    against the SAME (now-current) content must detect the extra stale
    points and prune them with ZERO re-embedding, converging to exactly 7."""
    fake = DeterministicFakeEmbeddings()
    vi = index_factory(embeddings=fake)
    file_path = tmp_path / "doc.txt"
    doc_id = "docStaleRetry"

    chunk_count_ref = {"n": 10}
    version_ref = {"v": "v1"}
    _install_fake_loader(monkeypatch, chunk_count_ref, version_ref)
    file_path.write_text("v1", encoding="utf-8")

    status, count = vi.reconcile_document(doc_id, file_path)
    assert status == "reindexed" and count == 10
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 10
    embed_calls_after_first = fake.embed_documents_call_count

    # "Change" the document to 7 chunks (new content -> new hash), but make
    # the stale-delete of the old trailing 3 points fail.
    chunk_count_ref["n"] = 7
    version_ref["v"] = "v2"
    file_path.write_text("v2", encoding="utf-8")

    real_delete = vi.client.delete

    def failing_delete(*args, **kwargs):
        raise RuntimeError("simulated stale-delete failure")

    monkeypatch.setattr(vi.client, "delete", failing_delete)

    with pytest.raises(RuntimeError):
        vi.reconcile_document(doc_id, file_path)

    # Operation surfaced failure — but the new 7 were already upserted
    # (safe-replacement embeds+upserts BEFORE deleting stale points), so
    # 10 points temporarily remain: the new 7 plus the 3 stale leftovers.
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 10
    embed_calls_after_failed_replace = fake.embed_documents_call_count
    assert embed_calls_after_failed_replace == embed_calls_after_first + 1

    # Restore delete() — simulating the NEXT real application startup.
    monkeypatch.setattr(vi.client, "delete", real_delete)

    status2, count2 = vi.reconcile_document(doc_id, file_path)
    assert status2 == "stale_pruned"
    assert count2 == 7
    # Zero re-embedding: the retry converges purely by deleting extras.
    assert fake.embed_documents_call_count == embed_calls_after_failed_replace
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 7
    assert vi._existing_point_ids(doc_id) == {point_id(doc_id, i) for i in range(7)}

    # Deterministic: a THIRD call is a complete no-op ("unchanged").
    status3, count3 = vi.reconcile_document(doc_id, file_path)
    assert status3 == "unchanged"
    assert count3 == 7
    assert fake.embed_documents_call_count == embed_calls_after_failed_replace
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 7


def test_reconcile_document_reindexes_when_an_expected_point_is_missing(index_factory, monkeypatch, tmp_path):
    """A matching content_sha256 on the points that DO exist is not
    sufficient if the expected point-ID SET doesn't fully match — e.g. one
    expected point was somehow removed (disk corruption, manual
    intervention, a partial external failure). reconcile_document() must
    detect the gap and perform a full safe replacement, never treat the
    document as merely 'has some stale extras'."""
    fake = DeterministicFakeEmbeddings()
    vi = index_factory(embeddings=fake)
    file_path = tmp_path / "doc.txt"
    doc_id = "docMissingPoint"

    chunk_count_ref = {"n": 5}
    version_ref = {"v": "v1"}
    _install_fake_loader(monkeypatch, chunk_count_ref, version_ref)
    file_path.write_text("v1", encoding="utf-8")

    vi.reconcile_document(doc_id, file_path)
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 5
    embed_calls_before = fake.embed_documents_call_count

    # Directly remove one expected point out from under the index (not
    # via reconcile_document — simulating an external partial loss).
    missing_pid = point_id(doc_id, 2)
    vi.client.delete(collection_name=vi.collection_name, points_selector=PointIdsList(points=[missing_pid]))
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 4

    status, count = vi.reconcile_document(doc_id, file_path)
    assert status == "reindexed"
    assert count == 5
    assert fake.embed_documents_call_count == embed_calls_before + 1
    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 5
    assert vi._existing_point_ids(doc_id) == {point_id(doc_id, i) for i in range(5)}


def test_reconcile_document_verifies_hash_before_embedding_and_refuses_to_mutate_on_mismatch(index_factory, tmp_path):
    """Stage 2B-C Section I: if expected_content_sha256 is passed and
    doesn't match the file's actual current content, reconcile_document()
    must raise and perform NO Qdrant mutation whatsoever."""
    from rag.index import SourceMutatedError

    vi = index_factory()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("real content", encoding="utf-8")

    with pytest.raises(SourceMutatedError):
        vi.reconcile_document("docMutated", file_path, expected_content_sha256="0" * 64)

    assert vi.get_stats(requesting_user_id=1)["total_documents"] == 0
