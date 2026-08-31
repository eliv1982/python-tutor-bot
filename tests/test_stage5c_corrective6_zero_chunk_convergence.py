"""
Stage 5C corrective pass #6 regression tests (Blocker 2):

VectorIndex.reconcile_document() must never take an early return for a
document whose CURRENT authoritative source parses to zero chunks WITHOUT
first inspecting/removing any points already indexed for it. Qdrant is
derived state (see rag/index.py's own module docstring) — an expected
point set of size zero must still converge Qdrant to zero points for that
document, exactly like any other "expected set differs from what's
currently stored" case.

An independent acceptance review reproduced: a document initially active
with valid Qdrant points, whose durable file/sidecar/catalog later
consistently describe content that now parses to zero chunks. Rebuild ran,
detected the zero-chunk condition, but the OLD `if not chunks: return
("unchanged", 0)` branch returned before ever looking at existing points —
the catalog stayed active AND the previous (now stale) Qdrant points
remained fully retrievable forever.

Lifecycle choice made here (see reconcile_document()'s new "emptied"
outcome and this module's tests): an already-'active' catalog row whose
current source reconciles to zero chunks stays 'active' — no new catalog
lifecycle state is introduced — but Qdrant converges to genuinely zero
points for it, so retrieval returns no stale content. A still-'pending'
document (nothing indexed yet) with zero current chunks remains exactly as
Stage 5C corrective pass #5 left it: never newly activated, "unchanged".

Section A exercises VectorIndex.reconcile_document() directly (no
PostgreSQL needed — pure Qdrant + deterministic fake embeddings). Section B
exercises the full scripts.rebuild_qdrant.apply_plan() lifecycle against a
REAL disposable PostgreSQL container (tests/conftest.py's
postgres_container()/postgres_db()).
"""

import uuid

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

import db.documents as db_documents
import db.identity as db_identity
import scripts.rebuild_qdrant as rebuild
from db.engine import get_sync_engine
from db.models import Document
from rag.identity import sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings


# ---------------------------------------------------------------------------
# Section A: VectorIndex.reconcile_document() direct unit tests.
# ---------------------------------------------------------------------------

@pytest.fixture
def index_factory(tmp_path):
    created = []

    def _make(collection_name="test_collection", embeddings=None):
        vi = VectorIndex(
            persist_directory=tmp_path / "qdrant",
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


def test_active_document_reconciled_to_zero_chunks_removes_all_old_points(index_factory, tmp_path):
    """Required tests 1-3: an already-indexed (non-empty) document whose
    CURRENT source now parses to zero chunks converges Qdrant to zero
    points for it — status 'emptied', chunk_count 0, no points left."""
    vi = index_factory()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Real, non-empty content that indexes into at least one chunk.", encoding="utf-8")

    status, count = vi.reconcile_document("docActiveThenEmpty", file_path)
    assert status == "reindexed"
    assert count >= 1
    assert vi._existing_point_ids("docActiveThenEmpty"), "sanity check: points genuinely exist before the mutation"

    # The durable source is now edited down to empty content — file/
    # sidecar/catalog would all consistently describe this new content in
    # the full-lifecycle scenario (see Section B); this direct-level test
    # isolates VectorIndex's own convergence responsibility.
    file_path.write_text("", encoding="utf-8")

    status2, count2 = vi.reconcile_document("docActiveThenEmpty", file_path)

    assert status2 == "emptied"
    assert count2 == 0
    assert vi._existing_point_ids("docActiveThenEmpty") == set(), "all previously-indexed points must be removed"


def test_stale_text_no_longer_appears_in_similarity_search_after_convergence(index_factory, tmp_path):
    """Required test 4: stale text must no longer be retrievable."""
    vi = index_factory()
    file_path = tmp_path / "doc.txt"
    marker_text = "UNIQUE_STALE_MARKER: content that must not be retrievable after emptying."
    file_path.write_text(marker_text, encoding="utf-8")
    vi.reconcile_document("docStaleRetrieval", file_path)

    requester = str(uuid.uuid4())
    before = vi.similarity_search_with_score(marker_text, requesting_user_uuid=requester, k=5)
    assert any(marker_text in d.page_content for d, _ in before)

    file_path.write_text("", encoding="utf-8")
    status, count = vi.reconcile_document("docStaleRetrieval", file_path)
    assert status == "emptied" and count == 0

    after = vi.similarity_search_with_score(marker_text, requesting_user_uuid=requester, k=5)
    assert all(marker_text not in d.page_content for d, _ in after), "stale content must no longer be retrievable"


def test_pending_zero_chunk_document_still_reconciles_unchanged(index_factory, tmp_path):
    """Required test 7 (direct level): pass #5's pending zero-chunk
    protection is preserved — a document with NO existing points that
    currently has zero chunks classifies as 'unchanged' (nothing to
    remove), never 'emptied'."""
    vi = index_factory()
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    status, count = vi.reconcile_document("docNeverIndexed", file_path)

    assert status == "unchanged"
    assert count == 0
    assert vi._existing_point_ids("docNeverIndexed") == set()


def test_already_zero_qdrant_state_is_idempotent(index_factory, tmp_path, monkeypatch):
    """Required test 8: a rerun against an already-converged (zero-point)
    document makes zero further Qdrant delete calls and stays 'unchanged'."""
    vi = index_factory()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Real content that will later be emptied.", encoding="utf-8")
    vi.reconcile_document("docIdempotent", file_path)
    file_path.write_text("", encoding="utf-8")
    status1, count1 = vi.reconcile_document("docIdempotent", file_path)
    assert status1 == "emptied"
    assert count1 == 0

    delete_calls = {"n": 0}
    real_delete = vi.client.delete

    def counting_delete(*args, **kwargs):
        delete_calls["n"] += 1
        return real_delete(*args, **kwargs)

    monkeypatch.setattr(vi.client, "delete", counting_delete)

    status2, count2 = vi.reconcile_document("docIdempotent", file_path)

    assert status2 == "unchanged"
    assert count2 == 0
    assert delete_calls["n"] == 0, "a second reconciliation against an already-empty Qdrant state must make zero delete calls"
    assert vi._existing_point_ids("docIdempotent") == set()


def test_ordinary_nonempty_rebuild_remains_unchanged(index_factory, tmp_path):
    """Required test 9: the zero-chunk convergence branch must never fire
    for a genuinely non-empty document — ordinary reconciliation
    (reindexed -> unchanged) is untouched by this fix."""
    vi = index_factory()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Ordinary, always non-empty content for a regression baseline.", encoding="utf-8")

    status1, count1 = vi.reconcile_document("docOrdinary", file_path)
    assert status1 == "reindexed"
    assert count1 >= 1

    status2, count2 = vi.reconcile_document("docOrdinary", file_path)
    assert status2 == "unchanged"
    assert count2 == count1


def test_delete_failure_during_convergence_propagates_never_reports_false_success(index_factory, tmp_path, monkeypatch):
    """Ordering safety: if removing stale points fails, that failure must
    propagate (never be swallowed into a status implying convergence
    succeeded), and the stale points must still genuinely be there
    afterward — never a false "converged" report."""
    vi = index_factory()
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Content that will be emptied while delete is broken.", encoding="utf-8")
    vi.reconcile_document("docDeleteFailure", file_path)
    file_path.write_text("", encoding="utf-8")

    def failing_delete(*args, **kwargs):
        raise RuntimeError("simulated transient Qdrant failure")

    monkeypatch.setattr(vi.client, "delete", failing_delete)

    with pytest.raises(RuntimeError):
        vi.reconcile_document("docDeleteFailure", file_path)

    assert vi._existing_point_ids("docDeleteFailure"), (
        "a failed delete must leave existing points untouched, never silently 'succeed'"
    )


# ---------------------------------------------------------------------------
# Section B: full scripts.rebuild_qdrant.apply_plan() lifecycle, against a
# REAL disposable PostgreSQL container.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module needs
    real identity resolution for genuine owner UUIDs with a real backing
    `users` row (documents.owner_user_id's FK target)."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's same-named autouse fixture — this module
    exercises the REAL db.documents functions (mark_active_sync/get_sync)
    against postgres_db; the whole point is proving a real catalog row's
    lifecycle status."""
    yield


@pytest.fixture
def owner(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(881200001)


def _write_active_upload(uploads_dir, uuid_hex, content, display_name, owner_uuid):
    """A managed upload with a valid v3 sidecar AND an ACTIVE PostgreSQL
    catalog row — simulates a document that already completed live
    ingestion successfully and is currently serving real indexed content."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    physical = uploads_dir / f"{uuid_hex}.txt"
    physical.write_bytes(content)
    document_id = upload_document_id(uuid_hex)
    content_sha256 = sha256_hex(content)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(document_id, display_name, physical.name, content_sha256, owner_user_uuid=str(owner_uuid)),
    )
    document_uuid = uuid.UUID(uuid_hex)
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=owner_uuid,
        stored_name=physical.name, display_name=display_name, content_sha256=content_sha256,
    )
    db_documents.mark_active_sync(document_id=document_uuid)
    return physical, document_id, document_uuid


def _write_pending_upload(uploads_dir, uuid_hex, content, display_name, owner_uuid):
    """Mirrors Stage 5C corrective pass #5's own helper: a managed upload
    with a valid v3 sidecar and a still-'pending' catalog row."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    physical = uploads_dir / f"{uuid_hex}.txt"
    physical.write_bytes(content)
    document_id = upload_document_id(uuid_hex)
    content_sha256 = sha256_hex(content)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(document_id, display_name, physical.name, content_sha256, owner_user_uuid=str(owner_uuid)),
    )
    document_uuid = uuid.UUID(uuid_hex)
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=owner_uuid,
        stored_name=physical.name, display_name=display_name, content_sha256=content_sha256,
    )
    return physical, document_id, document_uuid


def _mutate_to_empty_consistently(physical, document_id, document_uuid, display_name, owner_uuid):
    """Edits the durable source down to empty content and rewrites the
    sidecar/catalog content_sha256 to match — simulating an admin editing
    an already-active document's content in place, with EVERY durable
    record (physical file, sidecar, PostgreSQL catalog) left mutually
    consistent about the NEW (empty) content, exactly as Blocker 2
    describes. The catalog row's content_sha256 is updated directly here
    (there is no dedicated db.documents helper for an in-place content
    edit) — this is pure test setup, not the code path under test."""
    physical.write_bytes(b"")
    new_sha256 = sha256_hex(b"")
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(document_id, display_name, physical.name, new_sha256, owner_user_uuid=str(owner_uuid)),
    )
    with Session(get_sync_engine()) as session:
        session.execute(update(Document).where(Document.id == document_uuid).values(content_sha256=new_sha256))
        session.commit()


def _rebuild(documents_dir, uploads_dir, tmp_path, collection_name):
    # reference_filenames=[] (never None): this module's `documents_dir`/
    # `uploads_dir` are test tmp_path directories, not the real
    # config.MANAGED_UPLOADS_DIR — list_source_files() (what
    # reference_filenames=None would trigger) excludes files under the
    # REAL MANAGED_UPLOADS_DIR only, so it would otherwise also pick up
    # each upload's physical file a SECOND time as a synthetic "reference"
    # document via its rglob scan of documents_dir. An empty manifest
    # cleanly enumerates zero reference documents instead, isolating these
    # tests to exactly the upload-document lifecycle under test.
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=[])
    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(),
        collection_name=collection_name,
    )
    try:
        report = rebuild.apply_plan(plan, vi)
        indexed_ids = vi.list_document_ids()
        stats = vi.get_stats(requesting_user_uuid=str(uuid.uuid4()))
    finally:
        vi.close()
    return plan, report, indexed_ids, stats


def test_rebuild_removes_stale_points_for_active_document_reconciled_to_zero(postgres_db, tmp_path, owner):
    """Required tests 1/2/3/5/6: an active document with real Qdrant
    points, whose current source is edited down to zero chunks with file/
    sidecar/catalog left mutually consistent, converges to zero Qdrant
    points on rebuild, reports the condition, and the catalog stays
    'active' (this pass's lifecycle choice — see module docstring)."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uuid_hex = uuid.uuid4().hex
    physical, document_id, document_uuid = _write_active_upload(
        uploads_dir, uuid_hex, b"Real, substantial content that gets indexed with real Qdrant points.",
        "notes.txt", owner,
    )
    collection_name = "corrective6_active_zero"

    plan1, report1, indexed_ids1, _stats1 = _rebuild(documents_dir, uploads_dir, tmp_path, collection_name)
    assert document_id in indexed_ids1, "sanity check: the document is genuinely indexed before the mutation"
    assert report1.documents_emptied == 0

    _mutate_to_empty_consistently(physical, document_id, document_uuid, "notes.txt", owner)

    plan2, report2, indexed_ids2, _stats2 = _rebuild(documents_dir, uploads_dir, tmp_path, collection_name)

    # Required test 3: all old Qdrant points removed.
    assert document_id not in indexed_ids2, "all previously-indexed points for this document must be removed"
    # Required test 6: the report clearly surfaces the convergence.
    assert report2.documents_emptied == 1
    assert report2.documents_reindexed == 0, "an emptied document made zero embedding calls, so it is never 'reindexed'"
    # Required test 5: catalog lifecycle choice — stays 'active'.
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.status == "active", "the lifecycle choice: an already-active row stays active with zero indexed content"
    # Never orphan-pruned either — it's still a recognized plan document,
    # not "absent from source".
    assert report2.documents_removed == 0


def test_stale_content_no_longer_appears_in_retrieval_after_rebuild_convergence(postgres_db, tmp_path, owner, monkeypatch):
    """Required test 4, exercised through the full rebuild -> retrieval
    path (rag.query._validated_similarity_search), not just raw Qdrant
    point membership."""
    import rag.query as rag_query_module

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uuid_hex = uuid.uuid4().hex
    marker_text = b"UNIQUE_REBUILD_STALE_MARKER: must not be retrievable once the source is emptied."
    physical, document_id, document_uuid = _write_active_upload(
        uploads_dir, uuid_hex, marker_text, "notes.txt", owner,
    )
    collection_name = "corrective6_active_zero_retrieval"

    _rebuild(documents_dir, uploads_dir, tmp_path, collection_name)

    _mutate_to_empty_consistently(physical, document_id, document_uuid, "notes.txt", owner)
    _rebuild(documents_dir, uploads_dir, tmp_path, collection_name)

    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(),
        collection_name=collection_name,
    )
    monkeypatch.setattr(rag_query_module, "get_vector_index", lambda: vi)
    try:
        stats = rag_query_module.get_knowledge_base_stats(str(owner))
        assert stats["status"] == "ok"
        search_results = rag_query_module._validated_similarity_search(marker_text.decode(), str(owner), 5)
    finally:
        vi.close()

    assert all(marker_text.decode() not in d.page_content for d, _ in search_results), (
        "stale content must no longer be retrievable after rebuild convergence"
    )


def test_pending_zero_chunk_still_remains_pending_after_this_change(postgres_db, tmp_path, owner):
    """Required test 7 (full rebuild level): Stage 5C corrective pass #5's
    protection is unaffected by this pass — a still-'pending' zero-chunk
    document is never newly activated."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uuid_hex = uuid.uuid4().hex
    physical, document_id, document_uuid = _write_pending_upload(uploads_dir, uuid_hex, b"", "empty.txt", owner)

    plan, report, indexed_ids, _stats = _rebuild(documents_dir, uploads_dir, tmp_path, "corrective6_pending_zero")

    assert report.documents_skipped_zero_chunks == 1
    assert report.documents_emptied == 0, "nothing was ever indexed for it, so there is nothing to converge away"
    assert report.documents_catalog_activated == 0
    assert document_id not in indexed_ids
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.status == "pending"


def test_mixed_valid_active_to_zero_and_pending_to_zero_plan_behaves_independently(postgres_db, tmp_path, owner):
    """Required test 10: an ordinary valid document, an active document
    reconciled to zero, and a still-pending zero-chunk document, all in
    the SAME rebuild run, must be handled completely independently."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    collection_name = "corrective6_mixed"

    ordinary_hex = uuid.uuid4().hex
    _ordinary_physical, ordinary_document_id, ordinary_uuid = _write_pending_upload(
        uploads_dir, ordinary_hex, b"Ordinary content alongside its siblings in a mixed rebuild plan.",
        "ordinary.txt", owner,
    )

    active_hex = uuid.uuid4().hex
    active_physical, active_document_id, active_uuid = _write_active_upload(
        uploads_dir, active_hex, b"Active content that will be emptied out on the second rebuild pass.",
        "active.txt", owner,
    )

    pending_zero_hex = uuid.uuid4().hex
    _pending_physical, pending_zero_document_id, pending_zero_uuid = _write_pending_upload(
        uploads_dir, pending_zero_hex, b"", "pending-empty.txt", owner,
    )

    plan1, report1, indexed_ids1, _stats1 = _rebuild(documents_dir, uploads_dir, tmp_path, collection_name)
    assert ordinary_document_id in indexed_ids1
    assert active_document_id in indexed_ids1
    assert pending_zero_document_id not in indexed_ids1
    assert report1.documents_skipped_zero_chunks == 1  # the pending-empty document only

    _mutate_to_empty_consistently(active_physical, active_document_id, active_uuid, "active.txt", owner)

    plan2, report2, indexed_ids2, _stats2 = _rebuild(documents_dir, uploads_dir, tmp_path, collection_name)

    assert ordinary_document_id in indexed_ids2, "the ordinary document must be completely unaffected"
    assert active_document_id not in indexed_ids2, "the now-emptied document's stale points must be gone"
    assert pending_zero_document_id not in indexed_ids2, "the still-pending zero-chunk document remains unindexed"

    assert report2.documents_emptied == 1
    assert report2.documents_skipped_zero_chunks == 2  # active-emptied + still-pending-zero, independently

    assert db_documents.get_sync(document_id=ordinary_uuid).status == "active"
    assert db_documents.get_sync(document_id=active_uuid).status == "active"
    assert db_documents.get_sync(document_id=pending_zero_uuid).status == "pending"
