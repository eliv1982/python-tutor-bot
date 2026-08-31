"""
Stage 5C corrective pass #5 regression tests (Blocker 2):
scripts/rebuild_qdrant.py's apply_plan() must never NEWLY activate an
"upload" document whose current reconciliation produces ZERO chunks (an
empty or whitespace-only source).

Live ingestion already refuses this (app.documents._load_and_index_document()
raises EmptyDocumentError when reconcile_document() returns chunk_count==0 —
see tests/test_stage5c_documents_catalog.py's Section F) — an independent
audit reproduced rebuild NOT enforcing the identical rule: apply_plan() used
to call db.documents.mark_active_sync() unconditionally after
reconcile_document() succeeded, regardless of chunk_count, activating a
still-'pending' catalog row for a document with genuinely zero indexed
Qdrant points.

Against a REAL disposable PostgreSQL container (see tests/conftest.py's
postgres_container()/postgres_db()) plus a real local-persistent Qdrant with
deterministic fake embeddings (tests/rag_fakes.py) — proving genuine
catalog-lifecycle behavior, never a mocked stand-in.
"""

import uuid

import pytest

import db.documents as db_documents
import db.identity as db_identity
import scripts.rebuild_qdrant as rebuild
from rag.identity import sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings


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
    return db_identity.resolve_or_create_user_by_telegram_id_sync(881100001)


def _write_pending_upload(uploads_dir, uuid_hex, content, display_name, owner_uuid):
    """A managed upload with a valid v3 sidecar and a 'pending' PostgreSQL
    catalog row — deliberately never mark_active_sync()'d, mirroring a
    real upload whose live ingestion never confirmed activation (or, for
    this module's purposes, a fresh document rebuild is about to plan)."""
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


def _rebuild(documents_dir, uploads_dir, tmp_path, collection_name):
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(),
        collection_name=collection_name,
    )
    try:
        report = rebuild.apply_plan(plan, vi)
        indexed_ids = vi.list_document_ids()
    finally:
        vi.close()
    return plan, report, indexed_ids


# ---------------------------------------------------------------------------
# 1/2/3/4: pending zero-chunk documents (empty and whitespace-only) are
# never activated, produce no Qdrant points, and are surfaced explicitly in
# the rebuild report.
# ---------------------------------------------------------------------------

def test_pending_empty_text_document_remains_non_active(postgres_db, tmp_path, owner):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uuid_hex = uuid.uuid4().hex
    physical, document_id, document_uuid = _write_pending_upload(uploads_dir, uuid_hex, b"", "empty.txt", owner)

    plan, report, indexed_ids = _rebuild(documents_dir, uploads_dir, tmp_path, "zero_chunk_rebuild_empty")

    assert len(plan.upload_documents) == 1
    assert not plan.skipped_upload_reasons
    assert report.documents_skipped_zero_chunks == 1
    assert report.documents_catalog_activated == 0
    assert document_id not in indexed_ids, "an empty document must produce zero Qdrant points"
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.status == "pending", "a zero-chunk document must never be newly activated"


def test_pending_whitespace_only_document_remains_non_active(postgres_db, tmp_path, owner):
    """Whitespace-only content that the REAL RecursiveCharacterTextSplitter
    genuinely reduces to zero chunks — same rule as a literally-empty
    file, mirroring app.documents's own whitespace-only test."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uuid_hex = uuid.uuid4().hex
    physical, document_id, document_uuid = _write_pending_upload(
        uploads_dir, uuid_hex, b"   \n\n\t  \n   ", "whitespace.txt", owner
    )

    plan, report, indexed_ids = _rebuild(documents_dir, uploads_dir, tmp_path, "zero_chunk_rebuild_whitespace")

    assert report.documents_skipped_zero_chunks == 1
    assert report.documents_catalog_activated == 0
    assert document_id not in indexed_ids
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.status == "pending"


def test_zero_chunk_document_is_not_pruned_as_an_orphan(postgres_db, tmp_path, owner):
    """A zero-chunk document must be treated as "could not be usefully
    indexed", never as "absent from source" — orphan pruning must not
    delete anything on its account, and a rerun must still see it as a
    normal (still-pending, still-skippable) candidate."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uuid_hex = uuid.uuid4().hex
    physical, document_id, document_uuid = _write_pending_upload(uploads_dir, uuid_hex, b"", "empty.txt", owner)

    plan, report, indexed_ids = _rebuild(documents_dir, uploads_dir, tmp_path, "zero_chunk_rebuild_no_prune")

    assert report.documents_removed == 0
    assert report.orphan_pruning_skipped_incomplete_plan is False
    # The physical file/sidecar are themselves untouched by rebuild — a
    # rerun still finds exactly the same, still-recoverable candidate.
    assert physical.exists()
    rerun_plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(rerun_plan.upload_documents) == 1
    assert rerun_plan.upload_documents[0].document_id == document_id


# ---------------------------------------------------------------------------
# 5/6: an ordinary (non-empty) document still rebuilds and activates
# normally — the fix must not over-reject genuinely valid documents, and
# ordinary pending-recovery behavior (Section 3's deterministic
# reconciliation path) remains intact.
# ---------------------------------------------------------------------------

def test_ordinary_pending_document_still_activates_via_rebuild(postgres_db, tmp_path, owner):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uuid_hex = uuid.uuid4().hex
    physical, document_id, document_uuid = _write_pending_upload(
        uploads_dir, uuid_hex, b"A genuinely non-empty document with real content to index.", "notes.txt", owner
    )

    plan, report, indexed_ids = _rebuild(documents_dir, uploads_dir, tmp_path, "zero_chunk_rebuild_ordinary")

    assert report.documents_skipped_zero_chunks == 0
    assert report.documents_catalog_activated == 1
    assert document_id in indexed_ids
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.status == "active"


def test_mixed_zero_chunk_and_ordinary_documents_are_handled_independently(postgres_db, tmp_path, owner):
    """A zero-chunk document and an ordinary document in the SAME rebuild
    run must be handled independently — one skip must never block or
    affect the other's normal activation."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    empty_hex = uuid.uuid4().hex
    ordinary_hex = uuid.uuid4().hex
    _empty_physical, empty_document_id, empty_uuid = _write_pending_upload(uploads_dir, empty_hex, b"", "empty.txt", owner)
    _ordinary_physical, ordinary_document_id, ordinary_uuid = _write_pending_upload(
        uploads_dir, ordinary_hex, b"Ordinary content alongside a zero-chunk sibling document.", "ordinary.txt", owner
    )

    plan, report, indexed_ids = _rebuild(documents_dir, uploads_dir, tmp_path, "zero_chunk_rebuild_mixed")

    assert report.documents_skipped_zero_chunks == 1
    assert report.documents_catalog_activated == 1
    assert empty_document_id not in indexed_ids
    assert ordinary_document_id in indexed_ids
    assert db_documents.get_sync(document_id=empty_uuid).status == "pending"
    assert db_documents.get_sync(document_id=ordinary_uuid).status == "active"
