"""
Stage 3B regression tests: proves scripts/rebuild_qdrant.py's rebuild-from-
source path (build_plan() -> apply_plan() -> VectorIndex.reconcile_document())
preserves the exact Stage 3A multi-user visibility/ownership boundary —
private scope + exact owner, shared reference scope, cross-user isolation,
fail-closed legacy handling, and non-destructive reconcile-before-prune —
end to end through the REAL rebuild script, not just VectorIndex directly.

The Stage 3B gap analysis (see project history) concluded production
behavior is already correct and only regression proof was missing at the
rebuild-script layer. This module is test-only: it adds no production code
and expects none of scripts/rebuild_qdrant.py, rag/index.py, rag/sidecar.py,
or rag/identity.py to change as a result.

Entirely temporary fixtures: a synthetic documents directory shaped like the
real built-in reference manifest, a synthetic uploads directory, deterministic
local fake embeddings, and a temporary Qdrant path — same convention as
tests/test_stage2b_rebuild.py and tests/test_stage3a_multiuser_isolation.py.
No real documents, no real Qdrant, no provider calls anywhere in this module.
"""

import json

import pytest
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

import scripts.rebuild_qdrant as rebuild
from rag.identity import point_id, reference_document_id, sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings

import config as app_config


def _write_upload(uploads_dir, uuid_hex, extension, content, display_name, owner_user_id):
    """Same shape as test_stage2b_rebuild.py's _write_upload(), but always
    requires an explicit owner_user_id (every test in this module is about
    ownership, so there is deliberately no default)."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    physical = uploads_dir / f"{uuid_hex}{extension}"
    physical.write_bytes(content)
    document_id = upload_document_id(uuid_hex)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(document_id, display_name, physical.name, sha256_hex(content), owner_user_id=owner_user_id),
    )
    return physical, document_id


def _payloads_for(vi, document_id):
    """Every stored Qdrant payload dict currently indexed under
    `document_id` — direct payload inspection (not just retrieval), so
    tests can assert persisted `scope`/`owner_user_id` metadata itself."""
    records, _ = vi.client.scroll(
        collection_name=vi.collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
        limit=256,
        with_payload=True,
    )
    return [r.payload for r in records]


@pytest.fixture
def multi_owner_tree(tmp_path, monkeypatch):
    """Reference-document directory shaped like the real built-in manifest
    (config.BUILTIN_REFERENCE_FILES) plus a managed uploads directory —
    matches test_stage2b_rebuild.py's manifest_source_tree fixture, so
    build_plan()'s DEFAULT (non-None) reference_filenames path — what real
    callers (main()) actually use — is exercised, not the low-level generic
    directory-scan escape hatch."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    for filename in app_config.BUILTIN_REFERENCE_FILES:
        (documents_dir / filename).write_text(f"Reference content for {filename}.", encoding="utf-8")

    uploads_dir = documents_dir / "uploads"

    import rag.loader as rag_loader
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", uploads_dir)

    return documents_dir, uploads_dir


# ---------------------------------------------------------------------------
# Proof 1: multi-owner rebuild — two managed uploads with DIFFERENT owners,
# rebuilt through the real build_plan()/apply_plan() path, end up correctly
# scoped/owned and mutually isolated; shared reference stays visible to both.
# ---------------------------------------------------------------------------

def test_rebuild_multi_owner_private_isolation(multi_owner_tree, tmp_path):
    documents_dir, uploads_dir = multi_owner_tree
    content_a = b"User A's private tutoring notes on recursion."
    content_b = b"User B's private tutoring notes on closures."
    _physical_a, doc_id_a = _write_upload(uploads_dir, "1" * 32, ".txt", content_a, "notes_a.txt", owner_user_id=1001)
    _physical_b, doc_id_b = _write_upload(uploads_dir, "2" * 32, ".txt", content_b, "notes_b.txt", owner_user_id=2002)

    plan = rebuild.build_plan(documents_dir, uploads_dir)
    assert not plan.skipped_upload_reasons
    assert len(plan.upload_documents) == 2

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="multi_owner_test")
    try:
        report = rebuild.apply_plan(plan, vi)
        assert report.documents_removed == 0

        payloads_a = _payloads_for(vi, doc_id_a)
        payloads_b = _payloads_for(vi, doc_id_b)
        assert payloads_a and all(p["scope"] == "private" and p["owner_user_id"] == 1001 for p in payloads_a)
        assert payloads_b and all(p["scope"] == "private" and p["owner_user_id"] == 2002 for p in payloads_b)

        # User A retrieves A's own document, never B's.
        results_a = vi.similarity_search_with_score(content_a.decode(), requesting_user_id=1001, k=5)
        assert any(doc.metadata.get("document_id") == doc_id_a for doc, _ in results_a)
        assert all(doc.metadata.get("document_id") != doc_id_b for doc, _ in results_a)

        # User B retrieves B's own document, never A's — even querying with
        # A's exact private text.
        results_b_query_a = vi.similarity_search_with_score(content_a.decode(), requesting_user_id=2002, k=5)
        assert all(doc.metadata.get("document_id") != doc_id_a for doc, _ in results_b_query_a)
        results_b = vi.similarity_search_with_score(content_b.decode(), requesting_user_id=2002, k=5)
        assert any(doc.metadata.get("document_id") == doc_id_b for doc, _ in results_b)

        # Shared reference content remains visible to both users.
        ref_filename = app_config.BUILTIN_REFERENCE_FILES[0]
        ref_content = f"Reference content for {ref_filename}."
        for uid in (1001, 2002):
            ref_results = vi.similarity_search_with_score(ref_content, requesting_user_id=uid, k=5)
            assert any(doc.metadata.get("source") == ref_filename for doc, _ in ref_results)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Proof 2: identical content, different owners, through the REAL rebuild
# path (not just direct add_documents() as in Stage 3A's own test) — proves
# document/point identity is derived from storage identity, never content,
# so identical bytes from two different uploaders never collide.
# ---------------------------------------------------------------------------

def test_rebuild_identical_content_different_owners_remain_distinct(multi_owner_tree, tmp_path):
    documents_dir, uploads_dir = multi_owner_tree
    shared_content = b"Identical private wording uploaded independently by two different students."
    _physical_c, doc_id_c = _write_upload(uploads_dir, "3" * 32, ".txt", shared_content, "shared_c.txt", owner_user_id=3001)
    _physical_d, doc_id_d = _write_upload(uploads_dir, "4" * 32, ".txt", shared_content, "shared_d.txt", owner_user_id=4002)
    assert doc_id_c != doc_id_d

    plan = rebuild.build_plan(documents_dir, uploads_dir)
    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="identical_content_test")
    try:
        report = rebuild.apply_plan(plan, vi)
        assert report.documents_removed == 0

        payloads_c = _payloads_for(vi, doc_id_c)
        payloads_d = _payloads_for(vi, doc_id_d)
        assert len(payloads_c) == 1 and len(payloads_d) == 1
        assert payloads_c[0]["owner_user_id"] == 3001
        assert payloads_d[0]["owner_user_id"] == 4002
        assert payloads_c[0]["scope"] == payloads_d[0]["scope"] == "private"

        # Distinct point identities despite byte-for-byte identical embedded
        # text (DeterministicFakeEmbeddings would produce the SAME vector
        # for both — isolation here comes entirely from document_id/filter,
        # never from vector distance).
        point_c = point_id(doc_id_c, 0)
        point_d = point_id(doc_id_d, 0)
        assert point_c != point_d
        assert vi._existing_point_ids(doc_id_c) == {point_c}
        assert vi._existing_point_ids(doc_id_d) == {point_d}

        text = shared_content.decode()
        results_c = vi.similarity_search_with_score(text, requesting_user_id=3001, k=5)
        results_d = vi.similarity_search_with_score(text, requesting_user_id=4002, k=5)
        assert any(doc.metadata.get("document_id") == doc_id_c for doc, _ in results_c)
        assert all(doc.metadata.get("document_id") != doc_id_d for doc, _ in results_c)
        assert any(doc.metadata.get("document_id") == doc_id_d for doc, _ in results_d)
        assert all(doc.metadata.get("document_id") != doc_id_c for doc, _ in results_d)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Proof 3: reference payload restoration — asserts the PERSISTED payload
# metadata (not just retrievability) after a real rebuild apply.
# ---------------------------------------------------------------------------

def test_rebuild_reference_payload_scope_and_multi_user_visibility(multi_owner_tree, tmp_path):
    documents_dir, uploads_dir = multi_owner_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir)
    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="reference_payload_test")
    try:
        report = rebuild.apply_plan(plan, vi)
        assert report.documents_removed == 0

        ref_filename = app_config.BUILTIN_REFERENCE_FILES[0]
        ref_doc_id = reference_document_id(ref_filename)
        payloads = _payloads_for(vi, ref_doc_id)
        assert payloads
        for payload in payloads:
            assert payload["scope"] == "reference"
            assert "owner_user_id" not in payload

        # Visible to multiple, unrelated requesting users — never
        # owner-gated like a private document would be.
        ref_content = f"Reference content for {ref_filename}."
        for uid in (7001, 7002):
            results = vi.similarity_search_with_score(ref_content, requesting_user_id=uid, k=5)
            assert any(doc.metadata.get("document_id") == ref_doc_id for doc, _ in results)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Proof 4: a source that fails planning (ownerless legacy sidecar) is never
# reconciled under guessed ownership, and its PRE-EXISTING Qdrant points are
# completely removed by orphan pruning on the next --apply — locking in the
# "Qdrant is derived state; untrusted source is not retained as accessible
# index state" model traced during the Stage 3B gap analysis.
# ---------------------------------------------------------------------------

def test_rebuild_skipped_ownerless_source_orphan_pruned_completely(multi_owner_tree, tmp_path):
    documents_dir, uploads_dir = multi_owner_tree
    content = b"Legacy upload content that will lose its recorded owner."
    stem = "5" * 32
    physical, doc_id = _write_upload(uploads_dir, stem, ".txt", content, "legacy_before.txt", owner_user_id=5001)

    other_content = b"Unrelated valid upload content for a different owner."
    _physical_other, doc_id_other = _write_upload(uploads_dir, "6" * 32, ".txt", other_content, "other.txt", owner_user_id=6002)

    plan1 = rebuild.build_plan(documents_dir, uploads_dir)
    assert not plan1.skipped_upload_reasons

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="orphan_prune_test")
    try:
        report1 = rebuild.apply_plan(plan1, vi)
        assert report1.documents_removed == 0
        assert vi._existing_point_ids(doc_id)
        ref_filename = app_config.BUILTIN_REFERENCE_FILES[0]
        ref_doc_id = reference_document_id(ref_filename)
        assert vi._existing_point_ids(ref_doc_id)

        # The durable source now fails planning: sidecar downgraded to a v1
        # (pre-Stage-3A) record with no recorded owner at all — same
        # document_id/stored_name/content_sha256, exactly the shape a real
        # legacy sidecar has (rag.sidecar.REQUIRED_FIELDS_V1).
        legacy_sidecar = {
            "schema_version": 1,
            "document_id": doc_id,
            "display_name": "legacy_before.txt",
            "stored_name": physical.name,
            "content_sha256": sha256_hex(content),
        }
        sidecar_path_for(physical).write_text(json.dumps(legacy_sidecar), encoding="utf-8")

        plan2 = rebuild.build_plan(documents_dir, uploads_dir)
        assert "missing_owner" in plan2.skipped_upload_reasons
        assert doc_id not in {d.document_id for d in plan2.all_documents}

        report2 = rebuild.apply_plan(plan2, vi)
        assert report2.documents_removed == 1

        # Never reconciled under guessed ownership — completely absent, not
        # reclassified as reference or silently re-attributed to any owner.
        assert not vi._existing_point_ids(doc_id)
        assert doc_id not in vi.list_document_ids()

        # Unrelated valid documents remain intact, with their own metadata
        # untouched.
        assert vi._existing_point_ids(doc_id_other)
        assert vi._existing_point_ids(ref_doc_id)
        payloads_other = _payloads_for(vi, doc_id_other)
        assert payloads_other[0]["scope"] == "private"
        assert payloads_other[0]["owner_user_id"] == 6002
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Proof 5: cross-owner failure preservation — a failure reconciling one
# owner's document during apply_plan() must never disturb a DIFFERENT
# owner's already-valid private document (reconcile-before-prune, extended
# to prove the preserved data's owner/scope specifically, not just its mere
# presence).
# ---------------------------------------------------------------------------

def test_rebuild_failure_preserves_other_owners_private_document(multi_owner_tree, tmp_path, monkeypatch):
    documents_dir, uploads_dir = multi_owner_tree
    content_a = b"Owner A's already-valid private document, present before the failing rebuild."
    _physical_a, doc_id_a = _write_upload(uploads_dir, "7" * 32, ".txt", content_a, "owner_a.txt", owner_user_id=8001)

    plan1 = rebuild.build_plan(documents_dir, uploads_dir)
    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="cross_owner_failure_test")
    try:
        report1 = rebuild.apply_plan(plan1, vi)
        assert report1.documents_removed == 0
        payloads_a_before = _payloads_for(vi, doc_id_a)
        assert payloads_a_before[0]["scope"] == "private"
        assert payloads_a_before[0]["owner_user_id"] == 8001

        # A second owner's brand-new upload is added to source truth. On
        # this second plan, owner A's document and the whole reference set
        # are already current (zero-embedding "unchanged" fast path), so
        # owner B's document is the ONLY thing that will actually call
        # embed_documents() this run — sabotaging it in-place reproduces the
        # exact "later/only document fails" shape Stage 2B's rebuild tests
        # already use, without inventing a new failure mechanism.
        content_b = b"Owner B's new private document whose embedding will fail."
        _physical_b, doc_id_b = _write_upload(uploads_dir, "9" * 32, ".txt", content_b, "owner_b.txt", owner_user_id=9002)
        plan2 = rebuild.build_plan(documents_dir, uploads_dir)
        assert not plan2.skipped_upload_reasons

        def failing_embed_documents(texts):
            raise RuntimeError("simulated provider failure reconciling owner B's document")

        monkeypatch.setattr(fake, "embed_documents", failing_embed_documents)

        with pytest.raises(RuntimeError):
            rebuild.apply_plan(plan2, vi)

        # Reconcile-before-prune: the failure aborts before orphan pruning
        # ever runs, and owner A's already-valid document — payload,
        # scope, and owner — is byte-for-byte untouched.
        payloads_a_after = _payloads_for(vi, doc_id_a)
        assert payloads_a_after == payloads_a_before
        # Owner B's document never got any points at all (embedding failed
        # before any upsert) — never partially indexed under guessed data.
        assert not vi._existing_point_ids(doc_id_b)

        results_owner = vi.similarity_search_with_score(content_a.decode(), requesting_user_id=8001, k=5)
        assert any(doc.metadata.get("document_id") == doc_id_a for doc, _ in results_owner)
        results_other = vi.similarity_search_with_score(content_a.decode(), requesting_user_id=9002, k=5)
        assert all(doc.metadata.get("document_id") != doc_id_a for doc, _ in results_other)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Proof 6: destroy/rebuild visibility parity — the effective visibility
# boundary (per-user retrieval, per-user stats, persisted scope/owner
# metadata) established via normal live-style indexing primitives
# (VectorIndex.reconcile_document(), exactly as index_documents_directory()
# and handlers/document_upload.py call it) survives a clear_index() +
# rebuild-from-source-only round trip, byte-for-byte on the metadata that
# matters and set-membership (never list-order) on retrieval.
# ---------------------------------------------------------------------------

def test_rebuild_reproduces_live_index_visibility_parity(multi_owner_tree, tmp_path):
    documents_dir, uploads_dir = multi_owner_tree
    content_a = b"Parity check: user A's private document content."
    content_b = b"Parity check: user B's private document content."
    physical_a, doc_id_a = _write_upload(uploads_dir, "a" * 32, ".txt", content_a, "parity_a.txt", owner_user_id=11001)
    physical_b, doc_id_b = _write_upload(uploads_dir, "b" * 32, ".txt", content_b, "parity_b.txt", owner_user_id=12002)

    ref_filename = app_config.BUILTIN_REFERENCE_FILES[0]
    ref_doc_id = reference_document_id(ref_filename)
    ref_physical = documents_dir / ref_filename
    ref_content = ref_physical.read_text(encoding="utf-8")

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="parity_test")
    try:
        # Establish the index via the same primitives real callers use:
        # index_documents_directory()'s own reference call shape (applied to
        # EVERY manifest file, exactly like a real startup indexes the whole
        # built-in manifest, not just one file — otherwise the "before"
        # stats baseline would undercount relative to what rebuild legitimately
        # restores), and handlers/document_upload.py's own managed-upload
        # call shape for the two private documents.
        for filename in app_config.BUILTIN_REFERENCE_FILES:
            vi.reconcile_document(reference_document_id(filename), documents_dir / filename)
        vi.reconcile_document(
            doc_id_a, physical_a,
            display_name="parity_a.txt", stored_name=physical_a.name,
            expected_content_sha256=sha256_hex(content_a), source_bytes=content_a,
            owner_user_id=11001,
        )
        vi.reconcile_document(
            doc_id_b, physical_b,
            display_name="parity_b.txt", stored_name=physical_b.name,
            expected_content_sha256=sha256_hex(content_b), source_bytes=content_b,
            owner_user_id=12002,
        )

        def _boundary():
            def _sees(query_text, target_doc_id, requesting_user_id):
                results = vi.similarity_search_with_score(query_text, requesting_user_id=requesting_user_id, k=5)
                return any(doc.metadata.get("document_id") == target_doc_id for doc, _ in results)

            return {
                "stats": {uid: vi.get_stats(requesting_user_id=uid)["total_documents"] for uid in (11001, 12002)},
                "a_sees_a": _sees(content_a.decode(), doc_id_a, 11001),
                "a_sees_b": _sees(content_b.decode(), doc_id_b, 11001),
                "b_sees_b": _sees(content_b.decode(), doc_id_b, 12002),
                "b_sees_a": _sees(content_a.decode(), doc_id_a, 12002),
                "a_sees_ref": _sees(ref_content, ref_doc_id, 11001),
                "b_sees_ref": _sees(ref_content, ref_doc_id, 12002),
                "payload_a": sorted(_payloads_for(vi, doc_id_a), key=lambda p: p["chunk_index"]),
                "payload_b": sorted(_payloads_for(vi, doc_id_b), key=lambda p: p["chunk_index"]),
                "payload_ref": sorted(_payloads_for(vi, ref_doc_id), key=lambda p: p["chunk_index"]),
            }

        boundary_before = _boundary()
        # Sanity: the boundary actually distinguishes the two users before
        # comparing it post-rebuild (a test that "proves parity" against a
        # trivial/empty boundary would prove nothing).
        assert boundary_before["a_sees_a"] and boundary_before["b_sees_b"]
        assert not boundary_before["a_sees_b"] and not boundary_before["b_sees_a"]
        assert boundary_before["a_sees_ref"] and boundary_before["b_sees_ref"]

        vi.clear_index()
        assert vi.list_document_ids() == set()

        plan = rebuild.build_plan(documents_dir, uploads_dir)
        report = rebuild.apply_plan(plan, vi)
        assert report.documents_removed == 0

        boundary_after = _boundary()

        assert boundary_after["stats"] == boundary_before["stats"]
        assert boundary_after["a_sees_a"] == boundary_before["a_sees_a"]
        assert boundary_after["a_sees_b"] == boundary_before["a_sees_b"]
        assert boundary_after["b_sees_b"] == boundary_before["b_sees_b"]
        assert boundary_after["b_sees_a"] == boundary_before["b_sees_a"]
        assert boundary_after["a_sees_ref"] == boundary_before["a_sees_ref"]
        assert boundary_after["b_sees_ref"] == boundary_before["b_sees_ref"]

        # Semantic parity on the metadata that defines the visibility
        # boundary itself — content, scope, and owner — never on Qdrant's
        # own internal point-id/ordering, which is an implementation detail.
        def _semantic(payloads):
            return [{k: v for k, v in p.items() if k != "text"} for p in payloads]

        assert _semantic(boundary_after["payload_a"]) == _semantic(boundary_before["payload_a"])
        assert _semantic(boundary_after["payload_b"]) == _semantic(boundary_before["payload_b"])
        assert _semantic(boundary_after["payload_ref"]) == _semantic(boundary_before["payload_ref"])
    finally:
        vi.close()
