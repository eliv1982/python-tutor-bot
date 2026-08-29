"""
Stage 2B-B regression tests: scripts/rebuild_qdrant.py, the operator-facing
rebuild-FROM-SOURCE utility (Section R/W). This is deliberately NOT a
Chroma migration tool — it never reads legacy Chroma, only the built-in
Markdown reference documents and sidecar-backed managed uploads.

Entirely temporary fixtures: a synthetic documents directory, a synthetic
uploads directory, deterministic local fake embeddings, and a temporary
Qdrant path. No real documents, no real Chroma, no real Qdrant, no
provider calls anywhere in this module.
"""

import inspect
import json
import os
from pathlib import Path

import pytest

import scripts.rebuild_qdrant as rebuild
from rag.identity import point_id, reference_document_id, sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings


def _write_upload(uploads_dir: Path, uuid_hex: str, extension: str, content: bytes, display_name: str) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    physical = uploads_dir / f"{uuid_hex}{extension}"
    physical.write_bytes(content)
    document_id = upload_document_id(uuid_hex)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(document_id, display_name, physical.name, sha256_hex(content)),
    )
    return physical


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "guide-one.md").write_text("First reference guide content.", encoding="utf-8")
    (documents_dir / "guide-two.md").write_text("Second reference guide content, distinct text.", encoding="utf-8")

    uploads_dir = documents_dir / "uploads"
    _write_upload(uploads_dir, "a" * 32, ".txt", b"first upload content", "shared_name.txt")
    _write_upload(uploads_dir, "b" * 32, ".txt", b"second upload content", "shared_name.txt")  # duplicate display name

    # rag.loader.list_source_files() excludes MANAGED_UPLOADS_DIR (bound
    # at rag.loader's own import time, i.e. the REAL repo path) from its
    # reference-document scan — redirect it to this fixture's temp uploads
    # dir so the reference scan doesn't also pick up the upload files.
    import rag.loader as rag_loader
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", uploads_dir)

    return documents_dir, uploads_dir


@pytest.fixture
def manifest_source_tree(tmp_path, monkeypatch):
    """Like `source_tree`, but the reference-document directory is shaped
    like a REAL manifest root: files literally named after
    config.BUILTIN_REFERENCE_FILES (Stage 2B-C Blocker 5), so
    build_plan(..., reference_filenames=config.BUILTIN_REFERENCE_FILES) —
    what main()/the real CLI actually uses — succeeds against it."""
    import config as app_config

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    for filename in app_config.BUILTIN_REFERENCE_FILES:
        (documents_dir / filename).write_text(f"Reference content for {filename}.", encoding="utf-8")

    uploads_dir = documents_dir / "uploads"
    _write_upload(uploads_dir, "a" * 32, ".txt", b"first upload content", "shared_name.txt")
    _write_upload(uploads_dir, "b" * 32, ".txt", b"second upload content", "shared_name.txt")

    import rag.loader as rag_loader
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", uploads_dir)

    return documents_dir, uploads_dir


# ---------------------------------------------------------------------------
# Plan building (used by both dry-run and apply)
# ---------------------------------------------------------------------------

def test_build_plan_finds_reference_and_upload_documents(source_tree):
    """source_tree's two guide-*.md files deliberately do NOT match
    config.BUILTIN_REFERENCE_FILES — this test exercises the low-level
    generic directory scan explicitly via reference_filenames=None, never
    the (now safe-by-default) unqualified call. See
    test_build_plan_generic_scan_requires_explicit_none_and_is_not_the_default
    for the regression proof that omitting the argument no longer reaches
    this scan."""
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(plan.reference_documents) == 2
    assert {d.display_source for d in plan.reference_documents} == {"guide-one.md", "guide-two.md"}
    assert len(plan.upload_documents) == 2
    assert not plan.skipped_upload_reasons


def test_build_plan_makes_no_embedding_or_qdrant_calls(source_tree, monkeypatch):
    """Dry-run proof #2: purely local filesystem reads + hashing."""
    documents_dir, uploads_dir = source_tree

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build_plan() must never touch a provider/Qdrant client")

    # Nothing in build_plan should import/construct a VectorIndex or an
    # embeddings client at all; sabotage the constructor to prove it.
    monkeypatch.setattr(VectorIndex, "__init__", fail_if_called)

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(plan.all_documents) == 4


# ---------------------------------------------------------------------------
# Stage 2B-C1 correction: build_plan() must be safe BY DEFAULT — a caller
# who omits reference_filenames must get EXACTLY config.BUILTIN_REFERENCE_
# FILES, never scripts.rebuild_qdrant's previous silent fallback to an
# unconstrained directory scan. Mirrors
# VectorIndex.index_documents_directory()'s existing safe default.
# ---------------------------------------------------------------------------

def test_build_plan_default_enumerates_exactly_the_builtin_manifest(manifest_source_tree):
    import config as app_config
    documents_dir, uploads_dir = manifest_source_tree

    plan = rebuild.build_plan(documents_dir, uploads_dir)  # no reference_filenames passed

    assert len(plan.reference_documents) == len(app_config.BUILTIN_REFERENCE_FILES)
    assert {d.display_source for d in plan.reference_documents} == set(app_config.BUILTIN_REFERENCE_FILES)
    # Managed uploads remain separately discovered via valid sidecars,
    # unaffected by the reference-manifest default.
    assert len(plan.upload_documents) == 2
    assert not plan.skipped_upload_reasons


def test_build_plan_default_excludes_stray_legacy_txt_beside_manifest(manifest_source_tree):
    documents_dir, uploads_dir = manifest_source_tree
    (documents_dir / "old_legacy_notes.txt").write_text("Stale legacy content.", encoding="utf-8")

    import config as app_config
    plan = rebuild.build_plan(documents_dir, uploads_dir)  # no reference_filenames passed

    assert len(plan.reference_documents) == len(app_config.BUILTIN_REFERENCE_FILES)
    assert "old_legacy_notes.txt" not in {d.display_source for d in plan.reference_documents}


def test_build_plan_default_excludes_arbitrary_extra_markdown_beside_manifest(manifest_source_tree):
    documents_dir, uploads_dir = manifest_source_tree
    (documents_dir / "unrelated_extra.md").write_text("Some extra markdown.", encoding="utf-8")

    import config as app_config
    plan = rebuild.build_plan(documents_dir, uploads_dir)  # no reference_filenames passed

    assert len(plan.reference_documents) == len(app_config.BUILTIN_REFERENCE_FILES)
    assert "unrelated_extra.md" not in {d.display_source for d in plan.reference_documents}


def test_build_plan_default_fails_clearly_when_a_manifest_file_is_missing(manifest_source_tree):
    """A missing manifest file must be a hard, reported failure — never a
    silently reduced plan. Proven directly against build_plan() itself
    (not just main()'s CLI wrapper) using the DEFAULT, no-argument call."""
    import config as app_config
    documents_dir, uploads_dir = manifest_source_tree
    (documents_dir / app_config.BUILTIN_REFERENCE_FILES[0]).unlink()

    from rag.loader import MissingReferenceDocumentError
    with pytest.raises(MissingReferenceDocumentError):
        rebuild.build_plan(documents_dir, uploads_dir)  # no reference_filenames passed


def test_build_plan_generic_scan_requires_explicit_none_and_is_not_the_default(source_tree):
    """Regression test for the Stage 2B-C1 fix. Before it, omitting
    reference_filenames silently fell back to an unconstrained directory
    scan (list_source_files()) — this is exactly the footgun disclosed in
    the Stage 2B-C report. Now: the unconstrained scan is reachable ONLY
    via an explicit reference_filenames=None (used by this module's own
    apply_plan()/reconciliation tests against a synthetic two-guide corpus
    that intentionally does not match config.BUILTIN_REFERENCE_FILES), and
    the default, no-argument call rejects that same non-manifest corpus
    instead of silently scanning it."""
    documents_dir, uploads_dir = source_tree

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert {d.display_source for d in plan.reference_documents} == {"guide-one.md", "guide-two.md"}

    from rag.loader import MissingReferenceDocumentError
    with pytest.raises(MissingReferenceDocumentError):
        rebuild.build_plan(documents_dir, uploads_dir)  # default now rejects this non-manifest corpus


# ---------------------------------------------------------------------------
# 1/2/3: dry run writes nothing, makes zero embedding calls, zero Qdrant
# mutation (main() with no --apply)
# ---------------------------------------------------------------------------

def test_cli_dry_run_by_default_makes_no_mutation(manifest_source_tree, monkeypatch, tmp_path, capsys):
    documents_dir, uploads_dir = manifest_source_tree

    # Stage 2B-D Section D: main() now sources DOCUMENTS_DIR/
    # MANAGED_UPLOADS_DIR from the pure rag.constants module (never from
    # credential-validating `config`), via a fresh `from rag.constants
    # import ...` inside main() itself — patching rag_constants here is
    # what main() actually observes.
    import config as app_config
    import rag.constants as rag_constants
    monkeypatch.setattr(rag_constants, "DOCUMENTS_DIR", documents_dir)
    monkeypatch.setattr(rag_constants, "MANAGED_UPLOADS_DIR", uploads_dir)

    def fail_if_vector_index_constructed(*args, **kwargs):
        raise AssertionError("dry run must never construct a VectorIndex (no Qdrant mutation, no embeddings)")

    monkeypatch.setattr(VectorIndex, "__init__", fail_if_vector_index_constructed)

    exit_code = rebuild.main([])  # no --apply
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "Dry run only" in out
    assert f"Built-in reference documents found: {len(app_config.BUILTIN_REFERENCE_FILES)}" in out
    assert "Managed uploads with valid sidecars found: 2" in out


def test_cli_dry_run_uses_exactly_the_builtin_manifest(manifest_source_tree, monkeypatch, tmp_path, capsys):
    """Stage 2B-C Blocker 5: main()'s dry run must enumerate EXACTLY the
    four config.BUILTIN_REFERENCE_FILES — a stray legacy .txt or an
    arbitrary extra .md dropped alongside them must not change the count."""
    documents_dir, uploads_dir = manifest_source_tree
    (documents_dir / "old_legacy_notes.txt").write_text("Stale legacy content.", encoding="utf-8")
    (documents_dir / "unrelated_extra.md").write_text("Some extra markdown.", encoding="utf-8")

    import config as app_config
    import rag.constants as rag_constants
    monkeypatch.setattr(rag_constants, "DOCUMENTS_DIR", documents_dir)
    monkeypatch.setattr(rag_constants, "MANAGED_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(VectorIndex, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no VectorIndex in dry run")))

    exit_code = rebuild.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"Built-in reference documents found: {len(app_config.BUILTIN_REFERENCE_FILES)}" in out


def test_cli_dry_run_fails_clearly_when_a_manifest_file_is_missing(manifest_source_tree, monkeypatch, tmp_path):
    documents_dir, uploads_dir = manifest_source_tree
    import config as app_config
    import rag.constants as rag_constants
    (documents_dir / app_config.BUILTIN_REFERENCE_FILES[0]).unlink()

    monkeypatch.setattr(rag_constants, "DOCUMENTS_DIR", documents_dir)
    monkeypatch.setattr(rag_constants, "MANAGED_UPLOADS_DIR", uploads_dir)

    from rag.loader import MissingReferenceDocumentError
    with pytest.raises(MissingReferenceDocumentError):
        rebuild.main([])


# ---------------------------------------------------------------------------
# 4/5: apply rebuilds reference documents AND sidecar-managed uploads
# ---------------------------------------------------------------------------

def test_apply_plan_indexes_reference_and_upload_documents(source_tree, tmp_path):
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="apply_test")
    try:
        report = rebuild.apply_plan(plan, vi)
        assert report.documents_reconciled == 4
        assert report.documents_reindexed == 4  # everything is new on a fresh collection
        assert report.chunks_reindexed == 4  # all short single-chunk documents
        assert report.documents_removed == 0
        assert vi.get_stats()["total_documents"] == 4
        assert fake.embed_documents_call_count == 4  # one batch per document

        # Reference document is retrievable and correctly attributed.
        results = vi.similarity_search("First reference guide content.", k=1)
        assert results[0].metadata["source"] == "guide-one.md"

        # Both duplicate-display-name uploads are present as separate documents.
        upload_a_id = upload_document_id("a" * 32)
        upload_b_id = upload_document_id("b" * 32)
        assert vi._existing_point_ids(upload_a_id)
        assert vi._existing_point_ids(upload_b_id)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# 6: malformed sidecar fails safely (skipped, not a crash)
# ---------------------------------------------------------------------------

def test_malformed_sidecar_is_skipped_safely(source_tree):
    documents_dir, uploads_dir = source_tree
    # A third upload whose sidecar is corrupt JSON.
    broken_physical = uploads_dir / ("c" * 32 + ".txt")
    broken_physical.write_bytes(b"orphaned upload content")
    sidecar_path_for(broken_physical).write_text("{not valid json", encoding="utf-8")

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(plan.upload_documents) == 2  # the two valid ones only
    assert "invalid_sidecar" in plan.skipped_upload_reasons


def test_upload_with_no_sidecar_at_all_is_skipped_safely(source_tree):
    documents_dir, uploads_dir = source_tree
    orphan_physical = uploads_dir / ("d" * 32 + ".txt")
    orphan_physical.write_bytes(b"no sidecar for this one")

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(plan.upload_documents) == 2
    assert "missing_sidecar" in plan.skipped_upload_reasons


def test_sidecar_identity_mismatch_is_skipped_safely(source_tree):
    documents_dir, uploads_dir = source_tree
    mismatched_physical = uploads_dir / ("e" * 32 + ".txt")
    mismatched_physical.write_bytes(b"mismatched content")
    # Sidecar is internally well-formed (its own document_id/stored_name
    # correspond to EACH OTHER, "f" * 32) but claims a DIFFERENT storage
    # UUID than the physical file it's actually paired with ("e" * 32).
    write_sidecar_atomic(
        sidecar_path_for(mismatched_physical),
        build_sidecar(upload_document_id("f" * 32), "spoofed.txt", "f" * 32 + ".txt", sha256_hex(b"mismatched content")),
    )

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(plan.upload_documents) == 2
    assert "sidecar_identity_mismatch" in plan.skipped_upload_reasons


def test_rebuild_plan_skips_upload_whose_physical_entry_is_a_symlink_escaping_uploads_root(source_tree):
    """Stage 2B-C Section H: Codex proved a symlink placed inside the
    managed uploads directory could be transparently followed by rebuild's
    lower-level planning even though there is no direct exfiltration
    through the public upload path. build_plan()/_plan_upload_documents()
    must reject it via path-containment, not read through it."""
    documents_dir, uploads_dir = source_tree
    stem = "9" * 32
    link_path = uploads_dir / f"{stem}.txt"
    outside_secret = uploads_dir.parent / "outside_secret.txt"
    outside_secret.write_bytes(b"top secret content outside uploads root")

    try:
        os.symlink(outside_secret, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    write_sidecar_atomic(
        sidecar_path_for(link_path),
        build_sidecar(upload_document_id(stem), "escape.txt", link_path.name, sha256_hex(b"top secret content outside uploads root")),
    )

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(plan.upload_documents) == 2  # the two legitimate ones only
    assert "path_containment_violation" in plan.skipped_upload_reasons


def test_rebuild_plan_skips_upload_whose_sidecar_is_a_symlink_escaping_uploads_root(source_tree):
    """Stage 2B-D Blocker 1: Codex proved `_plan_upload_documents()` called
    `load_sidecar()` BEFORE validating containment of the SIDECAR PATH
    ITSELF — a `uploads/<uuid>.meta.json` implemented as a symlink to
    valid JSON outside uploads_root was followed and accepted. The
    external JSON here is deliberately crafted to otherwise look like a
    perfectly valid, matching sidecar (correct document_id/stored_name/
    content_sha256 for the real physical file) — proving the rejection is
    about the SIDECAR PATH's own containment, not merely a downstream
    identity/hash mismatch that would incidentally catch a less carefully
    crafted exploit."""
    documents_dir, uploads_dir = source_tree
    stem = "8" * 32
    physical = uploads_dir / f"{stem}.txt"
    physical_content = b"legitimate physical content"
    physical.write_bytes(physical_content)

    outside_json = uploads_dir.parent / "outside_sidecar.meta.json"
    outside_json.write_text(json.dumps(build_sidecar(
        upload_document_id(stem), "escape.txt", physical.name, sha256_hex(physical_content),
    )), encoding="utf-8")

    sidecar_link = sidecar_path_for(physical)
    try:
        os.symlink(outside_json, sidecar_link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(plan.upload_documents) == 2  # the two legitimate ones only
    assert "path_containment_violation" in plan.skipped_upload_reasons
    # No external content was ever incorporated into the plan.
    assert "escape.txt" not in {d.display_source for d in plan.upload_documents}
    assert upload_document_id(stem) not in {d.document_id for d in plan.upload_documents}


def test_sidecar_content_hash_mismatch_is_skipped_safely(source_tree):
    documents_dir, uploads_dir = source_tree
    tampered_physical = uploads_dir / ("1" * 32 + ".txt")
    tampered_physical.write_bytes(b"original content")
    write_sidecar_atomic(
        sidecar_path_for(tampered_physical),
        build_sidecar(upload_document_id("1" * 32), "tampered.txt", tampered_physical.name, sha256_hex(b"original content")),
    )
    # File contents changed after the sidecar was written (stale/tampered).
    tampered_physical.write_bytes(b"different content now")

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(plan.upload_documents) == 2
    assert "sidecar_content_hash_mismatch" in plan.skipped_upload_reasons


# ---------------------------------------------------------------------------
# 7/10: rebuild is idempotent / rerunning yields deterministic IDs, no
# duplication
# ---------------------------------------------------------------------------

def test_rebuild_is_idempotent_and_deterministic(source_tree, tmp_path):
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="idempotent_apply_test")
    try:
        report1 = rebuild.apply_plan(plan, vi)
        ids_after_first = {
            (d.document_id, point_id(d.document_id, 0))
            for d in plan.all_documents
        }
        count_after_first = vi.get_stats()["total_documents"]

        # Re-run the exact same plan a second time — Stage 2B-C Blocker 3:
        # NON-DESTRUCTIVE reconciliation, never a clear-then-rebuild.
        # Every document reconciles as "unchanged" this time (zero
        # re-embedding), which is what proves idempotency/convergence
        # rather than merely "re-embeds everything identically twice".
        report2 = rebuild.apply_plan(plan, vi)
        count_after_second = vi.get_stats()["total_documents"]

        assert report1.documents_reconciled == report2.documents_reconciled == 4
        assert report1.documents_reindexed == 4       # first run: all new
        assert report2.documents_reindexed == 0        # second run: nothing changed
        assert report2.documents_removed == 0
        assert count_after_first == count_after_second == 4  # no duplication

        for document_id, expected_point_id in ids_after_first:
            assert expected_point_id in vi._existing_point_ids(document_id)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# 8: duplicate display filenames remain distinct after rebuild
# ---------------------------------------------------------------------------

def test_rebuild_keeps_duplicate_display_names_as_distinct_documents(source_tree, tmp_path):
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    upload_docs = plan.upload_documents
    assert len(upload_docs) == 2
    assert {d.display_source for d in upload_docs} == {"shared_name.txt"}
    assert upload_docs[0].document_id != upload_docs[1].document_id


# ---------------------------------------------------------------------------
# 9: no legacy Chroma access — the module has no chromadb dependency at all
# ---------------------------------------------------------------------------

def test_rebuild_module_never_imports_or_references_chromadb():
    source = inspect.getsource(rebuild)
    assert "import chromadb" not in source
    assert "from chromadb" not in source
    assert not hasattr(rebuild, "chromadb")


# ---------------------------------------------------------------------------
# Stage 2B-C Blocker 3: apply_plan() must be a NON-DESTRUCTIVE
# reconciliation — never clear-then-rebuild. A failure partway through
# must leave every already-valid document (both from a previous run AND
# earlier in the SAME run) completely intact, and orphan pruning must run
# ONLY after every desired document reconciled successfully.
# ---------------------------------------------------------------------------

def test_apply_plan_never_calls_clear_index(source_tree, tmp_path, monkeypatch):
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="no_clear_test")
    try:
        # Pre-existing orphan data (not part of the plan) — if clear_index()
        # were ever called, this would be the first thing destroyed.
        from langchain_core.documents import Document
        vi.add_documents([Document(page_content="orphan content", metadata={"document_id": "docOrphanPre", "chunk_index": 0, "source": "x.md"})])

        def fail_if_cleared():
            raise AssertionError("apply_plan() must never call clear_index() — Stage 2B-C Blocker 3")

        monkeypatch.setattr(vi, "clear_index", fail_if_cleared)

        report = rebuild.apply_plan(plan, vi)
        assert report.documents_reconciled == 4
    finally:
        vi.close()


def test_apply_plan_first_document_embedding_failure_preserves_prior_valid_index(source_tree, tmp_path, monkeypatch):
    """Stage 2B-C Blocker 3, reproduction 1: if the FIRST document's
    embedding fails, previously-valid, unrelated index data must remain —
    never wiped by a clear-first step that no longer exists."""
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="first_failure_test")
    try:
        from langchain_core.documents import Document
        vi.add_documents([Document(page_content="pre-existing valid content", metadata={"document_id": "docPreexisting", "chunk_index": 0, "source": "old.md"})])
        assert vi.get_stats()["total_documents"] == 1

        def failing_embed_documents(texts):
            raise RuntimeError("simulated provider failure on first document")

        monkeypatch.setattr(fake, "embed_documents", failing_embed_documents)

        with pytest.raises(RuntimeError):
            rebuild.apply_plan(plan, vi)

        # Old, unrelated valid data survives untouched — nothing from the
        # plan was ever indexed, and the pre-existing document was never
        # removed (no clear-first, and orphan pruning never ran because
        # the plan failed).
        assert vi.get_stats()["total_documents"] == 1
        assert vi._existing_point_ids("docPreexisting")
    finally:
        vi.close()


def test_apply_plan_later_document_failure_preserves_earlier_reconciled_and_unrelated_documents(source_tree, tmp_path, monkeypatch):
    """Stage 2B-C Blocker 3, reproduction 2: a failure on a LATER document
    in the plan must not destroy documents already reconciled earlier in
    the SAME apply_plan() call, nor any unrelated pre-existing document —
    and orphan pruning must never run when the plan didn't fully succeed."""
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(plan.all_documents) == 4

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="later_failure_test")
    try:
        from langchain_core.documents import Document
        vi.add_documents([Document(page_content="pre-existing valid content", metadata={"document_id": "docPreexistingOrphan", "chunk_index": 0, "source": "old.md"})])

        real_embed_documents = fake.embed_documents
        call_count = {"n": 0}

        def failing_on_second_call(texts):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated provider failure on a later document")
            return real_embed_documents(texts)

        monkeypatch.setattr(fake, "embed_documents", failing_on_second_call)

        with pytest.raises(RuntimeError):
            rebuild.apply_plan(plan, vi)

        # The first document (reconciled before the failure) is indexed;
        # the pre-existing unrelated document is untouched; orphan pruning
        # (which would have removed docPreexistingOrphan) never ran.
        first_doc = plan.all_documents[0]
        assert vi._existing_point_ids(first_doc.document_id)
        assert vi._existing_point_ids("docPreexistingOrphan")
    finally:
        vi.close()


def test_apply_plan_upsert_failure_does_not_destroy_the_whole_index(source_tree, tmp_path, monkeypatch):
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="upsert_failure_test")
    try:
        from langchain_core.documents import Document
        vi.add_documents([Document(page_content="pre-existing valid content", metadata={"document_id": "docPreexisting", "chunk_index": 0, "source": "old.md"})])

        def failing_upsert(*args, **kwargs):
            raise RuntimeError("simulated Qdrant upsert failure")

        monkeypatch.setattr(vi.client, "upsert", failing_upsert)

        with pytest.raises(RuntimeError):
            rebuild.apply_plan(plan, vi)

        assert vi.get_stats()["total_documents"] == 1
        assert vi._existing_point_ids("docPreexisting")
    finally:
        vi.close()


def test_apply_plan_removes_orphans_only_after_full_success(source_tree, tmp_path):
    """Stage 2B-C Blocker 3, requirement 5/6: orphan logical documents
    (indexed but no longer present in source truth) are removed ONLY once
    every desired document has reconciled successfully, and a successful
    rebuild leaves Qdrant with EXACTLY the desired document set."""
    documents_dir, uploads_dir = source_tree
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    desired_ids = {d.document_id for d in plan.all_documents}

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="orphan_removal_test")
    try:
        from langchain_core.documents import Document
        vi.add_documents([Document(page_content="orphan content", metadata={"document_id": "docTrulyOrphaned", "chunk_index": 0, "source": "gone.md"})])

        report = rebuild.apply_plan(plan, vi)

        assert report.documents_removed == 1
        assert not vi._existing_point_ids("docTrulyOrphaned")
        assert vi.list_document_ids() == desired_ids
    finally:
        vi.close()


def test_apply_plan_deterministic_rerun_converges_after_stale_delete_failure(source_tree, tmp_path, monkeypatch):
    """apply_plan()-level version of the Blocker 2 stale-delete-failure
    convergence proof: a failure deleting stale points during
    reconciliation surfaces, but a subsequent apply_plan() call against
    the same plan converges (prunes the leftover stale points) with no
    re-embedding for the already-current document."""
    documents_dir, uploads_dir = source_tree
    (documents_dir / "guide-two.md").write_text(
        "Second reference guide content, distinct text. " * 40, encoding="utf-8"
    )
    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="apply_stale_retry_test")
    try:
        report1 = rebuild.apply_plan(plan, vi)
        assert report1.documents_reindexed == 4
        guide_two_id = next(d.document_id for d in plan.reference_documents if d.display_source == "guide-two.md")
        chunks_before = vi._existing_point_ids(guide_two_id)
        assert len(chunks_before) > 1

        # Shrink guide-two.md so its next reconciliation deletes stale
        # trailing points — but make that delete fail.
        (documents_dir / "guide-two.md").write_text("Much shorter now.", encoding="utf-8")
        plan2 = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

        real_delete = vi.client.delete
        monkeypatch.setattr(vi.client, "delete", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated stale-delete failure")))
        with pytest.raises(RuntimeError):
            rebuild.apply_plan(plan2, vi)
        monkeypatch.setattr(vi.client, "delete", real_delete)

        embed_calls_before_retry = fake.embed_documents_call_count
        report2 = rebuild.apply_plan(plan2, vi)
        assert report2.documents_reindexed == 0  # zero re-embedding on the retry
        assert fake.embed_documents_call_count == embed_calls_before_retry
        assert vi._existing_point_ids(guide_two_id) == {point_id(guide_two_id, 0)}
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Stage 2B-C Blocker 4: the documented/supported operator invocation is
# `python -m scripts.rebuild_qdrant`, run as a REAL subprocess (not an
# in-process main() call) from the repository root, against an ISOLATED
# temp copy of the source tree — proves the module form actually starts
# successfully and touches nothing real.
# ---------------------------------------------------------------------------

def test_module_invocation_dry_run_starts_successfully_as_a_real_subprocess(tmp_path):
    import shutil
    import subprocess
    import sys as _sys

    real_repo_root = Path(__file__).resolve().parents[1]
    real_qdrant_existed_before = (real_repo_root / "data" / "qdrant").exists()
    real_bot_log = real_repo_root / "bot.log"
    real_bot_log_stat_before = (real_bot_log.stat().st_mtime, real_bot_log.stat().st_size) if real_bot_log.exists() else None

    isolated_root = tmp_path / "isolated_repo"
    isolated_root.mkdir()

    # Copy ONLY the modules dry-run mode actually needs to import — never
    # the real repo's data/, .env, or bot.log. This makes config.py's own
    # `BASE_DIR = Path(__file__).parent` resolve to THIS isolated copy,
    # so LOG_FILE/DOCUMENTS_DIR/MANAGED_UPLOADS_DIR are fully isolated too
    # — genuine process-level isolation, not merely trusting dry-run's
    # read-only nature.
    for name in ("config.py",):
        shutil.copy2(real_repo_root / name, isolated_root / name)
    for pkg in ("rag", "scripts", "utils"):
        shutil.copytree(real_repo_root / pkg, isolated_root / pkg)

    documents_dir = isolated_root / "data" / "documents"
    documents_dir.mkdir(parents=True)
    import config as app_config
    for filename in app_config.BUILTIN_REFERENCE_FILES:
        (documents_dir / filename).write_text(f"Isolated reference content for {filename}.", encoding="utf-8")

    env = dict(os.environ)
    env.update({
        "TELEGRAM_BOT_TOKEN": "123456789:ISOLATED-TEST-TOKEN",
        "OPENAI_API_KEY": "sk-isolated-test-dummy-key",
        "ANTHROPIC_API_KEY": "sk-ant-isolated-test-dummy-key",
        "LLM_PROVIDER": "openai",
    })
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy", "OPENAI_PROXY"):
        env.pop(proxy_var, None)

    result = subprocess.run(
        [_sys.executable, "-m", "scripts.rebuild_qdrant"],
        cwd=str(isolated_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ModuleNotFoundError" not in result.stderr
    assert f"Built-in reference documents found: {len(app_config.BUILTIN_REFERENCE_FILES)}" in result.stdout
    assert "Dry run only" in result.stdout

    # No real Qdrant mutation, no mutation of the isolated tree's own
    # data/qdrant either (dry run makes zero Qdrant calls at all).
    assert not (isolated_root / "data" / "qdrant").exists()
    # The real repository's own data/qdrant and bot.log are untouched —
    # this subprocess's BASE_DIR/CWD was never the real repo root, so it
    # could only ever have written to the isolated copy.
    assert (real_repo_root / "data" / "qdrant").exists() == real_qdrant_existed_before
    real_bot_log_stat_after = (real_bot_log.stat().st_mtime, real_bot_log.stat().st_size) if real_bot_log.exists() else None
    assert real_bot_log_stat_after == real_bot_log_stat_before
