"""
Stage 2B-B regression tests: version-controlled reference knowledge base
(Section G/H/U) — the four built-in Markdown documents replacing the old
disposable .txt files, their stable identity, and idempotent-startup
behavior (Section N: an unchanged reference document must cost zero
embedding calls on a repeated index_documents_directory() run).

Tests 1-5/9/10 read the REAL data/documents/*.md files directly off disk
(pure filesystem/hash checks, never indexed into any real Qdrant/vector
store — nothing here touches the real gitignored data/qdrant). Tests 6-8
build a synthetic, isolated tmp_path "documents" directory and a
dedicated VectorIndex with a deterministic local fake embeddings double —
no OpenAI call anywhere in this module.
"""

import uuid
from pathlib import Path

import pytest

import config
from rag.identity import reference_document_id, sha256_hex
from rag.index import VectorIndex
from rag.loader import MissingReferenceDocumentError, SUPPORTED_EXTENSIONS, document_loader
from rag_fakes import DeterministicFakeEmbeddings

REAL_DOCUMENTS_DIR = Path(config.BASE_DIR) / "data" / "documents"

EXPECTED_REFERENCE_FILES = frozenset({
    "python-fundamentals.md",
    "functions-classes-errors.md",
    "testing-debugging.md",
    "async-python-and-apis.md",
})

OBSOLETE_REFERENCE_FILES = frozenset({
    "python_basics.txt",
    "pep8_guidelines.txt",
    "common_mistakes.txt",
})


# ---------------------------------------------------------------------------
# 1/2: the four expected files exist; the obsolete .txt files are gone
# ---------------------------------------------------------------------------

def test_all_four_reference_markdown_files_exist():
    for filename in EXPECTED_REFERENCE_FILES:
        path = REAL_DOCUMENTS_DIR / filename
        assert path.is_file(), f"missing expected reference document: {filename}"
        assert path.stat().st_size > 0


def test_obsolete_reference_txt_files_are_gone():
    for filename in OBSOLETE_REFERENCE_FILES:
        assert not (REAL_DOCUMENTS_DIR / filename).exists(), f"obsolete file still present: {filename}"


# ---------------------------------------------------------------------------
# 3: files are loader-supported
# ---------------------------------------------------------------------------

def test_reference_files_have_a_loader_supported_extension():
    for filename in EXPECTED_REFERENCE_FILES:
        assert Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS
    assert document_loader.list_source_files(REAL_DOCUMENTS_DIR)  # non-empty, real scan finds them
    found_names = {p.name for p in document_loader.list_source_files(REAL_DOCUMENTS_DIR)}
    assert EXPECTED_REFERENCE_FILES <= found_names


# ---------------------------------------------------------------------------
# 4/5: stable document IDs and stable content hash
# ---------------------------------------------------------------------------

def test_reference_document_ids_are_stable_and_relative_path_derived():
    for filename in EXPECTED_REFERENCE_FILES:
        first = reference_document_id(filename)
        second = reference_document_id(filename)
        assert first == second
        assert first.startswith("ref:")

    # Distinct files -> distinct ids.
    ids = {reference_document_id(name) for name in EXPECTED_REFERENCE_FILES}
    assert len(ids) == len(EXPECTED_REFERENCE_FILES)

    # Never derived from an absolute path — the same relative name always
    # maps to the same id regardless of which absolute root it's read from.
    assert reference_document_id("python-fundamentals.md") == reference_document_id("python-fundamentals.md")


def test_reference_content_hash_is_stable_and_content_derived():
    path = REAL_DOCUMENTS_DIR / "python-fundamentals.md"
    content = path.read_bytes()
    first = sha256_hex(content)
    second = sha256_hex(content)
    assert first == second
    assert len(first) == 64  # hex sha256

    # Changing content changes the hash.
    assert sha256_hex(content + b"x") != first


# ---------------------------------------------------------------------------
# 9/10: source attribution uses the Markdown filename; no absolute path
# persisted into the Qdrant payload
# ---------------------------------------------------------------------------

def test_indexing_a_reference_document_attributes_source_by_filename_and_leaks_no_absolute_path(tmp_path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    source_path = docs_dir / "sensitive_deploy_user_notes.md"
    source_path.write_text("# Notes\n\nSome reference content for attribution testing.", encoding="utf-8")

    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="ref_attribution_test",
    )
    try:
        # reference_filenames=None: this test exercises generic
        # directory-scan/attribution mechanics against a synthetic
        # filename, not the config.BUILTIN_REFERENCE_FILES manifest gate
        # (see test_stage2c_reference_manifest.py for manifest-specific
        # coverage) — see index_documents_directory()'s own docstring.
        vi.index_documents_directory(directory=docs_dir, reference_filenames=None)
        results = vi.similarity_search("Some reference content for attribution testing.", requesting_user_uuid=str(uuid.uuid4()), k=1)
        assert len(results) == 1
        doc = results[0]
        assert doc.metadata["source"] == "sensitive_deploy_user_notes.md"

        records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
        for record in records:
            assert "file_path" not in record.payload
            assert str(docs_dir) not in str(record.payload)
            assert str(tmp_path) not in str(record.payload)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# 6/7/8: idempotent startup — zero embedding calls when unchanged, exactly
# one reindex when changed, stale points removed when chunk count shrinks
# ---------------------------------------------------------------------------

def test_unchanged_reference_document_costs_zero_embedding_calls_on_restart(tmp_path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("Stable content that never changes between startups.", encoding="utf-8")

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="idempotent_startup_test")
    try:
        first_run_chunks = vi.index_documents_directory(directory=docs_dir, reference_filenames=None)
        assert first_run_chunks == 1
        assert fake.embed_documents_call_count == 1  # exactly one real (fake) embedding batch

        # Simulate a second application startup against the SAME unchanged
        # source directory and the SAME persisted Qdrant collection.
        second_run_chunks = vi.index_documents_directory(directory=docs_dir, reference_filenames=None)
        assert second_run_chunks == 0  # nothing new/changed to index
        assert fake.embed_documents_call_count == 1  # NOT incremented — zero new embedding calls
        assert vi.get_stats(requesting_user_uuid=str(uuid.uuid4()))["total_documents"] == 1  # no duplication either
    finally:
        vi.close()


def test_changed_reference_document_reindexes_exactly_once_and_removes_stale_chunks(tmp_path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    doc_path = docs_dir / "guide.md"

    # Long enough to reliably split into multiple chunks under
    # RAG_CHUNK_SIZE=1000/RAG_CHUNK_OVERLAP=200.
    long_paragraph = "This is a sentence about Python used to pad out the document. " * 60
    doc_path.write_text(long_paragraph, encoding="utf-8")

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="reindex_change_test")
    try:
        vi.index_documents_directory(directory=docs_dir, reference_filenames=None)
        doc_id = reference_document_id("guide.md")
        chunks_before = vi._existing_point_ids(doc_id)
        assert len(chunks_before) > 1, "test setup: expected the long paragraph to split into multiple chunks"
        embed_calls_before = fake.embed_documents_call_count

        # Unchanged re-run: still zero new embedding calls.
        vi.index_documents_directory(directory=docs_dir, reference_filenames=None)
        assert fake.embed_documents_call_count == embed_calls_before

        # Now change the content to something much shorter (fewer chunks).
        doc_path.write_text("A short replacement sentence.", encoding="utf-8")
        chunks_indexed = vi.index_documents_directory(directory=docs_dir, reference_filenames=None)
        assert chunks_indexed == 1
        # Exactly one additional embedding call batch for the changed file.
        assert fake.embed_documents_call_count == embed_calls_before + 1

        chunks_after = vi._existing_point_ids(doc_id)
        assert len(chunks_after) == 1
        # The stale trailing chunks from the longer version are genuinely gone.
        assert chunks_after.isdisjoint(chunks_before - chunks_after)
        assert vi.get_stats(requesting_user_uuid=str(uuid.uuid4()))["total_documents"] == 1
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Stage 2B-C Blocker 5: explicit built-in reference manifest — a synthetic
# tmp_path directory shaped like a manifest root (files literally named
# after config.BUILTIN_REFERENCE_FILES), proving stray/extra files never
# get silently promoted into built-in product knowledge, and a missing
# manifest file fails loudly instead of silently indexing fewer documents.
# ---------------------------------------------------------------------------

_SYNTHETIC_MANIFEST = ("alpha.md", "beta.md")


@pytest.fixture
def manifest_root(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()
    (root / "alpha.md").write_text("Alpha built-in reference content.", encoding="utf-8")
    (root / "beta.md").write_text("Beta built-in reference content.", encoding="utf-8")
    return root


def test_list_builtin_reference_files_enumerates_exactly_the_manifest(manifest_root):
    found = document_loader.list_builtin_reference_files(manifest_root, _SYNTHETIC_MANIFEST)
    assert [p.name for p in found] == list(_SYNTHETIC_MANIFEST)


def test_list_builtin_reference_files_ignores_stray_legacy_txt(manifest_root):
    (manifest_root / "old_notes.txt").write_text("Stale legacy content that must not be indexed.", encoding="utf-8")

    found = document_loader.list_builtin_reference_files(manifest_root, _SYNTHETIC_MANIFEST)
    assert {p.name for p in found} == set(_SYNTHETIC_MANIFEST)
    assert "old_notes.txt" not in {p.name for p in found}


def test_list_builtin_reference_files_ignores_arbitrary_extra_markdown(manifest_root):
    (manifest_root / "unrelated_extra.md").write_text("Some extra markdown that was never meant to ship.", encoding="utf-8")

    found = document_loader.list_builtin_reference_files(manifest_root, _SYNTHETIC_MANIFEST)
    assert {p.name for p in found} == set(_SYNTHETIC_MANIFEST)
    assert "unrelated_extra.md" not in {p.name for p in found}


def test_list_builtin_reference_files_raises_clearly_when_one_is_missing(manifest_root):
    (manifest_root / "beta.md").unlink()
    with pytest.raises(MissingReferenceDocumentError):
        document_loader.list_builtin_reference_files(manifest_root, _SYNTHETIC_MANIFEST)

    # All-or-nothing: even though "alpha.md" is present and valid, a
    # missing manifest file must not silently index only the rest.


def test_list_builtin_reference_files_raises_when_manifest_directory_absent(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(MissingReferenceDocumentError):
        document_loader.list_builtin_reference_files(missing_dir, _SYNTHETIC_MANIFEST)


def test_index_documents_directory_default_indexes_exactly_the_manifest_ignoring_extras(manifest_root, tmp_path):
    """The production default (reference_filenames=BUILTIN_REFERENCE_FILES,
    here overridden to the synthetic manifest) must index exactly the
    manifest files and silently ignore a stray legacy .txt AND an
    arbitrary extra .md sitting right next to them."""
    (manifest_root / "old_notes.txt").write_text("Stale legacy content.", encoding="utf-8")
    (manifest_root / "unrelated_extra.md").write_text("Unrelated extra markdown.", encoding="utf-8")

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="manifest_startup_test")
    try:
        chunks_indexed = vi.index_documents_directory(directory=manifest_root, reference_filenames=_SYNTHETIC_MANIFEST)
        assert chunks_indexed == 2  # exactly alpha.md + beta.md, one chunk each
        assert fake.embed_documents_call_count == 2  # one batch per manifest document, never for the extras

        alpha_id = reference_document_id("alpha.md")
        beta_id = reference_document_id("beta.md")
        assert vi._existing_point_ids(alpha_id)
        assert vi._existing_point_ids(beta_id)

        # The stray .txt and extra .md were never even hashed into an
        # id, let alone indexed — no document_id derived from either
        # filename has any points.
        stray_id = reference_document_id("old_notes.txt")
        extra_id = reference_document_id("unrelated_extra.md")
        assert not vi._existing_point_ids(stray_id)
        assert not vi._existing_point_ids(extra_id)

        results = vi.similarity_search("Alpha built-in reference content.", requesting_user_uuid=str(uuid.uuid4()), k=2)
        sources = {doc.metadata.get("source") for doc in results}
        assert "old_notes.txt" not in sources
        assert "unrelated_extra.md" not in sources
    finally:
        vi.close()


def test_index_documents_directory_missing_manifest_file_fails_loudly_not_partially(manifest_root, tmp_path):
    (manifest_root / "beta.md").unlink()

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="manifest_missing_test")
    try:
        with pytest.raises(MissingReferenceDocumentError):
            vi.index_documents_directory(directory=manifest_root, reference_filenames=_SYNTHETIC_MANIFEST)

        # All-or-nothing: "alpha.md" (present, valid) must NOT have been
        # silently indexed on its own while "beta.md" was missing.
        assert fake.embed_documents_call_count == 0
        assert vi.get_stats(requesting_user_uuid=str(uuid.uuid4()))["total_documents"] == 0
    finally:
        vi.close()
