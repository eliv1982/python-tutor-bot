"""
Stage 2B-E regression tests: Blocker 1 (check-then-open TOCTOU) from the
latest independent Codex REJECTED verdict.

Codex reproduced a race on BOTH Windows and Linux: a sidecar/source path
passed containment validation (rag.sidecar's resolve_sidecar_path() /
resolve_managed_upload_path()) and was THEN reopened a second time by a
separate call (the old load_sidecar()/Path.read_bytes()) — a window in
which the filesystem object the pathname refers to could be replaced
before that second open. This module proves, at three levels:

  1. rag.safe_files.read_regular_file_secure() itself detects a swap
     occurring in the narrowest possible window (immediately before its
     own single open call), via an injectable test-only hook — genuine
     OS-thread race timing would make this non-deterministic, so the hook
     lets the test land the swap exactly where the vulnerable window used
     to be;
  2. scripts.rebuild_qdrant.build_plan() rejects a sidecar/source swapped
     at the same boundary the OLD code's reopen sat at, and never lets the
     external content enter the plan;
  3. scripts.rebuild_qdrant.apply_plan() never embeds content swapped in
     AFTER a plan was already built but BEFORE apply ran (Section F:
     plan-to-apply content integrity) — proven by actually indexing and
     retrieving the result, not merely inspecting the plan object.

Stable/pre-existing symlink rejection (a symlink that is simply already in
place, no race involved), internal-symlink policy, and sibling-prefix
containment (`uploads` vs `uploads-other`) are unaffected by this pass and
remain covered by tests/test_stage2b_sidecar.py and
tests/test_stage2b_rebuild.py's existing tests, run unmodified against the
code in this pass — not duplicated here except for two direct
rag.safe_files-level checks (sibling-prefix, symlink-at-call-time) that
exercise the new primitive specifically.

Entirely temporary fixtures — no real documents/uploads/Qdrant anywhere.
"""

import io
import json
import os
import zipfile
from pathlib import Path

import pytest

import scripts.rebuild_qdrant as rebuild
from rag.identity import sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.safe_files import SecureReadError, read_regular_file_secure
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings


# ---------------------------------------------------------------------------
# Level 1: rag.safe_files.read_regular_file_secure() — the primitive itself.
# ---------------------------------------------------------------------------

def test_read_regular_file_secure_accepts_a_normal_file_with_no_race(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "a.txt"
    target.write_bytes(b"hello world")
    assert read_regular_file_secure(target, root=root) == b"hello world"


def test_read_regular_file_secure_rejects_sibling_prefix_directory(tmp_path):
    """`root=/uploads` must never accept a candidate under a sibling
    directory whose name merely starts with the same string
    (`/uploads-other`) — a naive string-prefix containment check would
    wrongly accept this."""
    root = tmp_path / "uploads"
    root.mkdir()
    sibling = tmp_path / "uploads-other"
    sibling.mkdir()
    target = sibling / "a.txt"
    target.write_bytes(b"hello")
    with pytest.raises(SecureReadError):
        read_regular_file_secure(target, root=root)


def test_read_regular_file_secure_rejects_a_symlink_candidate(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    link = root / "a.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")
    with pytest.raises(SecureReadError):
        read_regular_file_secure(link, root=root)


def test_read_regular_file_secure_rejects_missing_candidate(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    with pytest.raises(SecureReadError):
        read_regular_file_secure(root / "does_not_exist.txt", root=root)


def test_read_regular_file_secure_rejects_a_directory_candidate(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    directory = root / "a_directory"
    directory.mkdir()
    with pytest.raises(SecureReadError):
        read_regular_file_secure(directory, root=root)


def test_read_regular_file_secure_detects_swap_to_symlink_in_narrow_window(tmp_path):
    """The exact Codex TOCTOU shape: the candidate is a valid regular file
    at pre-open validation time, then becomes a symlink to external
    content in the window immediately before open(). Must be rejected —
    the external content must never be returned."""
    root = tmp_path / "uploads"
    root.mkdir()
    original = root / "a.txt"
    original.write_bytes(b"original content")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"attacker content")

    swapped = {"done": False}

    def swap(path):
        os.remove(path)
        try:
            os.symlink(outside, path)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        swapped["done"] = True

    with pytest.raises(SecureReadError):
        read_regular_file_secure(original, root=root, _test_pre_open_hook=swap)
    assert swapped["done"]


def test_read_regular_file_secure_detects_swap_to_different_regular_file_in_narrow_window(tmp_path):
    """Proves the identity check itself (device+inode via
    os.path.samestat()), independent of symlink rejection: the swap
    target here is a plain REGULAR file (no symlink/no special privilege
    required — runs unconditionally on every platform including Windows
    without Developer Mode/admin). O_NOFOLLOW alone could never catch
    this kind of swap (it isn't a symlink) — only the post-open fstat
    identity comparison against the pre-open lstat does."""
    root = tmp_path / "uploads"
    root.mkdir()
    original = root / "a.txt"
    original.write_bytes(b"original content")

    def swap(path):
        os.remove(path)
        Path(path).write_bytes(b"a completely different regular file's content")

    with pytest.raises(SecureReadError):
        read_regular_file_secure(original, root=root, _test_pre_open_hook=swap)


# ---------------------------------------------------------------------------
# Level 2: scripts.rebuild_qdrant.build_plan() — sidecar/source swapped at
# the exact boundary the old vulnerable reopen sat at.
# ---------------------------------------------------------------------------

@pytest.fixture
def uploads_tree(tmp_path, monkeypatch):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    uploads_dir = documents_dir / "uploads"
    uploads_dir.mkdir()

    import rag.loader as rag_loader
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", uploads_dir)

    return documents_dir, uploads_dir


def test_build_plan_rejects_sidecar_swapped_to_external_symlink_at_secure_read_boundary(uploads_tree, monkeypatch):
    """Section H.1 (SIDECAR): valid regular sidecar passes initial
    observation, then is swapped to a symlink pointing at valid,
    well-formed, perfectly MATCHING external JSON (deliberately crafted
    to look legitimate — proving rejection is about object identity, not
    merely a downstream content mismatch a less careful exploit would
    incidentally trip) exactly at the boundary the previous vulnerable
    reopen sat at. The external content must never enter the plan."""
    documents_dir, uploads_dir = uploads_tree
    stem = "a" * 32
    physical = uploads_dir / f"{stem}.txt"
    physical_content = b"legitimate physical content"
    physical.write_bytes(physical_content)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(upload_document_id(stem), "legit.txt", physical.name, sha256_hex(physical_content)),
    )

    external_json = uploads_dir.parent / "external.meta.json"
    external_json.write_text(json.dumps(build_sidecar(
        upload_document_id(stem), "legit.txt", physical.name, sha256_hex(physical_content),
    )), encoding="utf-8")

    real_secure_read = rebuild.secure_read_sidecar_bytes

    def racing_secure_read(uploads_root, sidecar_path):
        try:
            os.remove(sidecar_path)
            os.symlink(external_json, sidecar_path)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        return real_secure_read(uploads_root, sidecar_path)

    monkeypatch.setattr(rebuild, "secure_read_sidecar_bytes", racing_secure_read)

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(plan.upload_documents) == 0
    assert "path_containment_violation" in plan.skipped_upload_reasons
    assert "legit.txt" not in {d.display_source for d in plan.upload_documents}


def test_build_plan_rejects_source_swapped_to_external_symlink_at_secure_read_boundary(uploads_tree, monkeypatch):
    """Section H.2 (SOURCE): valid regular managed source passes initial
    validation, then is swapped to a symlink pointing at an external
    supported document exactly at the secure-read boundary. The external
    bytes must never enter the plan or its hash."""
    documents_dir, uploads_dir = uploads_tree
    stem = "b" * 32
    physical = uploads_dir / f"{stem}.txt"
    physical_content = b"legitimate physical content"
    physical.write_bytes(physical_content)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(upload_document_id(stem), "legit.txt", physical.name, sha256_hex(physical_content)),
    )

    external_source = uploads_dir.parent / "external_source.txt"
    external_source.write_bytes(b"EXTERNAL ATTACKER CONTENT")

    real_read = rebuild.read_regular_file_secure

    def racing_read(path, *, root):
        if Path(path) == physical:
            try:
                os.remove(path)
                os.symlink(external_source, path)
            except (OSError, NotImplementedError):
                pytest.skip("symlink creation not permitted on this platform/user")
        return real_read(path, root=root)

    monkeypatch.setattr(rebuild, "read_regular_file_secure", racing_read)

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(plan.upload_documents) == 0
    assert "path_containment_violation" in plan.skipped_upload_reasons
    assert b"EXTERNAL ATTACKER CONTENT" not in b"".join(
        d.content_bytes or b"" for d in plan.upload_documents
    )


def test_build_plan_accepts_regular_upload_with_no_race(uploads_tree):
    """Section H.3 (REGULAR NO-RACE): the new secure-read layer must not
    reject a perfectly ordinary, unmolested managed upload."""
    documents_dir, uploads_dir = uploads_tree
    stem = "c" * 32
    physical = uploads_dir / f"{stem}.txt"
    content = b"perfectly ordinary content"
    physical.write_bytes(content)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(upload_document_id(stem), "ordinary.txt", physical.name, sha256_hex(content)),
    )

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(plan.upload_documents) == 1
    assert not plan.skipped_upload_reasons
    assert plan.upload_documents[0].content_bytes == content


# ---------------------------------------------------------------------------
# Level 3: apply_plan() — plan-to-apply content integrity (Section F /
# Section H.7 APPLY BOUNDARY). The source is swapped AFTER the plan was
# built but BEFORE apply runs; apply must not embed the replacement bytes
# under the plan's old identity/hash.
# ---------------------------------------------------------------------------

def test_apply_plan_does_not_embed_source_swapped_after_plan_built(uploads_tree, tmp_path):
    documents_dir, uploads_dir = uploads_tree
    stem = "d" * 32
    physical = uploads_dir / f"{stem}.txt"
    original_content = b"ORIGINAL PLAN-TIME CONTENT, never to be replaced by an attacker."
    physical.write_bytes(original_content)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(upload_document_id(stem), "doc.txt", physical.name, sha256_hex(original_content)),
    )

    plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)
    assert len(plan.upload_documents) == 1
    upload_doc = plan.upload_documents[0]
    assert upload_doc.content_bytes == original_content

    # Swap the on-disk source to a symlink pointing at different content
    # AFTER the plan already captured its bytes, BEFORE apply_plan() runs.
    external = uploads_dir.parent / "external_after_plan.txt"
    external.write_bytes(b"SWAPPED EXTERNAL CONTENT AFTER PLAN BUILD")
    physical.unlink()
    try:
        os.symlink(external, physical)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="apply_boundary_test")
    try:
        report = rebuild.apply_plan(plan, vi)
        assert report.documents_reconciled == 1
        assert report.documents_reindexed == 1

        results = vi.similarity_search("ORIGINAL PLAN-TIME CONTENT", k=1)
        assert results
        assert "ORIGINAL PLAN-TIME CONTENT" in results[0].page_content
        assert "SWAPPED" not in results[0].page_content
    finally:
        vi.close()


def test_reconcile_document_with_source_bytes_never_reopens_file_path(tmp_path):
    """Unit-level proof of the mechanism apply_plan() relies on:
    VectorIndex.reconcile_document(source_bytes=...) must not touch
    `file_path` on disk at all when bytes are already provided — even a
    file_path that doesn't exist on disk (or points at completely
    different content) must not matter, since it's never read."""
    from langchain_core.documents import Document  # noqa: F401  (import parity with other tests)

    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="never_reopen_test",
    )
    try:
        source_bytes = b"content that only ever exists in memory"
        content_sha256 = sha256_hex(source_bytes)
        nonexistent_path = tmp_path / "this_file_does_not_exist_on_disk.txt"
        assert not nonexistent_path.exists()

        status, chunk_count = vi.reconcile_document(
            "upload:" + "e" * 32,
            nonexistent_path,
            display_name="in_memory.txt",
            expected_content_sha256=content_sha256,
            source_bytes=source_bytes,
        )

        assert status == "reindexed"
        assert chunk_count >= 1
        assert not nonexistent_path.exists()  # never created/touched
        results = vi.similarity_search("content that only ever exists in memory", k=1)
        assert results
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# Level 4 (Stage 2B-F Blocker 1 — a later audit finding against the fix
# proven at Level 3 above): reconcile_document(source_bytes=...) used to
# hand those already-verified bytes to the loader via a private TEMPORARY
# SNAPSHOT FILE, which the format-specific parser (PyPDFLoader/TextLoader/
# Docx2txtLoader) then reopened BY PATHNAME to actually parse — the exact
# same "hash one object / parse another" TOCTOU shape Level 3 proves closed
# for `file_path` itself, just moved one level down onto the new snapshot
# path instead of eliminated. document_loader.load_document_bytes() closes
# this for good by parsing `source_bytes` directly in memory for every
# supported format (in-memory Blob for PDF, BytesIO for DOCX, direct decode
# for TXT/MD — never a filesystem path of any kind). Proven here across all
# four supported formats by pointing `file_path` at a DIFFERENT, fully
# readable file of the same format and confirming the content actually
# indexed is derived exclusively from `source_bytes`, never from that
# pathname.
# ---------------------------------------------------------------------------

def _build_minimal_pdf_bytes(text: str) -> bytes:
    """A minimal single-page real PDF with `text` drawn via a raw content
    stream — parseable end-to-end by pypdf (through PyPDFParser) with no
    external font/image dependency. No PDF-generation library (reportlab/
    fpdf) is available in this environment, so this uses pypdf's own
    low-level object API directly; test-fixture-only, never production
    code."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("latin-1")
    stream = DecodedStreamObject()
    stream.set_data(content)
    page[NameObject("/Contents")] = writer._add_object(stream)
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    resources = DictionaryObject()
    fonts_dict = DictionaryObject()
    fonts_dict[NameObject("/F1")] = writer._add_object(font)
    resources[NameObject("/Font")] = fonts_dict
    page[NameObject("/Resources")] = resources
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _build_minimal_docx_bytes(text: str) -> bytes:
    """A minimal real .docx (a zip containing only word/document.xml) —
    parseable end-to-end by docx2txt, which reads only word/document.xml
    (+ optional header/footer parts, absent here)."""
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>'
        '</w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


@pytest.mark.parametrize("extension", [".txt", ".md", ".pdf", ".docx"])
def test_reconcile_document_indexes_exactly_the_verified_bytes_for_every_format(tmp_path, extension):
    marker = f"VERIFIED{extension.strip('.').upper()}CONTENTMARKER"
    decoy_marker = f"DECOY{extension.strip('.').upper()}CONTENTTHATMUSTNEVERBEINDEXED"

    if extension in (".txt", ".md"):
        source_bytes = marker.encode("utf-8")
        decoy_bytes = decoy_marker.encode("utf-8")
    elif extension == ".pdf":
        source_bytes = _build_minimal_pdf_bytes(marker)
        decoy_bytes = _build_minimal_pdf_bytes(decoy_marker)
    else:
        source_bytes = _build_minimal_docx_bytes(marker)
        decoy_bytes = _build_minimal_docx_bytes(decoy_marker)

    content_sha256 = sha256_hex(source_bytes)

    # `file_path` points at a DIFFERENT, fully readable, correctly-formatted
    # file — if reconcile_document() (or the loader it calls) ever reopened
    # this pathname to parse instead of using `source_bytes` directly, the
    # indexed content would reflect the decoy marker instead, and the
    # assertions below would fail.
    decoy_path = tmp_path / f"decoy{extension}"
    decoy_path.write_bytes(decoy_bytes)

    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name=f"verified_bytes_identity_{extension.strip('.')}",
    )
    try:
        status, chunk_count = vi.reconcile_document(
            "upload:" + "9" * 32,
            decoy_path,
            display_name=f"report{extension}",
            expected_content_sha256=content_sha256,
            source_bytes=source_bytes,
        )
        assert status == "reindexed"
        assert chunk_count >= 1

        # Read the indexed payload text directly (bypassing similarity
        # search, which would require guessing the parser's exact final
        # whitespace/formatting) — this is the single source of truth for
        # what was actually embedded/indexed.
        records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=10, with_payload=True)
        indexed_text = " ".join(r.payload["text"] for r in records)

        assert marker in indexed_text
        assert decoy_marker not in indexed_text
        assert decoy_path.exists()  # the decoy file itself was never touched/removed
    finally:
        vi.close()
