"""
Stage 2B-F regression tests: immediate Telegram upload indexing must never
parse content obtained by an uncontrolled second reopen of the managed
source pathname — the same threat class Stage 2B-E already closed for
scripts/rebuild_qdrant.py, now closed for
handlers/document_upload.py::_load_and_index_document().

Before this pass, `_load_and_index_document()` hashed `stored.physical_path`,
then let `document_loader.load_document()` independently reopen that same
pathname to actually parse it, then re-hashed it a second time — a
"hash one object / parse another" TOCTOU window. It now reads
`stored.physical_path` exactly ONCE via
`rag.safe_files.read_regular_file_secure()` (rooted at MANAGED_UPLOADS_DIR)
and reconciles Qdrant from those captured bytes via the Stage 2B-E
`VectorIndex.reconcile_document(..., source_bytes=...)` snapshot mechanism
— never a second read of the pathname.

Entirely temporary fixtures — no real documents/uploads/Qdrant. Symlink-
dependent tests are skipped (never silently treated as pass) if the
current platform/user cannot create a symlink, matching the convention
already established in tests/test_stage2b_sidecar.py,
tests/test_stage2b_rebuild.py, and tests/test_stage2e_toctou_hardening.py.
"""

import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document

import db.documents as db_documents
import db.identity as db_identity
import handlers.document_upload as document_upload
import app.documents as app_documents
from rag.identity import sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.safe_files import SecureReadError, read_regular_file_secure
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings

_TEST_OWNER_UUID = str(uuid.uuid4())


def _make_stored_upload(uploads_dir: Path, stem: str, extension: str, content: bytes, display_name: str, owner_user_uuid: str = _TEST_OWNER_UUID) -> "app_documents.StoredUpload":
    """Builds the physical file + sidecar directly (bypassing
    _store_document_exclusively(), which several of this module's tests
    call app_documents._load_and_index_document() on its own to isolate
    the indexing step) — and, Stage 5C corrective pass, also registers the
    matching PostgreSQL 'pending' catalog row create_pending_sync() would
    normally create at storage time. Without it, _load_and_index_document()'s
    own mark_active_sync() call (now non-best-effort — see app/documents.py)
    would raise "no pending document row found to update" for every one of
    these directly-constructed StoredUpload fixtures."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    physical = uploads_dir / f"{stem}{extension}"
    physical.write_bytes(content)
    document_id = upload_document_id(stem)
    document_uuid = uuid.UUID(stem)
    content_sha256 = sha256_hex(content)
    sidecar_path = sidecar_path_for(physical)
    write_sidecar_atomic(sidecar_path, build_sidecar(document_id, display_name, physical.name, content_sha256, owner_user_uuid=owner_user_uuid))
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=uuid.UUID(owner_user_uuid),
        stored_name=physical.name, display_name=display_name, content_sha256=content_sha256,
    )
    return app_documents.StoredUpload(
        physical_path=physical, sidecar_path=sidecar_path,
        document_id=document_id, document_uuid=document_uuid, content_sha256=content_sha256, owner_user_id=uuid.UUID(owner_user_uuid),
    )


def _make_document_message(user_id: int, file_name: str, file_id: str = "fid"):
    document = SimpleNamespace(file_name=file_name, mime_type="text/plain", file_id=file_id, file_size=100)
    message = SimpleNamespace(from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=user_id), document=document)
    return message, document


def _patch_telegram(monkeypatch, file_bytes: bytes):
    monkeypatch.setattr(document_upload.bot, "get_file", AsyncMock(return_value=SimpleNamespace(file_path="documents/file.txt")))
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=file_bytes))
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())


# ---------------------------------------------------------------------------
# 1. NORMAL MANAGED UPLOAD
# ---------------------------------------------------------------------------

def test_normal_managed_upload_secure_read_succeeds_and_indexes_correct_bytes(monkeypatch, tmp_path):
    uploads_dir = tmp_path / "uploads"
    stored = _make_stored_upload(uploads_dir, "a" * 32, ".txt", b"Ordinary managed upload content.", "notes.txt")
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="normal_upload_test")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    try:
        chunk_count = app_documents._load_and_index_document(stored, "notes.txt")
        assert chunk_count == 1

        results = vi.similarity_search("Ordinary managed upload content.", requesting_user_uuid=_TEST_OWNER_UUID, k=1)
        assert len(results) == 1
        assert results[0].metadata["source"] == "notes.txt"

        # Source + sidecar remain untouched after a successful index.
        assert stored.physical_path.exists()
        assert stored.sidecar_path.exists()
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# 2. STABLE SOURCE SYMLINK
# ---------------------------------------------------------------------------

def test_stable_source_symlink_rejected_by_secure_read_directly(tmp_path):
    """Unit-level: the managed source pathname is ALREADY a symlink
    (no race — a stable, pre-existing condition) when
    _load_and_index_document() runs. Must be rejected outright; external
    content must never be returned/parsed."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"EXTERNAL ATTACKER CONTENT")
    stem = "b" * 32
    link_path = uploads_dir / f"{stem}.txt"
    try:
        os.symlink(outside, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    document_id = upload_document_id(stem)
    document_uuid = uuid.UUID(stem)
    write_sidecar_atomic(
        sidecar_path_for(link_path),
        build_sidecar(document_id, "escape.txt", link_path.name, sha256_hex(b"EXTERNAL ATTACKER CONTENT"), owner_user_uuid=_TEST_OWNER_UUID),
    )
    stored = app_documents.StoredUpload(
        physical_path=link_path, sidecar_path=sidecar_path_for(link_path),
        document_id=document_id, document_uuid=document_uuid, content_sha256=sha256_hex(b"EXTERNAL ATTACKER CONTENT"), owner_user_id=uuid.UUID(_TEST_OWNER_UUID),
    )

    with pytest.raises(SecureReadError):
        read_regular_file_secure(link_path, root=uploads_dir)
    # Same rejection reached through the actual production entry point.
    with pytest.raises(SecureReadError):
        app_documents._load_and_index_document(stored, "escape.txt")


@pytest.mark.asyncio
async def test_stable_source_symlink_end_to_end_cleanup_runs(monkeypatch, tmp_path):
    """Full lifecycle: by the time _load_and_index_document() runs, the
    managed source is a symlink (simulated by wrapping the real storage
    step so the physical file it just created is swapped for a symlink
    before indexing observes it). Indexing must reject it, and brand-new-
    upload cleanup (source + sidecar + defensive Qdrant delete) must run —
    nothing left behind."""
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    vi_mock = Mock()
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi_mock)

    external = tmp_path / "external_content.txt"
    external.write_bytes(b"EXTERNAL CONTENT THAT MUST NEVER BE INDEXED")

    real_store = app_documents._store_document_exclusively

    def storing_then_swapping_for_symlink(file_bytes, extension, display_name, owner_user_id, attempts=5):
        stored = real_store(file_bytes, extension, display_name, owner_user_id, attempts=attempts)
        stored.physical_path.unlink()
        try:
            os.symlink(external, stored.physical_path)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        return stored

    monkeypatch.setattr(app_documents, "_store_document_exclusively", storing_then_swapping_for_symlink)

    _patch_telegram(monkeypatch, b"original valid content")
    message, document = _make_document_message(1, "notes.txt")
    await document_upload.process_document_upload(message, document)

    assert list(uploads_dir.iterdir()) == []  # source (symlink) + sidecar both cleaned up
    vi_mock.add_documents.assert_not_called()
    vi_mock.reconcile_document.assert_not_called()
    vi_mock.delete_document.assert_called_once()  # defensive Qdrant cleanup still attempted


# ---------------------------------------------------------------------------
# 3. DETERMINISTIC CHECK/OPEN SWAP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deterministic_check_open_swap_rejected_and_cleanup_runs(monkeypatch, tmp_path):
    """Begin with a valid regular managed source; use the existing narrow
    rag.safe_files._test_pre_open_hook to swap the pathname to a symlink
    in the exact window between lstat and open — the same boundary the
    Stage 2B-E rebuild TOCTOU regressions exercise. Secure read must
    reject it; external bytes must never reach the loader/Qdrant; cleanup
    must leave no managed source/sidecar/Qdrant orphan."""
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    vi_mock = Mock()
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi_mock)

    external = tmp_path / "external_content.txt"
    external.write_bytes(b"EXTERNAL CONTENT THAT MUST NEVER BE INDEXED")

    def swap_hook(path):
        os.remove(path)
        try:
            os.symlink(external, path)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")

    def racing_secure_read(path, *, root):
        return read_regular_file_secure(path, root=root, _test_pre_open_hook=swap_hook)

    monkeypatch.setattr(app_documents, "read_regular_file_secure", racing_secure_read)

    _patch_telegram(monkeypatch, b"original valid content")
    message, document = _make_document_message(1, "notes.txt")
    await document_upload.process_document_upload(message, document)

    assert list(uploads_dir.iterdir()) == []
    vi_mock.add_documents.assert_not_called()
    vi_mock.reconcile_document.assert_not_called()
    vi_mock.delete_document.assert_called_once()


# ---------------------------------------------------------------------------
# 4. HASH MISMATCH
# ---------------------------------------------------------------------------

def test_hash_mismatch_fails_before_qdrant_mutation(monkeypatch, tmp_path):
    """The stored physical file is securely readable (valid, in-root,
    regular) but its content does not match the authoritative
    content_sha256 recorded at storage time (a controlled post-storage
    mutation) — must fail with NO Qdrant mutation, never silently index
    the new content under the old hash."""
    uploads_dir = tmp_path / "uploads"
    stored = _make_stored_upload(uploads_dir, "c" * 32, ".txt", b"original stored content", "notes.txt")
    # Tamper with the on-disk content AFTER the sidecar/hash were recorded.
    stored.physical_path.write_bytes(b"tampered content, different from the recorded hash")
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="hash_mismatch_test")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    try:
        with pytest.raises(Exception) as exc_info:
            app_documents._load_and_index_document(stored, "notes.txt")
        assert type(exc_info.value).__name__ == "SourceMutatedError"
        assert vi.get_stats(requesting_user_uuid=_TEST_OWNER_UUID)["total_documents"] == 0  # no Qdrant mutation occurred
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# 5. NO ORIGINAL-PATH REOPEN
#
# Stage 2B-F Blocker 1 (a later audit finding against the fix this whole
# module was originally written to prove): reconcile_document() used to
# hand the already-verified bytes to the loader via a private TEMPORARY
# SNAPSHOT FILE — which document_loader.load_document() (PyPDFLoader /
# TextLoader / Docx2txtLoader under the hood) would then reopen BY
# PATHNAME to actually parse it. That is the exact same "hash one object /
# parse another" TOCTOU shape this module's other tests already prove is
# closed for `stored.physical_path` — just moved one level down onto the
# new snapshot path instead of eliminated. The fix now parses the verified
# bytes directly in memory via document_loader.load_document_bytes() (an
# in-memory `Blob` for PDF, a `BytesIO` stream for DOCX, a direct decode
# for TXT/MD — see rag/loader.py) — no snapshot file is ever written, so
# there is no pathname left for a concurrent write to swap between
# verification and parsing.
# ---------------------------------------------------------------------------

def test_reconcile_parses_bytes_directly_never_via_a_reopened_pathname(monkeypatch, tmp_path):
    uploads_dir = tmp_path / "uploads"
    original_bytes = b"content for in-memory-parsing proof"
    stored = _make_stored_upload(uploads_dir, "d" * 32, ".txt", original_bytes, "notes.txt")
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    # The path-based loader (which would reopen whatever pathname it's
    # given) must never be invoked at all for a managed upload.
    path_based_load = Mock(wraps=app_documents.document_loader.load_document)
    monkeypatch.setattr(app_documents.document_loader, "load_document", path_based_load)

    captured_bytes = []
    real_load_document_bytes = app_documents.document_loader.load_document_bytes

    def spying_load_document_bytes(source_bytes, **kwargs):
        captured_bytes.append(source_bytes)
        return real_load_document_bytes(source_bytes, **kwargs)

    monkeypatch.setattr(app_documents.document_loader, "load_document_bytes", spying_load_document_bytes)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="no_reopen_test")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    try:
        pre_existing = set(uploads_dir.iterdir())

        app_documents._load_and_index_document(stored, "notes.txt")

        path_based_load.assert_not_called()  # no pathname-based reopen at all
        assert len(captured_bytes) == 1
        assert captured_bytes[0] == original_bytes  # the exact verified bytes, not a copy read back from disk

        # No temporary snapshot file (or anything else) was ever created
        # alongside the managed source/sidecar.
        assert set(uploads_dir.iterdir()) == pre_existing
    finally:
        vi.close()


@pytest.mark.asyncio
async def test_original_pathname_replaced_after_secure_read_has_no_effect_on_indexed_content(monkeypatch, tmp_path):
    """Make the original managed pathname unusable/replaced immediately
    AFTER secure read has already captured its bytes.

    Stage 5C corrective pass #4 (Blocker 5): a mutation landing here used
    to be harmless-by-omission (the already-captured original bytes got
    indexed, the replacement was simply never consumed) — but that still
    left Qdrant/sidecar/catalog describing a snapshot the DURABLE FILE ON
    DISK no longer matched, which Principle 1 forbids reporting as
    success. `_load_and_index_document()`'s final pre-activation
    revalidation (a SECOND secure read, added by Blocker 5) now re-detects
    this exact mutation and fails ingestion closed instead: the
    replacement content is still never indexed, but neither is the
    original snapshot — no document becomes active while the file
    disagrees with what was captured."""
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="post_read_swap_test")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)

    real_secure_read = app_documents.read_regular_file_secure

    def racing_secure_read(path, *, root):
        secure_bytes = real_secure_read(path, root=root)
        Path(path).write_bytes(b"REPLACEMENT CONTENT AFTER SECURE READ - must never be indexed")
        return secure_bytes

    monkeypatch.setattr(app_documents, "read_regular_file_secure", racing_secure_read)

    _patch_telegram(monkeypatch, b"Content captured before any replacement occurs.")
    message, document = _make_document_message(1, "notes.txt")
    try:
        await document_upload.process_document_upload(message, document)

        # This upload went through the REAL Telegram handler (telegram_id=1),
        # which resolves ownership via app.identity.resolve_user_uuid() ->
        # db.identity's fixture-faked resolver — not the module-level
        # _TEST_OWNER_UUID constant used by this file's direct
        # _load_and_index_document() calls elsewhere.
        owner_uuid = str(db_identity.resolve_or_create_user_by_telegram_id_sync(1))
        assert vi.get_stats(requesting_user_uuid=owner_uuid)["total_documents"] == 0
        results = vi.similarity_search("Content captured before any replacement occurs.", requesting_user_uuid=owner_uuid, k=5)
        assert all("REPLACEMENT CONTENT" not in r.page_content for r in results)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# 6. SAFE PAYLOAD
# ---------------------------------------------------------------------------

def test_indexed_payload_carries_no_temp_or_absolute_path(monkeypatch, tmp_path):
    uploads_dir = tmp_path / "uploads"
    stored = _make_stored_upload(uploads_dir, "e" * 32, ".txt", b"Payload privacy check content.", "My Report.txt")
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="safe_payload_test")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    try:
        app_documents._load_and_index_document(stored, "My Report.txt")

        results = vi.similarity_search("Payload privacy check content.", requesting_user_uuid=_TEST_OWNER_UUID, k=1)
        assert len(results) == 1
        metadata = results[0].metadata

        assert metadata["source"] == "My Report.txt"  # original display filename preserved
        assert "file_path" not in metadata  # no snapshot/managed path field at all
        for value in metadata.values():
            if isinstance(value, str):
                assert str(tmp_path) not in value  # no absolute path leaked into any field
                assert not value.startswith("/tmp")
        assert metadata["document_id"] == stored.document_id
        assert metadata["stored_name"] == stored.physical_path.name
        assert metadata["content_sha256"] == stored.content_sha256
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# 7. FORMAT PRESERVATION — the stored physical file's extension still
# selects the correct in-memory parser via load_document_bytes(suffix=...).
# document_loader.load_document_bytes() is mocked (never actually parses)
# so this proves the dispatch/suffix contract without needing genuinely-
# parseable PDF/DOCX binary content and without any live/provider call.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extension", [".txt", ".md", ".pdf", ".docx"])
def test_extension_selects_correct_format_for_in_memory_parsing(monkeypatch, tmp_path, extension):
    uploads_dir = tmp_path / "uploads"
    original_bytes = b"arbitrary bytes standing in for real format content (loader mocked below, never actually parsed)"
    stored = _make_stored_upload(
        uploads_dir, "f" * 32, extension,
        original_bytes,
        f"report{extension}",
    )
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    captured = {}

    def spy_load_document_bytes(source_bytes, **kwargs):
        captured["source_bytes"] = source_bytes
        captured["suffix"] = kwargs.get("suffix")
        # A non-empty stub chunk (Stage 5C corrective pass #4, Blocker 10:
        # a genuinely empty chunk list is no longer accepted as a
        # successful ingest) — this test only proves suffix/byte dispatch,
        # never real chunk content.
        return [Document(page_content="stub chunk content", metadata={"chunk_index": 0})]

    monkeypatch.setattr(app_documents.document_loader, "load_document_bytes", spy_load_document_bytes)

    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name=f"format_test_{extension.strip('.')}")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    try:
        app_documents._load_and_index_document(stored, f"report{extension}")

        assert captured["suffix"].lower() == extension
        assert captured["source_bytes"] == original_bytes
    finally:
        vi.close()
