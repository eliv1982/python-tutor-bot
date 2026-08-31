"""
Stage 1B / 1B.1 regression tests: safe document upload containment.

Covers:
- path traversal is impossible by construction (physical path is always
  application-generated, never derived from the user-supplied filename)
- duplicate original filenames never overwrite each other
- extension allowlist (source of truth: rag.loader.SUPPORTED_EXTENSIONS)
  rejects unsupported formats before the loader/vector-index are invoked
- size limit is enforced against actual downloaded bytes, before any
  disk write / parsing / indexing
- the original filename is preserved for RAG source attribution even
  though it never controls physical storage
- newly created storage files are cleaned up on loader/indexing failure
- Stage 1A's download_telegram_file() token-safety boundary still holds
  for the document upload path
- (1B.1) application-managed uploads are not blindly reloaded with their
  opaque UUID filename as source by the startup/reference directory scan
- (1B.1) a notification failure after successful ingestion does not roll
  back the stored file or misreport the upload as failed
- (1B.1) physical file creation is exclusive: a UUID collision retries
  onto a fresh candidate instead of touching the existing file
- (1B.1) legacy `.doc` is rejected consistently by both the loader
  dispatch and the directory scan, matching SUPPORTED_EXTENSIONS

All Telegram/OpenAI/Chroma/embeddings boundaries are mocked. No network
access is performed by this test module. Every test monkeypatches
handlers.app_documents.MANAGED_UPLOADS_DIR (and, where the directory
scan is exercised, rag.loader.MANAGED_UPLOADS_DIR) to a pytest tmp_path,
so the real (gitignored) data/documents directory is never touched.
"""

import itertools
import re
import uuid as uuid_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import db.identity as db_identity
import handlers.document_upload as document_upload
import app.documents as app_documents
import rag.loader as rag_loader
from rag.index import VectorIndex
from rag.loader import SUPPORTED_EXTENSIONS, document_loader as real_document_loader
from rag_fakes import DeterministicFakeEmbeddings

UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")



def _requesting_uuid_for_telegram_id(telegram_id: int) -> str:
    """The tests below drive uploads through the real Telegram handler
    (document_upload.process_document_upload), which resolves ownership
    via app.identity.resolve_user_uuid() -> db.identity's (fixture-faked,
    see conftest.py's _default_fake_preferences) resolver — stable per
    telegram_id within a test. Retrieving that same UUID here (rather than
    a fresh/unrelated one) is what lets these tests query VectorIndex as
    "the uploader themselves" without hardcoding a Telegram integer as if
    it were still the canonical Qdrant owner identity."""
    return str(db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id))


def _make_document_message(
    user_id: int,
    file_name: str,
    mime_type: str = "application/pdf",
    file_id: str = "fake-doc-id",
    file_size: int = 1234,
):
    document = SimpleNamespace(
        file_name=file_name,
        mime_type=mime_type,
        file_id=file_id,
        file_size=file_size,
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        document=document,
    )
    return message, document


def _patch_telegram(monkeypatch, file_bytes: bytes, file_path: str = "documents/file_1.pdf"):
    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path=file_path)),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=file_bytes))
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())


# ---------------------------------------------------------------------------
# A. Path traversal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("malicious_name", [
    "../../outside.txt",
    "..\\..\\outside.txt",
    "/absolute/path.txt",
    "C:\\outside.txt",
    "....//....//outside.txt",
])
async def test_path_traversal_impossible_by_construction(monkeypatch, tmp_path, malicious_name):
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    # reconcile_document() (Stage 2B-F: what _load_and_index_document()
    # actually calls, superseding the old direct add_documents() call) is
    # mocked directly rather than its loader/embeddings internals — this
    # test is about storage mechanics, not indexed content, and must never
    # reach the real OpenAIEmbeddings network call.
    monkeypatch.setattr(app_documents.get_vector_index(), "reconcile_document", Mock(return_value=("reindexed", 1)))

    _patch_telegram(monkeypatch, b"hello world")
    message, document = _make_document_message(1, malicious_name)

    await document_upload.process_document_upload(message, document)

    created = list(tmp_path.iterdir())
    assert len(created) == 2, f"expected exactly one physical file + one sidecar inside tmp_path, got {created}"
    physical_path = next(p for p in created if not p.name.endswith(".meta.json"))
    sidecar_path = next(p for p in created if p.name.endswith(".meta.json"))

    # Proves the path is application-generated, not merely sanitized: the
    # stem must be a uuid4 hex, bearing no relation to the malicious name.
    assert UUID_HEX_RE.match(physical_path.stem), physical_path.name
    assert physical_path.suffix == ".txt"
    assert physical_path.name != Path(malicious_name).name
    assert sidecar_path.name == f"{physical_path.stem}.meta.json"

    # And it must actually live inside the intended root.
    assert physical_path.resolve().parent == tmp_path.resolve()


# ---------------------------------------------------------------------------
# B. Duplicate filenames
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_original_filenames_do_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    # reconcile_document() (Stage 2B-F: what _load_and_index_document()
    # actually calls, superseding the old direct add_documents() call) is
    # mocked directly rather than its loader/embeddings internals — this
    # test is about storage mechanics, not indexed content, and must never
    # reach the real OpenAIEmbeddings network call.
    monkeypatch.setattr(app_documents.get_vector_index(), "reconcile_document", Mock(return_value=("reindexed", 1)))

    _patch_telegram(monkeypatch, b"first content")
    message1, document1 = _make_document_message(1, "notes.txt", file_id="id-1")
    await document_upload.process_document_upload(message1, document1)

    _patch_telegram(monkeypatch, b"second content, different bytes")
    message2, document2 = _make_document_message(1, "notes.txt", file_id="id-2")
    await document_upload.process_document_upload(message2, document2)

    created = list(tmp_path.iterdir())
    assert len(created) == 4  # 2 physical files + 2 sidecars
    physical_files = [p for p in created if not p.name.endswith(".meta.json")]
    assert len(physical_files) == 2
    names = {p.name for p in physical_files}
    assert len(names) == 2  # different physical identities

    contents = {p.read_bytes() for p in physical_files}
    assert contents == {b"first content", b"second content, different bytes"}


# ---------------------------------------------------------------------------
# C. Extension/type validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unsupported_extension_rejected_before_loader_and_index(monkeypatch, tmp_path):
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    load_mock = Mock()
    add_mock = Mock()
    monkeypatch.setattr(app_documents.document_loader, "load_document", load_mock)
    monkeypatch.setattr(app_documents.get_vector_index(), "add_documents", add_mock)

    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)
    get_file_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "get_file", get_file_mock)

    message, document = _make_document_message(1, "malware.exe", mime_type="application/pdf")
    await document_upload.process_document_upload(message, document)

    load_mock.assert_not_called()
    add_mock.assert_not_called()
    get_file_mock.assert_not_called()  # rejected before even downloading
    assert list(tmp_path.iterdir()) == []
    send_message_mock.assert_awaited_once()
    assert "не поддерживается".casefold() in send_message_mock.await_args.args[1].casefold() \
        or "неподдерживаемое" in send_message_mock.await_args.args[1].casefold()


def test_extension_allowlist_matches_actual_loader_capabilities():
    """.doc is intentionally excluded: Docx2txtLoader cannot parse the
    legacy binary format, so it must not be advertised as supported."""
    assert SUPPORTED_EXTENSIONS == frozenset({'.pdf', '.txt', '.md', '.docx'})
    assert '.doc' not in SUPPORTED_EXTENSIONS


def test_doc_extension_rejected_by_loader_dispatch():
    """SUPPORTED_EXTENSIONS must be authoritative for the loader dispatch
    too, not just the Telegram upload gate — .doc has no matching branch
    and must raise before any file access is attempted."""
    with pytest.raises(ValueError):
        real_document_loader.load_document(Path("legacy_report.doc"))


def test_doc_files_skipped_by_directory_scan(tmp_path, monkeypatch):
    """The startup/reference directory scan must derive its accepted
    formats from the same SUPPORTED_EXTENSIONS source of truth, so a
    stray .doc file placed in data/documents is silently skipped rather
    than reaching (and failing inside) the loader."""
    managed_dir = tmp_path / "uploads"
    managed_dir.mkdir()
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", managed_dir)

    (tmp_path / "legacy.doc").write_bytes(b"fake legacy binary doc bytes")
    (tmp_path / "notes.txt").write_text("hello world", encoding="utf-8")

    chunks = real_document_loader.load_directory(tmp_path)
    sources = {c.metadata["source"] for c in chunks}
    assert sources == {"notes.txt"}


# ---------------------------------------------------------------------------
# D. Size limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oversized_document_rejected_before_parsing_and_indexing(monkeypatch, tmp_path):
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(app_documents, "MAX_DOCUMENT_SIZE_BYTES", 100)
    load_mock = Mock()
    add_mock = Mock()
    monkeypatch.setattr(app_documents.document_loader, "load_document", load_mock)
    monkeypatch.setattr(app_documents.get_vector_index(), "add_documents", add_mock)

    oversized_bytes = b"x" * 101
    _patch_telegram(monkeypatch, oversized_bytes)
    send_message_mock = document_upload.bot.send_message

    message, document = _make_document_message(1, "big.txt", file_size=10)  # Telegram metadata lies
    await document_upload.process_document_upload(message, document)

    load_mock.assert_not_called()
    add_mock.assert_not_called()
    assert list(tmp_path.iterdir()) == []  # never written to disk
    assert any("слишком" in call.args[1].lower() for call in send_message_mock.await_args_list)


# ---------------------------------------------------------------------------
# E. Original filename metadata / source attribution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_original_filename_preserved_as_rag_source_not_uuid(monkeypatch, tmp_path):
    """
    Stage 2B-F: `_load_and_index_document()` now indexes via
    `VectorIndex.reconcile_document(..., source_bytes=...)`, not a direct
    `add_documents()` call — a real local-persistent Qdrant (deterministic
    fake embeddings, never a real provider call) is used here so the
    actually-indexed payload can be inspected directly, exactly as it will
    be retrieved, rather than intercepting an internal call that no longer
    exists on this path.
    """
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="original_filename_test")
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    # Use the REAL loader (TextLoader for .txt) to prove end-to-end that a
    # UUID storage filename does not leak into displayed/cited sources.
    monkeypatch.setattr(app_documents, "document_loader", real_document_loader)

    _patch_telegram(monkeypatch, b"Python is a great language for tutoring.")
    message, document = _make_document_message(1, "My Study Notes.txt")

    try:
        await document_upload.process_document_upload(message, document)

        physical_files = [p for p in uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
        assert len(physical_files) == 1

        results = vi.similarity_search(
            "Python is a great language for tutoring.",
            requesting_user_uuid=_requesting_uuid_for_telegram_id(1),
            k=1,
        )
        assert len(results) == 1
        metadata = results[0].metadata
        assert metadata["source"] == "My Study Notes.txt"
        assert not UUID_HEX_RE.match(Path(metadata["source"]).stem)
        # The internal (uuid-named) temp-snapshot/managed-storage path is
        # never part of the safe Qdrant payload at all (Stage 2B-F Section
        # C: no snapshot path may reach Qdrant metadata).
        assert "file_path" not in metadata
        # (Stage 2B) stable identity/fingerprint metadata is attached too.
        assert metadata["document_id"].startswith("upload:")
        assert metadata["stored_name"] == physical_files[0].name
        assert metadata["content_sha256"]
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# E.1 (1B.1) Startup/reference scan must not re-attribute managed uploads
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_managed_upload_not_reloaded_with_opaque_source_on_startup_scan(monkeypatch, tmp_path):
    """
    Goal 1 regression (Stage 1B.1): after an upload, a startup/reference
    directory scan (load_directory) must not blindly reload the
    application-managed physical file and re-index it under its opaque
    UUID filename as source.

    Deliberate interim limitation: this proves the managed upload is
    skipped by the scan (so it can't get duplicated/mis-attributed), not
    that it is reconstructable from disk if the persistent Chroma store is
    destroyed. That recovery path belongs to the later PostgreSQL/Qdrant
    registry stage — until then, an upload only already in Chroma is the
    sole source of truth for it.
    """
    documents_dir = tmp_path / "documents"
    managed_dir = documents_dir / "uploads"
    managed_dir.mkdir(parents=True)

    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", managed_dir)
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", managed_dir)
    monkeypatch.setattr(app_documents, "document_loader", real_document_loader)

    # Stage 2B-F: real local Qdrant instead of intercepting add_documents()
    # directly, which _load_and_index_document() no longer calls — see
    # test_original_filename_preserved_as_rag_source_not_uuid above.
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="startup_scan_test")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)

    _patch_telegram(monkeypatch, b"Python functions are defined with the def keyword.")
    message, document = _make_document_message(1, "python_notes.txt")
    await document_upload.process_document_upload(message, document)

    # Step 2: initial ingest used the original filename as source.
    results = vi.similarity_search(
        "Python functions are defined with the def keyword.",
        requesting_user_uuid=_requesting_uuid_for_telegram_id(1),
        k=1,
    )
    assert len(results) == 1
    assert results[0].metadata["source"] == "python_notes.txt"

    # A manually-managed reference document living alongside (not inside)
    # the uploads subdirectory, to prove the scan still indexes real
    # reference content normally.
    (documents_dir / "python_intro.txt").write_text(
        "Python intro reference material.", encoding="utf-8"
    )

    # Step 3/4: exercise the startup/reference scan directly.
    try:
        reloaded_chunks = real_document_loader.load_directory(documents_dir)
        reloaded_sources = {c.metadata["source"] for c in reloaded_chunks}

        assert reloaded_sources == {"python_intro.txt"}
        for source in reloaded_sources:
            assert not UUID_HEX_RE.match(Path(source).stem)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# F. Failure cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loader_failure_cleans_up_newly_created_file(monkeypatch, tmp_path, caplog):
    """
    Stage 1D.1: document_loader.load_document_bytes() (Stage 2B-F: the
    in-memory bytes parser reconcile_document() now calls for managed
    uploads) can fail on a provider/HTTP exception too (its caller,
    vector_index.reconcile_document(), reaches OpenAIEmbeddings over the
    network for OTHER chunks in the same request lifecycle), so the
    failure log here only ever carries safe metadata — never the raw
    exception text. Cleanup/user-response guarantees from Stage 1B are
    unchanged.
    """
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(
        app_documents.document_loader, "load_document_bytes",
        Mock(side_effect=ValueError("corrupt PDF stream at offset 42")),
    )
    add_mock = Mock()
    monkeypatch.setattr(app_documents.get_vector_index(), "add_documents", add_mock)

    _patch_telegram(monkeypatch, b"%PDF-1.4 fake pdf bytes")
    send_message_mock = document_upload.bot.send_message
    message, document = _make_document_message(1, "report.pdf")

    import logging
    with caplog.at_level(logging.ERROR):
        await document_upload.process_document_upload(message, document)

    assert list(tmp_path.iterdir()) == []  # orphan file removed
    add_mock.assert_not_called()

    # The raw exception message is never logged...
    assert "corrupt PDF stream at offset 42" not in caplog.text
    # ...but a safe, structured event + exception class name still is.
    assert "Document upload failed" in caplog.text
    assert "ValueError" in caplog.text
    # ...and it's not exposed raw to the user either.
    last_message = send_message_mock.await_args.args[1]
    assert "corrupt PDF stream" not in last_message
    assert "ошибка" in last_message.lower()


@pytest.mark.asyncio
async def test_indexing_failure_cleans_up_newly_created_file(monkeypatch, tmp_path, caplog):
    """
    Stage 1D.1: vector_index.reconcile_document() (Stage 2B-F: the call
    _load_and_index_document() now makes, superseding the old direct
    add_documents() call) embeds chunks via OpenAIEmbeddings (a network
    call) before writing to Qdrant, so a RuntimeError here can genuinely
    be a provider/HTTP exception — the failure log only ever carries safe
    metadata, never raw exception text.
    """
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(
        app_documents.get_vector_index(), "reconcile_document",
        Mock(side_effect=RuntimeError("qdrant collection unavailable")),
    )

    _patch_telegram(monkeypatch, b"some text content")
    send_message_mock = document_upload.bot.send_message
    message, document = _make_document_message(1, "notes.txt")

    import logging
    with caplog.at_level(logging.ERROR):
        await document_upload.process_document_upload(message, document)

    assert list(tmp_path.iterdir()) == []  # orphan file removed
    assert "qdrant collection unavailable" not in caplog.text
    assert "Document upload failed" in caplog.text
    assert "RuntimeError" in caplog.text
    last_message = send_message_mock.await_args.args[1]
    assert "qdrant collection unavailable" not in last_message
    assert "ошибка" in last_message.lower()


# ---------------------------------------------------------------------------
# F.1 (1B.1) Notification failure after successful ingestion must not roll back
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_failure_after_successful_ingestion_does_not_rollback(monkeypatch, tmp_path, caplog):
    """
    Goal 2 regression (Stage 1B.1 / hardened in 1B.2): vector_index.
    reconcile_document() (Stage 2B-F: the call _load_and_index_document()
    now makes) succeeding must be the ingestion success boundary. A later
    failure sending the final "success" Telegram message must not delete
    the already-ingested file, must not be reported to the user as a
    failed upload, must not trigger a second cleanup attempt — and (1B.2)
    the exception itself, which can be a pyTelegramBotAPI HTTP error
    carrying a token-bearing Telegram API URL, must never reach the logs
    verbatim or via exc_info. Real local Qdrant (deterministic fake
    embeddings) rather than mocking add_documents(), which
    _load_and_index_document() no longer calls directly.
    """
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=DeterministicFakeEmbeddings(), collection_name="notification_failure_test")
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"some content"))

    fake_token = "123456789:FAKE-TOKEN-FOR-NOTIFICATION-LOG-LEAK-TEST"
    leaking_message = (
        f"Failed to fetch https://api.telegram.org/bot{fake_token}"
        "/sendMessage: 502 Bad Gateway"
    )

    sent_messages = []

    async def flaky_send_message(chat_id, text, *args, **kwargs):
        sent_messages.append(text)
        # 1st call: "loading", 2nd: "indexing" (both must succeed so
        # ingestion actually completes), 3rd: the final success message,
        # which fails here with a token-bearing exception to simulate a
        # realistic pyTelegramBotAPI HTTP delivery failure.
        if len(sent_messages) >= 3:
            raise Exception(leaking_message)

    monkeypatch.setattr(document_upload.bot, "send_message", flaky_send_message)

    message, document = _make_document_message(1, "notes.txt")

    import logging
    try:
        with caplog.at_level(logging.DEBUG):
            await document_upload.process_document_upload(message, document)

        # Ingestion happened exactly once and was never treated as failed.
        assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(1))["total_documents"] >= 1
        created = list(uploads_dir.iterdir())
        assert len(created) == 2, "successfully ingested file + its sidecar must not be deleted"

        # The failure was logged safely (event name + user_id + exception type)...
        log_text = caplog.text
        assert "notification failed" in log_text.lower()
        assert "already committed" in log_text.lower()
        assert "Exception" in log_text  # exception class/type is the safe part
        # ...but neither the fake token, the token-bearing URL, nor the raw
        # exception text was logged, and no traceback was emitted for it.
        assert fake_token not in log_text
        assert leaking_message not in log_text
        assert "api.telegram.org" not in log_text
        assert "Traceback" not in log_text
        # ...and no misleading "upload failed" message was ever sent to the user.
        assert not any("ошибка при загрузке" in m.lower() for m in sent_messages)
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# G. Exclusive physical file creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exclusive_creation_does_not_overwrite_existing_candidate(monkeypatch, tmp_path):
    """
    Goal 3 regression (Stage 1B.1): a UUID candidate collision must retry
    onto a fresh candidate rather than truncating/overwriting whatever
    already occupies that path.
    """
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    # reconcile_document() (Stage 2B-F: what _load_and_index_document()
    # actually calls, superseding the old direct add_documents() call) is
    # mocked directly rather than its loader/embeddings internals — this
    # test is about storage mechanics, not indexed content, and must never
    # reach the real OpenAIEmbeddings network call.
    monkeypatch.setattr(app_documents.get_vector_index(), "reconcile_document", Mock(return_value=("reindexed", 1)))

    colliding_hex = "a" * 32
    fresh_hex = "b" * 32
    existing_path = tmp_path / f"{colliding_hex}.txt"
    existing_path.write_bytes(b"PRECIOUS EXISTING CONTENT")

    # Resolve (and thereby cache, in conftest.py's autouse fake resolver —
    # see _default_fake_preferences) this test's real owner UUID BEFORE
    # patching uuid.uuid4() below. Stage 5C corrective pass #5 (Blocker 1)
    # now freshly re-reads and validates the durable sidecar before
    # activation, including its owner_user_uuid field — without this
    # priming call, the global uuid4() patch two lines down would also be
    # hit by the fake resolver's own internal `uuid.uuid4()` call (the SAME
    # `uuid` module object, same rationale as the comment below), producing
    # a non-UUID-shaped SimpleNamespace as this upload's owner id and
    # therefore a sidecar that fails that fresh validation.
    _requesting_uuid_for_telegram_id(1)

    # First two calls drive the physical-file collision/retry under test;
    # `app_documents.uuid` and `rag.sidecar`'s own `uuid` import are the
    # SAME module object (Python caches imports in sys.modules), so this
    # patch also affects the sidecar's internal temp-file uuid4() call —
    # an unbounded repeat() keeps the test robust to that without needing
    # a distinct assertion on it.
    hex_values = itertools.chain([colliding_hex, fresh_hex], itertools.repeat("c" * 32))
    monkeypatch.setattr(
        app_documents.uuid, "uuid4",
        lambda: SimpleNamespace(hex=next(hex_values)),
    )

    _patch_telegram(monkeypatch, b"new upload bytes")
    message, document = _make_document_message(1, "notes.txt")
    await document_upload.process_document_upload(message, document)

    fresh_path = tmp_path / f"{fresh_hex}.txt"
    fresh_sidecar = tmp_path / f"{fresh_hex}.meta.json"
    assert existing_path.read_bytes() == b"PRECIOUS EXISTING CONTENT"  # untouched
    assert fresh_path.exists()
    assert fresh_path.read_bytes() == b"new upload bytes"
    assert fresh_sidecar.exists()
    assert len(list(tmp_path.iterdir())) == 3  # bystander + fresh physical + fresh sidecar


def test_store_document_exclusively_cleans_up_after_write_failure(monkeypatch, tmp_path):
    """
    Goal (Stage 1B.2): once the exclusive-create open() call itself has
    succeeded, a later write/close failure must remove exactly that
    partially-created file, must not touch any unrelated/pre-existing
    file, and the original exception must remain the one that propagates
    (not masked by anything cleanup does).
    """
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)

    bystander = tmp_path / "bystander.txt"
    bystander.write_bytes(b"UNRELATED PRE-EXISTING CONTENT")

    created_paths = []

    class FailingWriteHandle:
        """Simulates open(path, 'xb') succeeding (file created, empty,
        on disk) but the subsequent write failing."""

        def __init__(self, path):
            self.path = Path(path)
            self.path.touch()  # exclusive creation itself succeeded
            created_paths.append(self.path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data):
            raise OSError("simulated disk write failure")

    def fake_open(path, mode):
        assert mode == 'xb'
        if Path(path).exists():
            raise FileExistsError()
        return FailingWriteHandle(path)

    monkeypatch.setattr(app_documents, "open", fake_open, raising=False)

    with pytest.raises(OSError, match="simulated disk write failure"):
        app_documents._store_document_exclusively(b"payload bytes", ".txt", "notes.txt", uuid_module.uuid4())

    assert len(created_paths) == 1, "exactly one candidate should have been exclusively created"
    assert not created_paths[0].exists(), "the partially-written file must be cleaned up"

    # The unrelated bystander file must never be touched by this cleanup.
    assert bystander.exists()
    assert bystander.read_bytes() == b"UNRELATED PRE-EXISTING CONTENT"
    assert list(tmp_path.iterdir()) == [bystander]


# ---------------------------------------------------------------------------
# H. Existing Telegram download security boundary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_download_exception_does_not_leak_token_in_logs(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    fake_token = "987654321:FAKE-TOKEN-FOR-DOC-LOG-LEAK-TEST"
    leaking_message = (
        f"Failed to fetch https://api.telegram.org/file/bot{fake_token}"
        "/documents/file_1.pdf: 404 Not Found"
    )

    async def raise_leaking_error(file_id):
        raise Exception(leaking_message)

    monkeypatch.setattr(document_upload.bot, "get_file", raise_leaking_error)
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())
    monkeypatch.setattr(app_documents.document_loader, "load_document", Mock())
    monkeypatch.setattr(app_documents.get_vector_index(), "add_documents", Mock())

    message, document = _make_document_message(1, "notes.pdf")

    with caplog.at_level(logging.DEBUG):
        await document_upload.process_document_upload(message, document)

    log_text = caplog.text
    assert fake_token not in log_text
    assert leaking_message not in log_text
    assert "api.telegram.org" not in log_text
    assert "document_download" in log_text
    assert list(tmp_path.iterdir()) == []  # nothing was ever written
