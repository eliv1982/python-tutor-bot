"""
Stage 5B regression tests: the document ingestion transaction
(app.documents.ingest_document()) as an adapter-independent application
boundary, callable directly with no Telegram message/document objects.

The underlying hardened primitives (_store_document_exclusively,
_load_and_index_document, _cleanup_new_upload, the cancellation
resolvers) are unchanged code, moved verbatim from
handlers/document_upload.py, and remain exhaustively covered end to end
through the Telegram handler by the existing (retargeted) Stage 1B/1E/2B/
2C/2D/2E/2F/3A suites — this file does not re-derive that coverage. It
proves specifically:
- the new ingest_document() entry point itself is usable with no
  telebot/Message/Document objects anywhere in the call;
- its structured DocumentIngestResult correctly reports each outcome
  (success, unsupported extension, oversized, storage/indexing failure
  with rollback);
- the `before_indexing` hook fires at the right point and inherits the
  same cancellation-safety window the original inline Telegram status
  message send had;
- handlers/document_upload.py delegates to it rather than owning a
  second copy of the transaction.

All Qdrant access uses the existing deterministic local-persistent
VectorIndex pattern (tests/rag_fakes.py) — no real OpenAI/Qdrant network
calls anywhere in this module.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.documents as app_documents
from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings


@pytest.fixture
def real_vector_index(tmp_path, monkeypatch):
    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage5b_documents_test",
    )
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)
    yield SimpleNamespace(vi=vi, uploads_dir=uploads_dir)
    vi.close()


# ---------------------------------------------------------------------------
# A. Callable without Telegram objects + success path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_document_success_with_no_telegram_objects(real_vector_index):
    """ingest_document() takes only bytes/str/int primitives — no
    telebot.types.Message/Document, no bot instance."""
    result = await app_documents.ingest_document(
        file_bytes=b"Python functions are defined with the def keyword.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=42,
    )

    assert result.success is True
    assert result.chunk_count == 1
    assert result.stored is not None
    assert result.stored.owner_user_id == 42
    assert result.file_size_bytes == len(b"Python functions are defined with the def keyword.")

    physical_files = [p for p in real_vector_index.uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    assert len(physical_files) == 1

    results = real_vector_index.vi.similarity_search(
        "Python functions are defined with the def keyword.", requesting_user_id=42, k=1
    )
    assert len(results) == 1
    assert results[0].metadata["source"] == "notes.txt"


@pytest.mark.asyncio
async def test_ingest_document_before_indexing_hook_runs_between_storage_and_indexing(real_vector_index, monkeypatch):
    """The hook replaces the original inline Telegram
    'Индексирую документ...' send — it must fire AFTER the file is
    durably stored but BEFORE indexing runs."""
    call_order = []
    real_load_and_index = app_documents._load_and_index_document

    def spying_load_and_index(stored, display_name):
        call_order.append("indexing")
        return real_load_and_index(stored, display_name)

    async def hook():
        call_order.append("hook")
        # The physical file must already be durable when the hook fires.
        assert list(real_vector_index.uploads_dir.iterdir())

    monkeypatch.setattr(app_documents, "_load_and_index_document", spying_load_and_index)
    result = await app_documents.ingest_document(
        file_bytes=b"hook ordering content",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=1,
        before_indexing=hook,
    )

    assert result.success is True
    assert call_order == ["hook", "indexing"]


# ---------------------------------------------------------------------------
# B. Structured rejection — no exception, no side effects
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_document_rejects_unsupported_extension_before_any_storage(real_vector_index):
    result = await app_documents.ingest_document(
        file_bytes=b"anything",
        extension=".exe",
        display_name="malware.exe",
        owner_user_id=1,
    )

    assert result.success is False
    assert result.rejected_reason == "unsupported_extension"
    assert result.error_type is None
    assert not real_vector_index.uploads_dir.exists()


@pytest.mark.asyncio
async def test_ingest_document_rejects_oversized_before_any_storage(real_vector_index, monkeypatch):
    monkeypatch.setattr(app_documents, "MAX_DOCUMENT_SIZE_BYTES", 10)
    oversized = b"x" * 11

    result = await app_documents.ingest_document(
        file_bytes=oversized,
        extension=".txt",
        display_name="big.txt",
        owner_user_id=1,
    )

    assert result.success is False
    assert result.rejected_reason == "oversized"
    assert result.file_size_bytes == 11
    assert not real_vector_index.uploads_dir.exists()


# ---------------------------------------------------------------------------
# C. Failure during indexing -> rollback, reported via the result (no raise)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_document_rolls_back_on_indexing_failure(real_vector_index, monkeypatch, caplog):
    monkeypatch.setattr(
        real_vector_index.vi, "reconcile_document",
        Mock(side_effect=RuntimeError("qdrant collection unavailable")),
    )

    import logging
    with caplog.at_level(logging.ERROR):
        result = await app_documents.ingest_document(
            file_bytes=b"some content",
            extension=".txt",
            display_name="notes.txt",
            owner_user_id=1,
        )

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert result.cleanup_complete is True
    assert list(real_vector_index.uploads_dir.iterdir()) == []  # orphan removed
    assert "qdrant collection unavailable" not in caplog.text  # privacy-safe logging preserved
    assert "Document upload failed" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_ingest_document_before_indexing_hook_exception_rolls_back_like_any_other_failure(real_vector_index):
    """A plain (non-cancellation) exception from the hook must be treated
    exactly like any other indexing-region failure: rolled back and
    reported as success=False, matching the original inline
    bot.send_message() call's behavior when it failed with a regular
    exception."""
    async def failing_hook():
        raise ValueError("telegram send failed")

    result = await app_documents.ingest_document(
        file_bytes=b"content that should be rolled back",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=1,
        before_indexing=failing_hook,
    )

    assert result.success is False
    assert result.error_type == "ValueError"
    assert list(real_vector_index.uploads_dir.iterdir()) == []
    assert real_vector_index.vi.get_stats(requesting_user_id=1)["total_documents"] == 0


# ---------------------------------------------------------------------------
# D. Cancellation safety, exercised directly against ingest_document()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancellation_during_before_indexing_hook_leaves_no_orphan(real_vector_index):
    """Reproduces the original Stage 2B-C Blocker 1 scenario directly
    against the new boundary: cancellation landing on the hook await
    (standing in for the original inline status-message send) must still
    resolve via cleanup before the CancelledError propagates."""
    entered_hook = asyncio.Event()

    async def blocking_hook():
        entered_hook.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(app_documents.ingest_document(
        file_bytes=b"cancel during hook",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=9,
        before_indexing=blocking_hook,
    ))

    await asyncio.wait_for(entered_hook.wait(), timeout=5)
    assert list(real_vector_index.uploads_dir.iterdir())  # durable before cancellation

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert list(real_vector_index.uploads_dir.iterdir()) == []
    assert real_vector_index.vi.get_stats(requesting_user_id=9)["total_documents"] == 0


@pytest.mark.asyncio
async def test_cancellation_while_indexing_worker_runs_leaves_no_orphan(real_vector_index, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def fake_load_and_index(stored, display_name):
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("never reached")

    monkeypatch.setattr(app_documents, "_load_and_index_document", fake_load_and_index)

    task = asyncio.create_task(app_documents.ingest_document(
        file_bytes=b"cancel during worker",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=9,
    ))

    for _ in range(500):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "worker was abandoned instead of awaited to completion"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert list(real_vector_index.uploads_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_cancellation_after_indexing_already_committed_retains_the_upload(real_vector_index, monkeypatch):
    """If the worker finishes successfully before the cancellation is
    observed, the already-committed upload must be retained — never
    deleted just because the caller was cancelled."""
    real_load_and_index = app_documents._load_and_index_document
    committed = threading.Event()

    def slow_then_real_load_and_index(stored, display_name):
        import time
        time.sleep(0.05)
        result = real_load_and_index(stored, display_name)
        committed.set()
        return result

    monkeypatch.setattr(app_documents, "_load_and_index_document", slow_then_real_load_and_index)
    task = asyncio.create_task(app_documents.ingest_document(
        file_bytes=b"survives cancellation after commit",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=9,
    ))
    await asyncio.sleep(0.02)  # let storage complete, worker start
    task.cancel()
    for _ in range(500):
        if committed.is_set():
            break
        await asyncio.sleep(0.01)
    assert committed.is_set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert real_vector_index.vi.get_stats(requesting_user_id=9)["total_documents"] == 1
    physical_files = [p for p in real_vector_index.uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    assert len(physical_files) == 1


# ---------------------------------------------------------------------------
# E. Ownership propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_owner_user_id_propagates_into_stored_upload_and_qdrant_scope(real_vector_index):
    result = await app_documents.ingest_document(
        file_bytes=b"owned by user 123 only",
        extension=".txt",
        display_name="private.txt",
        owner_user_id=123,
    )
    assert result.stored.owner_user_id == 123

    own_results = real_vector_index.vi.similarity_search("owned by user 123 only", requesting_user_id=123, k=1)
    other_results = real_vector_index.vi.similarity_search("owned by user 123 only", requesting_user_id=456, k=1)
    assert len(own_results) == 1
    assert len(other_results) == 0  # cross-user isolation preserved through the new boundary


# ---------------------------------------------------------------------------
# F. Adapter thinness — the Telegram handler delegates, no duplicate transaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_handler_delegates_to_ingest_document(monkeypatch):
    """handlers/document_upload.py must call app.documents.ingest_document()
    rather than reimplementing storage/indexing itself. Proven by
    replacing ingest_document() wholesale with a spy and checking the
    handler both calls it with the right arguments and faithfully
    forwards its result, with no independent storage/indexing logic of
    its own in between."""
    import handlers.document_upload as document_upload

    fake_result = app_documents.DocumentIngestResult(
        success=True, chunk_count=7, file_size_bytes=123,
        stored=app_documents.StoredUpload(
            physical_path=SimpleNamespace(), sidecar_path=SimpleNamespace(),
            document_id="upload:fake", content_sha256="deadbeef", owner_user_id=1,
        ),
    )
    ingest_mock = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(document_upload.document_pipeline, "ingest_document", ingest_mock)
    monkeypatch.setattr(document_upload.bot, "get_file", AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")))
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"hello world"))
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    document = SimpleNamespace(file_name="notes.txt", mime_type="text/plain", file_id="fid", file_size=11)
    message = SimpleNamespace(from_user=SimpleNamespace(id=1), chat=SimpleNamespace(id=1), document=document)

    await document_upload.process_document_upload(message, document)

    ingest_mock.assert_awaited_once()
    call_kwargs = ingest_mock.await_args.kwargs
    assert call_kwargs["file_bytes"] == b"hello world"
    assert call_kwargs["extension"] == ".txt"
    assert call_kwargs["display_name"] == "notes.txt"
    assert call_kwargs["owner_user_id"] == 1
    assert callable(call_kwargs["before_indexing"])

    # The handler reports success using ingest_document()'s own returned
    # values, not anything it computed/tracked itself.
    final_message = send_message_mock.await_args.args[1]
    assert "7" in final_message  # chunk_count forwarded verbatim
