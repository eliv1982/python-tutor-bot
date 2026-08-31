"""
Stage 5B regression tests: the document ingestion transaction
(app.documents.ingest_document()) as an adapter-independent application
boundary, callable directly with no Telegram message/document objects.
Ownership identity migrated from Telegram int to canonical internal UUID
(Stage 5C) — every owner_user_id value below is a uuid.UUID, not an int.

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
- handlers/document_upload.py delegates to it (having already resolved
  canonical identity via app.identity.resolve_user_uuid()) rather than
  owning a second copy of the transaction.

All Qdrant access uses the existing deterministic local-persistent
VectorIndex pattern (tests/rag_fakes.py) — no real OpenAI/Qdrant network
calls anywhere in this module.

`_uid(n)` derives a stable, distinct canonical UUID string from a small
int (uuid5 off a fixed namespace) — preserves this module's original
relational structure (same n -> same identity, different n -> different
identity) from before the Telegram-int-owner -> canonical-UUID-owner
migration.
"""

import asyncio
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.documents as app_documents
from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings

_TEST_NAMESPACE = uuid.uuid4()


def _uid(n: int) -> uuid.UUID:
    return uuid.uuid5(_TEST_NAMESPACE, str(n))


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
    """ingest_document() takes only bytes/str/UUID primitives — no
    telebot.types.Message/Document, no bot instance."""
    owner = _uid(42)
    result = await app_documents.ingest_document(
        file_bytes=b"Python functions are defined with the def keyword.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner,
    )

    assert result.success is True
    assert result.chunk_count == 1
    assert result.stored is not None
    assert result.stored.owner_user_id == owner
    assert result.file_size_bytes == len(b"Python functions are defined with the def keyword.")

    physical_files = [p for p in real_vector_index.uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    assert len(physical_files) == 1

    results = real_vector_index.vi.similarity_search(
        "Python functions are defined with the def keyword.", requesting_user_uuid=str(owner), k=1
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
        owner_user_id=_uid(1),
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
        owner_user_id=_uid(1),
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
        owner_user_id=_uid(1),
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
            owner_user_id=_uid(1),
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
    owner = _uid(1)

    async def failing_hook():
        raise ValueError("telegram send failed")

    result = await app_documents.ingest_document(
        file_bytes=b"content that should be rolled back",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner,
        before_indexing=failing_hook,
    )

    assert result.success is False
    assert result.error_type == "ValueError"
    assert list(real_vector_index.uploads_dir.iterdir()) == []
    assert real_vector_index.vi.get_stats(requesting_user_uuid=str(owner))["total_documents"] == 0


# ---------------------------------------------------------------------------
# D. Cancellation safety, exercised directly against ingest_document()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancellation_during_before_indexing_hook_leaves_no_orphan(real_vector_index):
    """Reproduces the original Stage 2B-C Blocker 1 scenario directly
    against the new boundary: cancellation landing on the hook await
    (standing in for the original inline status-message send) must still
    resolve via cleanup before the CancelledError propagates."""
    owner = _uid(9)
    entered_hook = asyncio.Event()

    async def blocking_hook():
        entered_hook.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(app_documents.ingest_document(
        file_bytes=b"cancel during hook",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner,
        before_indexing=blocking_hook,
    ))

    await asyncio.wait_for(entered_hook.wait(), timeout=5)
    assert list(real_vector_index.uploads_dir.iterdir())  # durable before cancellation

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert list(real_vector_index.uploads_dir.iterdir()) == []
    assert real_vector_index.vi.get_stats(requesting_user_uuid=str(owner))["total_documents"] == 0


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
        owner_user_id=_uid(9),
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
    """If the durable commit (Qdrant reconciliation + catalog activation)
    has ALREADY genuinely happened before cancellation is even issued, the
    already-committed upload must be retained — never deleted just because
    the caller was cancelled afterward.

    Stage 5C corrective pass #2, Section 6 (second independent audit
    finding): the prior version of this test synchronized on a `started`
    Event set at the very BEGINNING of the indexing worker, then called
    `task.cancel()`, then separately waited for a `committed` Event set
    only after the worker's call returned. That proves the worker
    eventually reaches commit DESPITE an earlier cancellation (a real, but
    different, invariant already covered by
    `test_cancellation_while_indexing_worker_runs_leaves_no_orphan` above)
    — it does NOT prove cancellation landing AFTER commit is handled
    correctly, because cancellation was issued at `started`, strictly
    BEFORE any Qdrant/catalog mutation had occurred.

    This version synchronizes on the actual commit point instead:
    `db.documents.mark_active_sync()` — the exact call that flips the
    PostgreSQL catalog row to 'active', i.e. the durable transition this
    test's name claims to test — is wrapped so it blocks (via `resume`)
    immediately AFTER the real mark_active_sync() has already returned
    (i.e. after the commit has already happened) and signals `committed`
    the moment that return happens. The test waits for `committed` BEFORE
    ever calling `task.cancel()`, and only releases the worker to finish
    (the remaining verification read + return) AFTER `task.cancel()` has
    already been issued. This ordering is airtight: cancellation is
    proven to land strictly after the commit, never merely "eventually
    converges to committed despite an earlier cancel"."""
    import db.documents as db_documents_module

    owner = _uid(9)
    committed = threading.Event()
    resume = threading.Event()

    real_mark_active = db_documents_module.mark_active_sync

    def blocking_after_commit_mark_active(*, document_id):
        real_mark_active(document_id=document_id)
        # The durable commit has ALREADY happened by this point — signal
        # it, then block until the test has issued cancellation.
        committed.set()
        assert resume.wait(timeout=5), "resume was never set by the test"

    monkeypatch.setattr(app_documents.db_documents, "mark_active_sync", blocking_after_commit_mark_active)
    task = asyncio.create_task(app_documents.ingest_document(
        file_bytes=b"survives cancellation after commit",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner,
    ))

    for _ in range(500):
        if committed.is_set():
            break
        await asyncio.sleep(0.01)
    assert committed.is_set(), "indexing never reached the commit point (mark_active_sync)"

    # The commit has already genuinely happened — cancellation issued here
    # is unambiguously AFTER it, never before.
    task.cancel()
    resume.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # No compensating rollback occurred: the committed upload is retained.
    assert real_vector_index.vi.get_stats(requesting_user_uuid=str(owner))["total_documents"] == 1
    physical_files = [p for p in real_vector_index.uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    assert len(physical_files) == 1
    from db.documents import ACTIVE_STATUSES
    document_uuid = uuid.UUID(physical_files[0].stem)
    record = db_documents_module.get_sync(document_id=document_uuid)
    assert record is not None and record.status in ACTIVE_STATUSES


# ---------------------------------------------------------------------------
# E. Ownership propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_owner_user_id_propagates_into_stored_upload_and_qdrant_scope(real_vector_index):
    owner_123, owner_456 = _uid(123), _uid(456)
    result = await app_documents.ingest_document(
        file_bytes=b"owned by user 123 only",
        extension=".txt",
        display_name="private.txt",
        owner_user_id=owner_123,
    )
    assert result.stored.owner_user_id == owner_123

    own_results = real_vector_index.vi.similarity_search("owned by user 123 only", requesting_user_uuid=str(owner_123), k=1)
    other_results = real_vector_index.vi.similarity_search("owned by user 123 only", requesting_user_uuid=str(owner_456), k=1)
    assert len(own_results) == 1
    assert len(other_results) == 0  # cross-user isolation preserved through the new boundary


# ---------------------------------------------------------------------------
# F. Adapter thinness — the Telegram handler delegates, no duplicate transaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_handler_delegates_to_ingest_document(monkeypatch):
    """handlers/document_upload.py must resolve canonical identity (via
    app.identity.resolve_user_uuid(), fixture-faked in conftest.py — see
    its own docstring) and call app.documents.ingest_document() rather
    than reimplementing storage/indexing itself. Proven by replacing
    ingest_document() wholesale with a spy and checking the handler both
    calls it with the right (already-resolved-to-UUID) arguments and
    faithfully forwards its result, with no independent storage/indexing
    logic of its own in between."""
    import db.identity as db_identity
    import handlers.document_upload as document_upload

    fake_result = app_documents.DocumentIngestResult(
        success=True, chunk_count=7, file_size_bytes=123,
        stored=app_documents.StoredUpload(
            physical_path=SimpleNamespace(), sidecar_path=SimpleNamespace(),
            document_id="upload:fake", document_uuid=uuid.uuid4(), content_sha256="deadbeef", owner_user_id=uuid.uuid4(),
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
    # The handler's own resolve_user_uuid(1) call and this assertion's own
    # resolution both hit the SAME fixture-faked (stable-per-telegram-id)
    # resolver — see conftest.py's _default_fake_preferences.
    assert call_kwargs["owner_user_id"] == db_identity.resolve_or_create_user_by_telegram_id_sync(1)
    assert callable(call_kwargs["before_indexing"])

    # The handler reports success using ingest_document()'s own returned
    # values, not anything it computed/tracked itself.
    final_message = send_message_mock.await_args.args[1]
    assert "7" in final_message  # chunk_count forwarded verbatim


# ---------------------------------------------------------------------------
# G. Stage 5C corrective pass #2, Section 3: final ingestion verification
# must cover display_name (not just status/owner/stored_name/hash)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_display_name_mismatch_in_catalog_fails_ingestion_and_cleans_up(real_vector_index, monkeypatch):
    """Reproduces the audit finding directly: the final catalog
    verification in _load_and_index_document() must fail closed if the
    PostgreSQL row's display_name disagrees with the document actually
    being ingested — even when status/owner/stored_name/content_sha256 all
    otherwise agree. A dedicated monkeypatch simulates the row disagreeing
    (standing in for any real-world way this field could diverge —
    concurrent mutation, a bug elsewhere) without needing a second writer."""
    import dataclasses
    owner = _uid(77)

    real_get_sync = app_documents.db_documents.get_sync

    def tampering_get_sync(*, document_id):
        record = real_get_sync(document_id=document_id)
        if record is None:
            return None
        # Every other field matches reality — ONLY display_name disagrees.
        return dataclasses.replace(record, display_name="a-completely-different-name.txt")

    monkeypatch.setattr(app_documents.db_documents, "get_sync", tampering_get_sync)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose catalog display_name will be tampered with",
        extension=".txt",
        display_name="original-name.txt",
        owner_user_id=owner,
    )

    assert result.success is False
    assert result.error_type == "CatalogConsistencyError"
    assert result.cleanup_complete is True
    # Full rollback: no orphaned physical file, sidecar, or Qdrant points.
    assert list(real_vector_index.uploads_dir.iterdir()) == []
    assert real_vector_index.vi.get_stats(requesting_user_uuid=str(owner))["total_documents"] == 0


@pytest.mark.asyncio
async def test_matching_display_name_still_succeeds(real_vector_index):
    """Counterpart proof: an ordinary, untampered ingestion (display_name
    genuinely consistent end to end) is NOT affected by the new check."""
    owner = _uid(78)

    result = await app_documents.ingest_document(
        file_bytes=b"ordinary content with a consistent display name",
        extension=".txt",
        display_name="consistent-name.txt",
        owner_user_id=owner,
    )

    assert result.success is True
    assert result.stored is not None


# ---------------------------------------------------------------------------
# H. Stage 5C corrective pass #2, Section 4: partial-storage cleanup
# reporting must never claim cleanup_complete=True when it wasn't
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_storage_cleanup_failure_is_reported_as_incomplete(real_vector_index, monkeypatch):
    """Reproduces the exact window the audit identified: a physical file
    IS created by _store_document_exclusively() (the exclusive open()
    genuinely succeeds), a LATER step in that same function fails (here:
    the durable catalog insert), and the internal best-effort cleanup of
    that physical file (and its sidecar) is then FORCED to fail too. The
    outer ingest_document() layer never gets an owned StoredUpload in this
    scenario (the storage worker raised before ever returning one) — it
    must not default to cleanup_complete=True regardless; it must report
    the real, false, outcome, and the artifact must genuinely still be on
    disk afterward (proving this isn't a false-negative from the test's
    own bookkeeping)."""
    owner = _uid(79)

    def failing_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        raise RuntimeError("simulated catalog insert failure")

    def failing_cleanup_file(filepath):
        # Simulates cleanup_file() itself failing (e.g. a permission
        # error mid-unlink) — deliberately never removes anything, so the
        # artifact genuinely remains on disk exactly like a real failed
        # unlink would leave it, rather than merely returning False while
        # secretly still deleting the file.
        return False

    monkeypatch.setattr(app_documents.db_documents, "create_pending_sync", failing_create_pending)
    monkeypatch.setattr(app_documents, "cleanup_file", failing_cleanup_file)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose storage will partially fail",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner,
    )

    assert result.success is False
    assert result.cleanup_complete is False, "cleanup genuinely failed and must never be reported as complete"
    # The physical artifact (and its sidecar) genuinely remain on disk —
    # proving cleanup_complete=False reflects real, observable state.
    remaining = list(real_vector_index.uploads_dir.iterdir())
    assert len(remaining) >= 1


@pytest.mark.asyncio
async def test_partial_storage_cleanup_success_is_reported_as_complete(real_vector_index, monkeypatch):
    """Counterpart proof: the SAME failure (catalog insert failing after
    the physical file was created) but with cleanup succeeding normally
    must still report cleanup_complete=True, exactly as before this
    corrective pass — the fix must not make every storage failure look
    incomplete."""
    owner = _uid(80)

    def failing_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        raise RuntimeError("simulated catalog insert failure")

    monkeypatch.setattr(app_documents.db_documents, "create_pending_sync", failing_create_pending)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose storage fails but cleans up successfully",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner,
    )

    assert result.success is False
    assert result.cleanup_complete is True
    assert list(real_vector_index.uploads_dir.iterdir()) == []
