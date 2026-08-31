"""
Stage 2B-C Blocker 1 regression tests: cancellation-boundary lifecycle for
a BRAND-NEW document upload, after the independent Codex REJECTED verdict
on Stage 2B-B.

Codex proved a real orphaning path: once the uploaded source file + its
durable `.meta.json` sidecar are written, `process_document_upload()`
awaited a cosmetic Telegram status message ("Индексирую документ…") with
NO `except asyncio.CancelledError:` boundary covering that specific await
— cancellation landing there propagated `asyncio.CancelledError`
(a `BaseException`, never caught by the function's own `except Exception:`)
straight out of the coroutine, uncaught, leaving the durable file+sidecar
on disk forever with no cleanup and no indexing.

The fix (see handlers/document_upload.py's `_resolve_cancelled_after_storage()`)
wraps EVERY await between durable storage and a successfully observed
indexing result in one protected region. This module proves the required
invariant end to end:

    For a BRAND-NEW upload, after durable source+sidecar creation, exactly
    one stable final outcome is allowed:
      SUCCESS: source remains; sidecar remains; intended Qdrant document exists.
      FAILURE/CANCELLATION BEFORE SUCCESS: source removed; sidecar removed;
      no Qdrant document remains.

Every test here uses a REAL local-persistent Qdrant VectorIndex (tmp_path,
deterministic local fake embeddings — tests/rag_fakes.py) swapped in by
monkeypatching handlers.document_upload's `get_vector_index` to return it
(Stage 2B-D Blocker 4 replaced the old eager module-level `vector_index`
singleton with this lazy accessor), so cleanup is verified against actual
indexed state, not merely mock call assertions (the audit's requirement
8). All monkeypatching goes through pytest's `monkeypatch` fixture
(auto-reverted) since `document_loader`/the shared VectorIndex singleton
are shared resources other test modules also depend on. No network, no
real Telegram/OpenAI/Qdrant.
"""

import asyncio
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.documents as app_documents
import db.identity as db_identity
from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(interval)


def _make_message(user_id: int, file_name: str, file_id: str = "fid"):
    document = SimpleNamespace(file_name=file_name, mime_type="text/plain", file_id=file_id, file_size=100)
    message = SimpleNamespace(from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=user_id), document=document)
    return message, document


def _requesting_uuid_for_telegram_id(telegram_id: int) -> str:
    """These tests drive uploads through the real Telegram handler
    (document_upload.process_document_upload), which resolves ownership
    via app.identity.resolve_user_uuid() -> db.identity's (fixture-faked,
    see conftest.py's _default_fake_preferences) resolver — stable per
    telegram_id within a test. Retrieving that same UUID here lets these
    tests query VectorIndex as "the uploader themselves"."""
    return str(db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id))


@pytest.fixture
def real_upload_env(monkeypatch, tmp_path):
    """A brand-new-upload environment backed by REAL local-persistent
    Qdrant (not mocked) so tests can assert on genuine indexed state."""
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="upload_lifecycle_test",
    )
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)

    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"Real content for the upload lifecycle test."))
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    yield SimpleNamespace(document_upload=document_upload, vi=vi, uploads_dir=uploads_dir, send_message_mock=send_message_mock)
    vi.close()


# ---------------------------------------------------------------------------
# 1/2/3: cancel immediately after durable storage / during the "Индексирую
# документ…" send / before the indexing worker ever starts — all the SAME
# previously-uncovered boundary (there is no other await between durable
# storage and worker submission).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_during_status_message_after_storage_leaves_no_orphan(real_upload_env, monkeypatch):
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi
    uploads_dir = real_upload_env.uploads_dir

    entered_status_send = asyncio.Event()
    real_send_message = document_upload.bot.send_message

    async def blocking_status_send(chat_id, text, *args, **kwargs):
        if "Индексирую" in text:
            entered_status_send.set()
            await asyncio.sleep(3600)  # cancelled long before this could elapse
        return await real_send_message(chat_id, text, *args, **kwargs)

    monkeypatch.setattr(document_upload.bot, "send_message", blocking_status_send)

    load_mock = Mock()
    add_mock = Mock()
    monkeypatch.setattr(app_documents.document_loader, "load_document", load_mock)
    monkeypatch.setattr(app_documents.get_vector_index(), "add_documents", add_mock)

    message, document = _make_message(42, "notes.txt")
    task = asyncio.create_task(document_upload.process_document_upload(message, document))

    await _wait_until(entered_status_send.is_set)

    # Storage already durable: source file + sidecar exist on disk.
    created_before_cancel = list(uploads_dir.iterdir())
    assert len(created_before_cancel) == 2

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No orphan: cleanup ran even though cancellation landed on the
    # cosmetic status message, before the indexing worker ever started —
    # this is the exact gap Codex proved.
    assert list(uploads_dir.iterdir()) == []
    load_mock.assert_not_called()
    add_mock.assert_not_called()
    assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(42))["total_documents"] == 0


# ---------------------------------------------------------------------------
# 4: cancel while the indexing worker is genuinely running
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_while_indexing_worker_runs_leaves_no_orphan(real_upload_env, monkeypatch):
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi
    uploads_dir = real_upload_env.uploads_dir

    started = threading.Event()
    release = threading.Event()

    def fake_load_and_index(stored, display_name):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        raise RuntimeError("never reached in this test path")

    monkeypatch.setattr(app_documents, "_load_and_index_document", fake_load_and_index)

    message, document = _make_message(42, "notes.txt")
    task = asyncio.create_task(document_upload.process_document_upload(message, document))

    await _wait_until(started.is_set)
    task.cancel()

    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "handler finished before the indexing worker did — it was abandoned"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert list(uploads_dir.iterdir()) == []
    assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(42))["total_documents"] == 0


# ---------------------------------------------------------------------------
# 5: repeated cancellation while the indexing worker is still running
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repeated_cancellation_while_indexing_worker_runs_leaves_no_orphan(real_upload_env, monkeypatch):
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi
    uploads_dir = real_upload_env.uploads_dir

    started = threading.Event()
    release = threading.Event()

    def fake_load_and_index(stored, display_name):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        raise RuntimeError("boom")

    monkeypatch.setattr(app_documents, "_load_and_index_document", fake_load_and_index)

    message, document = _make_message(42, "notes.txt")
    task = asyncio.create_task(document_upload.process_document_upload(message, document))

    await _wait_until(started.is_set)

    for _ in range(5):
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert list(uploads_dir.iterdir()) == []
    assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(42))["total_documents"] == 0


# ---------------------------------------------------------------------------
# 6: cancel after the worker ALREADY succeeded, but before the outer task
# observed that — successfully indexed data must never be deleted.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_after_worker_succeeds_before_observed_retains_everything(real_upload_env, monkeypatch):
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi
    uploads_dir = real_upload_env.uploads_dir

    started = threading.Event()
    release = threading.Event()
    real_load_and_index = app_documents._load_and_index_document

    def fake_load_and_index(stored, display_name):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        return real_load_and_index(stored, display_name)

    monkeypatch.setattr(app_documents, "_load_and_index_document", fake_load_and_index)

    message, document = _make_message(42, "notes.txt")
    task = asyncio.create_task(document_upload.process_document_upload(message, document))

    await _wait_until(started.is_set)

    # Cancel FIRST (caller cancelled), THEN let the worker actually finish
    # successfully — the classic "worker completed between cancellation
    # and observation" race await_worker()'s shield-loop must resolve
    # without ever deleting the successfully-committed data.
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    physical_files = [p for p in uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    sidecar_files = [p for p in uploads_dir.iterdir() if p.name.endswith(".meta.json")]
    assert len(physical_files) == 1
    assert len(sidecar_files) == 1
    assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(42))["total_documents"] >= 1  # successfully committed, never deleted

    sent_texts = [c.args[1] for c in real_upload_env.send_message_mock.await_args_list]
    assert not any("успешно загружен" in t for t in sent_texts)  # cancelled: no success notification


# ---------------------------------------------------------------------------
# 7: ordinary (non-cancelled) indexing failure — same cleanup guarantee
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ordinary_indexing_failure_leaves_no_orphan(real_upload_env, monkeypatch):
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi
    uploads_dir = real_upload_env.uploads_dir

    monkeypatch.setattr(app_documents.document_loader, "load_document_bytes", Mock(side_effect=ValueError("simulated parse failure")))

    message, document = _make_message(42, "notes.txt")
    await document_upload.process_document_upload(message, document)

    assert list(uploads_dir.iterdir()) == []
    assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(42))["total_documents"] == 0

    sent_texts = [c.args[1] for c in real_upload_env.send_message_mock.await_args_list]
    assert any("ошибка" in t.lower() for t in sent_texts)


# ---------------------------------------------------------------------------
# 9: successful upload preserves all three artifacts — source, sidecar,
# and the intended Qdrant document (proven via a real similarity search,
# not merely a mock call).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_upload_preserves_source_sidecar_and_qdrant_document(real_upload_env):
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi
    uploads_dir = real_upload_env.uploads_dir

    message, document = _make_message(42, "notes.txt")
    await document_upload.process_document_upload(message, document)

    physical_files = [p for p in uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    sidecar_files = [p for p in uploads_dir.iterdir() if p.name.endswith(".meta.json")]
    assert len(physical_files) == 1
    assert len(sidecar_files) == 1

    results = vi.similarity_search("Real content for the upload lifecycle test.", requesting_user_uuid=_requesting_uuid_for_telegram_id(42), k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "notes.txt"

    sent_texts = [c.args[1] for c in real_upload_env.send_message_mock.await_args_list]
    assert any("успешно загружен" in t for t in sent_texts)


# ---------------------------------------------------------------------------
# 10: cleanup keys by document_id, never by display filename — two uploads
# sharing the SAME display name must be independently addressable.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_targets_document_id_not_display_filename(real_upload_env, monkeypatch):
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi
    uploads_dir = real_upload_env.uploads_dir

    # First upload succeeds normally.
    message1, document1 = _make_message(42, "shared_name.txt")
    await document_upload.process_document_upload(message1, document1)
    assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(42))["total_documents"] >= 1
    first_physical = [p for p in uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    assert len(first_physical) == 1

    # Second upload, SAME display filename, fails during indexing.
    monkeypatch.setattr(
        app_documents.document_loader, "load_document_bytes",
        Mock(side_effect=ValueError("simulated parse failure for the second upload")),
    )
    message2, document2 = _make_message(42, "shared_name.txt", file_id="fid2")
    await document_upload.process_document_upload(message2, document2)

    # The first upload's physical file, sidecar, and Qdrant points must
    # all still be present — cleanup only ever touched the SECOND upload's
    # own document_id, never anything keyed merely by the shared filename.
    remaining_physical = [p for p in uploads_dir.iterdir() if not p.name.endswith(".meta.json")]
    remaining_sidecars = [p for p in uploads_dir.iterdir() if p.name.endswith(".meta.json")]
    assert remaining_physical == first_physical
    assert len(remaining_sidecars) == 1

    results = vi.similarity_search("Real content for the upload lifecycle test.", requesting_user_uuid=_requesting_uuid_for_telegram_id(42), k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "shared_name.txt"


# ---------------------------------------------------------------------------
# Stage 2B-C Section I / Stage 2B-F: hash/read consistency for managed
# uploads. The ORIGINAL Stage 2B-C mechanism this test proved (re-hash
# `stored.physical_path` both before and immediately after
# document_loader.load_document() independently reopened that SAME
# pathname) is exactly the "hash one object / parse another" TOCTOU shape
# Stage 2B-F closed: `_load_and_index_document()` now reads
# `stored.physical_path` exactly ONCE via
# rag.safe_files.read_regular_file_secure() and reconciles Qdrant from
# those captured bytes (VectorIndex.reconcile_document(...,
# source_bytes=...)). Stage 2B-F Blocker 1 (a later audit finding against
# the FIRST fix above) went one step further: reconcile_document() used to
# hand those bytes to document_loader.load_document() via a private
# temporary snapshot FILE — still a pathname the loader's own parser would
# reopen, just moved one level down. document_loader.load_document_bytes()
# now parses those bytes directly in memory (no snapshot, no pathname of
# any kind) — see tests/test_stage2f_upload_secure_read.py for the
# dedicated TOCTOU-swap/hash-mismatch/no-reopen regressions. Mutating the
# loader's OWN argument no longer means anything for a managed upload — it
# now receives copied in-memory bytes, not any shared pathname — so this
# test proves the property the mechanism actually provides: mutating the
# REAL managed source after its bytes were already securely captured must
# have ZERO effect on what gets indexed.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_managed_source_mutated_after_secure_read_does_not_reach_qdrant(real_upload_env, monkeypatch):
    """
    Stage 5C corrective pass #4 (Blocker 5): a source rewrite landing right
    after the FIRST secure read (the one whose bytes get indexed) is no
    longer merely "harmless because the old bytes were already captured" —
    that used to leave Qdrant/sidecar/catalog self-consistently describing
    a snapshot the DURABLE FILE ON DISK no longer matches, exactly the
    disk/sidecar/catalog disagreement Principle 1 forbids reporting as
    success. `_load_and_index_document()`'s final pre-activation
    revalidation (a SECOND secure read, immediately before
    mark_active_sync()) now re-detects this same mutation and fails the
    whole ingestion closed instead: no active document, no orphaned
    Qdrant content, a truthful failure notification to the user.
    """
    document_upload = real_upload_env.document_upload
    vi = real_upload_env.vi

    real_secure_read = app_documents.read_regular_file_secure

    def racing_secure_read(path, *, root):
        secure_bytes = real_secure_read(path, root=root)
        # Simulate a controlled race: the managed source is rewritten
        # immediately AFTER its bytes were already securely captured —
        # the narrower lstat-to-open race is proven closed separately
        # (test_stage2f_upload_secure_read.py). This module-level
        # monkeypatch affects BOTH the initial secure read (whose bytes
        # get indexed) AND the final pre-activation revalidation read
        # added by Blocker 5 — the second call observes the mutation this
        # first call just made and fails ingestion closed.
        Path(path).write_bytes(b"MUTATED CONTENT - must never be indexed under the old hash")
        return secure_bytes

    monkeypatch.setattr(app_documents, "read_regular_file_secure", racing_secure_read)

    message, document = _make_message(42, "notes.txt")
    await document_upload.process_document_upload(message, document)

    # Ingestion failed closed — the mutated content never reached Qdrant,
    # but neither did the original snapshot: no document became active
    # while the durable file on disk disagreed with what was indexed.
    assert vi.get_stats(requesting_user_uuid=_requesting_uuid_for_telegram_id(42))["total_documents"] == 0
    results = vi.similarity_search("Real content for the upload lifecycle test.", requesting_user_uuid=_requesting_uuid_for_telegram_id(42), k=5)
    assert all("MUTATED CONTENT" not in r.page_content for r in results)

    sent_texts = [c.args[1] for c in real_upload_env.send_message_mock.await_args_list]
    assert not any("успешно загружен" in t for t in sent_texts)
    assert any("ошибка" in t.lower() for t in sent_texts)


# ---------------------------------------------------------------------------
# Stage 2B-C Section J: cleanup failure visibility — `_cleanup_new_upload()`
# must not silently claim success when the (best-effort) Qdrant cleanup
# failed. Filesystem cleanup still completes; the caller can tell the two
# outcomes apart via the return value; cancellation semantics (the
# original CancelledError still propagates) are preserved.
# ---------------------------------------------------------------------------

def test_cleanup_new_upload_returns_false_when_qdrant_cleanup_fails(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    physical = tmp_path / "abc.txt"
    physical.write_bytes(b"content")
    sidecar = tmp_path / "abc.meta.json"
    sidecar.write_text("{}", encoding="utf-8")
    stored = app_documents.StoredUpload(
        physical_path=physical, sidecar_path=sidecar,
        document_id="upload:" + "a" * 32, document_uuid=uuid.UUID("a" * 32), content_sha256="b" * 64, owner_user_id=uuid.uuid4(),
    )

    monkeypatch.setattr(
        app_documents.get_vector_index(), "delete_document",
        Mock(side_effect=RuntimeError("simulated Qdrant cleanup failure")),
    )

    result = app_documents._cleanup_new_upload(stored)

    assert result is False
    # Filesystem cleanup still completed regardless of the Qdrant failure.
    assert not physical.exists()
    assert not sidecar.exists()


def test_cleanup_new_upload_returns_true_on_full_success(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    physical = tmp_path / "abc.txt"
    physical.write_bytes(b"content")
    sidecar = tmp_path / "abc.meta.json"
    sidecar.write_text("{}", encoding="utf-8")
    stored = app_documents.StoredUpload(
        physical_path=physical, sidecar_path=sidecar,
        document_id="upload:" + "a" * 32, document_uuid=uuid.UUID("a" * 32), content_sha256="b" * 64, owner_user_id=uuid.uuid4(),
    )

    monkeypatch.setattr(app_documents.get_vector_index(), "delete_document", Mock())

    assert app_documents._cleanup_new_upload(stored) is True
    assert not physical.exists()
    assert not sidecar.exists()


@pytest.mark.asyncio
async def test_cancellation_still_propagates_when_qdrant_cleanup_also_fails(real_upload_env, monkeypatch, caplog):
    """A Qdrant cleanup failure during cancellation resolution must be
    recorded (logged) but must NEVER replace or swallow the original
    CancelledError — the caller-visible outcome stays cancellation."""
    import logging

    document_upload = real_upload_env.document_upload

    started = threading.Event()
    release = threading.Event()

    def fake_load_and_index(stored, display_name):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        raise RuntimeError("boom")

    monkeypatch.setattr(app_documents, "_load_and_index_document", fake_load_and_index)
    monkeypatch.setattr(
        app_documents.get_vector_index(), "delete_document",
        Mock(side_effect=RuntimeError("simulated Qdrant cleanup failure")),
    )

    message, document = _make_message(42, "notes.txt")
    task = asyncio.create_task(document_upload.process_document_upload(message, document))

    await _wait_until(started.is_set)
    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done()
    release.set()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "cleanup incomplete" in caplog.text.lower()
