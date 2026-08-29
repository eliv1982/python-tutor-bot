"""
Stage 1E.1/1E.2 regression tests: cancellation-safe worker lifecycle after
two independent Codex audits of Stage 1E's blocking-work offload.

Stage 1E moved several blocking operations onto worker threads. Cancelling
the AWAITING coroutine does not stop the worker thread — Python cannot
interrupt a running thread — so a naive `await asyncio.to_thread(...)`
creates cancellation states that never existed before Stage 1E: a worker
can still be mid-write/mid-parse/mid-ffmpeg when the caller has already
moved on, racing whatever the caller does next (cleanup, reuse of the same
path, etc).

Stage 1E.2 replaced `asyncio.create_task(asyncio.to_thread(...))` +
`await_shielded()` with `utils/helpers.py:submit_worker()` (a bare
`loop.run_in_executor()` Future — never an `asyncio.Task`) +
`await_worker()` (survives repeated/shutdown-style cancellation without
ever calling `.cancel()` on that Future). A cancelled Task wrapping
`to_thread(...)` can reach a terminal ("cancelled") state immediately even
while the underlying executor thread keeps running — that assumption was
the root defect Stage 1E.1 had not fully eliminated. See
`utils/helpers.py` for the full mechanism explanation. On cancellation
(including repeated cancellation), the worker is never abandoned — the
resolver waits for it to reach a genuine terminal state before deciding
cleanup vs. retention, then the ORIGINAL `asyncio.CancelledError` is
re-raised.

Every test here uses `threading.Event`-based synchronization to force a
cancellation to land while a worker is deterministically still running —
never an arbitrary sleep as the actual correctness assertion. All external
calls (Telegram, OpenAI, Chroma/embeddings, ffmpeg/pydub) are mocked. No
network, no live ffmpeg process, and no mutation of the real
data/documents, data/documents/uploads, data/chroma_db, or bot.log paths
(same tests/conftest.py isolation as every other test module).
"""

import asyncio
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    """Poll `predicate()` on the event loop until it's True or `timeout`
    elapses. Used only to detect a worker-thread Event being set — never as
    the correctness assertion itself (that's always a deterministic Event
    wait/check afterward)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# A. Cancellation during document storage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_during_storage_waits_for_worker_and_removes_only_owned_file(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload

    started = threading.Event()
    release = threading.Event()
    created_path_holder = {}

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    bystander = tmp_path / "bystander.txt"
    bystander.write_bytes(b"UNRELATED PRE-EXISTING CONTENT")

    def fake_store(file_bytes, extension, display_name, owner_user_id):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        path = tmp_path / f"owned_upload{extension}"
        path.write_bytes(file_bytes)
        created_path_holder["path"] = path
        return document_upload.StoredUpload(
            physical_path=path,
            sidecar_path=tmp_path / "owned_upload.meta.json",
            document_id="upload:test-fake",
            content_sha256="deadbeef",
            owner_user_id=owner_user_id,
        )

    load_mock = Mock()
    add_mock = Mock()
    monkeypatch.setattr(document_upload, "_store_document_exclusively", fake_store)
    monkeypatch.setattr(document_upload.document_loader, "load_document", load_mock)
    monkeypatch.setattr(document_upload.get_vector_index(), "add_documents", add_mock)

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"hello world"))
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=42),
        document=SimpleNamespace(file_name="notes.txt", mime_type="text/plain", file_id="fid", file_size=11),
    )

    task = asyncio.create_task(document_upload.process_document_upload(message, message.document))

    await _wait_until(started.is_set)

    task.cancel()

    # The worker is still blocked on `release` — the handler must NOT finish
    # before it does; if it did, the worker would have been abandoned.
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "handler finished before the storage worker did — it was abandoned"
    assert "path" not in created_path_holder  # worker genuinely hasn't returned yet

    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The newly-owned file was created, then removed — never abandoned as
    # an orphan.
    assert "path" in created_path_holder
    assert not created_path_holder["path"].exists()

    # Parser/indexer must never have started.
    load_mock.assert_not_called()
    add_mock.assert_not_called()

    # No bystander file touched.
    assert bystander.exists()
    assert bystander.read_bytes() == b"UNRELATED PRE-EXISTING CONTENT"
    assert list(tmp_path.iterdir()) == [bystander]

    # No Telegram message sent while resolving the cancellation (only the
    # two progress messages already sent before cancellation happened).
    sent_texts = [c.args[1] for c in send_message_mock.await_args_list]
    assert not any("ошибка" in t.lower() for t in sent_texts)
    assert not any("успешно" in t.lower() for t in sent_texts)


# ---------------------------------------------------------------------------
# B. Cancellation during load/index — SUCCESS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_during_indexing_success_retains_file_and_sends_no_success_message(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload

    started = threading.Event()
    release = threading.Event()
    call_count = {"n": 0}

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)
    # Real storage step (fast, not cancelled) so a real owned file exists.

    def fake_load_and_index(stored, display_name):
        call_count["n"] += 1
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        return ["chunk-a", "chunk-b"]

    monkeypatch.setattr(document_upload, "_load_and_index_document", fake_load_and_index)

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"hello world"))
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=42),
        document=SimpleNamespace(file_name="notes.txt", mime_type="text/plain", file_id="fid", file_size=11),
    )

    task = asyncio.create_task(document_upload.process_document_upload(message, message.document))

    await _wait_until(started.is_set)

    # Real storage step (not cancelled) already committed a physical file
    # + its sidecar by the time indexing started.
    created_before_cancel = [p for p in tmp_path.iterdir() if p.is_file()]
    assert len(created_before_cancel) == 2
    physical_path = next(p for p in created_before_cancel if not p.name.endswith(".meta.json"))

    task.cancel()

    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "handler finished before the indexing worker did — it was abandoned"
        assert physical_path.exists(), "physical file removed while the indexing worker was still active"

    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Ingestion happened exactly once.
    assert call_count["n"] == 1
    # The physical file must be retained — never delete successfully
    # ingested data.
    assert physical_path.exists()

    # No success notification sent after cancellation (only the two
    # progress messages).
    sent_texts = [c.args[1] for c in send_message_mock.await_args_list]
    assert any("Индексирую" in t for t in sent_texts)
    assert not any("успешно загружен" in t for t in sent_texts)


# ---------------------------------------------------------------------------
# C. Cancellation during load/index — FAILURE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_during_indexing_failure_removes_file_only_after_worker_ends_and_stays_private(
    monkeypatch, tmp_path, caplog
):
    import handlers.document_upload as document_upload

    started = threading.Event()
    release = threading.Event()
    sensitive_detail = (
        "chroma write failed for C:\\Users\\confidential_employee\\secret_report.pdf: "
        "Authorization: Bearer sk-FAKE1234567890SECRET"
    )

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    def fake_load_and_index(stored, display_name):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        raise RuntimeError(sensitive_detail)

    monkeypatch.setattr(document_upload, "_load_and_index_document", fake_load_and_index)

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"hello world"))
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=42),
        document=SimpleNamespace(file_name="notes.txt", mime_type="text/plain", file_id="fid", file_size=11),
    )

    task = asyncio.create_task(document_upload.process_document_upload(message, message.document))

    await _wait_until(started.is_set)

    created_before_cancel = [p for p in tmp_path.iterdir() if p.is_file()]
    assert len(created_before_cancel) == 2
    physical_path = next(p for p in created_before_cancel if not p.name.endswith(".meta.json"))
    sidecar_path = next(p for p in created_before_cancel if p.name.endswith(".meta.json"))

    task.cancel()

    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done()
        # Must not be removed while the worker might still be using it.
        assert physical_path.exists(), "physical file removed before the failing worker terminated"

    release.set()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.CancelledError):
            await task

    # Removed only after the worker actually terminated (with a failure) —
    # both the physical file and its sidecar, no orphan left behind.
    assert not physical_path.exists()
    assert not sidecar_path.exists()

    log_text = caplog.text
    assert sensitive_detail not in log_text
    assert "C:\\Users\\confidential_employee" not in log_text
    assert "sk-FAKE1234567890SECRET" not in log_text
    assert "RuntimeError" in log_text  # safe: exception type only

    # No generic Telegram failure message was attempted after cancellation.
    sent_texts = [c.args[1] for c in send_message_mock.await_args_list]
    assert not any("ошибка при загрузке" in t.lower() for t in sent_texts)


# ---------------------------------------------------------------------------
# FFmpeg / voice cancellation safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_during_ogg_conversion_does_not_abandon_worker_or_race_ogg_cleanup(monkeypatch, tmp_path):
    """End-to-end through handlers/voice.py: proves the OGG input is never
    removed while the (fake) ffmpeg worker could still be reading it, that
    the WAV output is cleaned up only after the worker actually terminates,
    and that voice.py's own OGG cleanup only runs afterward — all via
    threading.Event synchronization, no real ffmpeg process, no sleep-based
    race."""
    import handlers.voice as voice_handler
    from services import stt as stt_module

    started = threading.Event()
    release = threading.Event()
    wav_holder = {}

    async def fake_save_file_async(content, extension="tmp"):
        # Keep this test fully inside tmp_path — never the real data/ dir.
        p = tmp_path / f"voice_input.{extension}"
        p.write_bytes(content)
        return p

    def fake_convert(path):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        wav_path = Path(path).with_suffix('.wav')
        wav_path.write_bytes(b"RIFF....WAVEfmt fake wav bytes")
        wav_holder["path"] = wav_path
        return wav_path

    monkeypatch.setattr(voice_handler, "save_file_async", fake_save_file_async)
    monkeypatch.setattr(stt_module, "convert_ogg_to_wav", fake_convert)
    monkeypatch.setattr(
        voice_handler.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="voice/file_1.oga")),
    )
    monkeypatch.setattr(voice_handler.bot, "download_file", AsyncMock(return_value=b"OggS fake ogg bytes"))
    monkeypatch.setattr(voice_handler.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(voice_handler.bot, "send_message", AsyncMock())

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=555),
        chat=SimpleNamespace(id=555),
        voice=SimpleNamespace(file_id="fake-voice-id"),
    )

    task = asyncio.create_task(voice_handler.handle_voice_message(message))

    await _wait_until(started.is_set)

    ogg_path = tmp_path / "voice_input.ogg"
    assert ogg_path.exists()

    task.cancel()

    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "handler finished before the conversion worker did — it was abandoned"
        assert ogg_path.exists(), "OGG input removed while the conversion worker was still active"

    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # WAV cleaned up only after the worker actually terminated.
    assert "path" in wav_holder
    assert not wav_holder["path"].exists()
    # handlers/voice.py's own `finally` cleanup of the OGG input could only
    # safely run once the worker was done — and did.
    assert not ogg_path.exists()


@pytest.mark.asyncio
async def test_cancel_during_ogg_conversion_failure_stays_private_and_cleans_predictable_output(
    monkeypatch, tmp_path, caplog
):
    """If the conversion worker fails while the caller is already
    cancelled, the worker exception must be retrieved (never left for
    asyncio to report as "never retrieved"), any predictable WAV output
    cleaned, and only exception TYPE (never raw ffmpeg/pydub stderr text)
    logged.

    The fake worker deliberately WRITES the deterministic partial `.wav`
    output before raising — real ffmpeg/pydub failures can leave a
    partially-written file behind — so this test actually proves
    `_resolve_cancelled_conversion()`'s cleanup removes a file that
    genuinely existed, rather than merely asserting a path stays absent
    that no code path ever created."""
    from services import stt as stt_module

    started = threading.Event()
    release = threading.Event()
    sensitive_stderr = "ffmpeg error reading C:\\Users\\confidential\\salary_review.ogg: Authorization: Bearer sk-FAKE"

    ogg_path = tmp_path / "voice.ogg"
    ogg_path.write_bytes(b"OggS fake ogg bytes")
    expected_wav_path = ogg_path.with_suffix('.wav')

    def fake_convert(path):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        # Simulate ffmpeg having written a partial output before the
        # failure it then raises.
        expected_wav_path.write_bytes(b"RIFF....WAVEfmt partial/corrupt wav bytes")
        raise RuntimeError(sensitive_stderr)

    monkeypatch.setattr(stt_module, "convert_ogg_to_wav", fake_convert)

    task = asyncio.create_task(stt_module.transcribe_voice_message(ogg_path))

    await _wait_until(started.is_set)

    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done()
        assert not expected_wav_path.exists(), "partial WAV appeared before the worker actually wrote it"

    release.set()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.CancelledError):
            await task

    log_text = caplog.text
    assert sensitive_stderr not in log_text
    assert "salary_review" not in log_text
    assert "sk-FAKE" not in log_text
    assert "RuntimeError" in log_text

    # The partial WAV genuinely existed (written by the worker above) and
    # was removed only once the worker had actually terminated. A cleanup
    # regression (e.g. removing _resolve_cancelled_conversion()'s cleanup
    # call) would make this assertion fail, since the file really was
    # created.
    assert not expected_wav_path.exists()
