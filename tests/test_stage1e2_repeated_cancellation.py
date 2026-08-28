"""
Stage 1E.2 regression tests: repeated / shutdown-style cancellation safety.

A second independent Codex audit of Stage 1E.1 found that
`asyncio.create_task(asyncio.to_thread(...))` + `await_shielded()` was not
actually safe: cancelling a `Task` wrapping `to_thread(...)` can flip that
Task's underlying `run_in_executor` Future to CANCELLED immediately —
`concurrent.futures.Future.cancel()` silently fails once the callable is
already running, but the asyncio-level wrapper still marks itself cancelled
regardless — so "the Task reached a terminal state" was never actually
proof that the executor thread had stopped. `await_shielded()` also only
handled a SINGLE cancellation: a second `cancel()` arriving while it was
already in its `except CancelledError: await asyncio.wait({task})` block
would interrupt THAT await and abandon the worker.

`utils/helpers.py:submit_worker()` + `await_worker()` (Stage 1E.2) fix
both problems: `submit_worker()` returns a bare `loop.run_in_executor()`
Future — never a Task — that nothing in this codebase ever calls
`.cancel()` on, so it is not reachable/cancellable via a shutdown sweep over
`asyncio.all_tasks()`, and its terminal state can only be produced by the
executor thread itself actually finishing. `await_worker()` loops on
`asyncio.shield(future)`, catching and preserving the FIRST
`CancelledError` no matter how many further cancellations (including a
broad/shutdown-style sweep) arrive at the same await point.

Every test here uses `threading.Event`-based synchronization to force
cancellation to land while a worker is deterministically still running —
never an arbitrary sleep as the actual correctness assertion. All external
calls (Telegram, OpenAI, Chroma/embeddings, ffmpeg/pydub) are mocked. No
network, no live ffmpeg process, and no mutation of the real
data/documents, data/documents/uploads, data/chroma_db, or bot.log paths
(same tests/conftest.py isolation as every other test module).
"""

import asyncio
import gc
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    """Poll `predicate()` on the event loop until it's True or `timeout`
    elapses. Used only to detect a worker-thread Event being set — never as
    the correctness assertion itself."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# A. Double cancellation during document storage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_twice_during_storage_still_waits_for_worker(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload

    started = threading.Event()
    release = threading.Event()
    created_path_holder = {}

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    def fake_store(file_bytes, extension):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        path = tmp_path / f"owned_upload{extension}"
        path.write_bytes(file_bytes)
        created_path_holder["path"] = path
        return path

    load_mock = Mock()
    add_mock = Mock()
    monkeypatch.setattr(document_upload, "_store_document_exclusively", fake_store)
    monkeypatch.setattr(document_upload.document_loader, "load_document", load_mock)
    monkeypatch.setattr(document_upload.vector_index, "add_documents", add_mock)

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

    # First cancellation.
    task.cancel()
    await asyncio.sleep(0)  # let the first CancelledError actually land inside await_worker()
    assert not task.done(), "handler finished after the FIRST cancel — worker abandoned"
    assert "path" not in created_path_holder

    # Second cancellation, arriving while await_worker() is already
    # reconciling the first one.
    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "handler finished after the SECOND cancel — worker abandoned"
    assert "path" not in created_path_holder  # worker genuinely hasn't returned yet

    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The newly-owned file was created, then removed — never abandoned.
    assert "path" in created_path_holder
    assert not created_path_holder["path"].exists()

    # Parser/indexer must never have started.
    load_mock.assert_not_called()
    add_mock.assert_not_called()

    sent_texts = [c.args[1] for c in send_message_mock.await_args_list]
    assert not any("ошибка" in t.lower() for t in sent_texts)
    assert not any("успешно" in t.lower() for t in sent_texts)


# ---------------------------------------------------------------------------
# B. Double cancellation during document indexing — SUCCESS
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_twice_during_indexing_success_still_retains_file(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload

    started = threading.Event()
    release = threading.Event()
    call_count = {"n": 0}

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    def fake_load_and_index(physical_path, display_name):
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

    created_before_cancel = [p for p in tmp_path.iterdir() if p.is_file()]
    assert len(created_before_cancel) == 1
    physical_path = created_before_cancel[0]

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "handler finished after the FIRST cancel — worker abandoned"
    assert physical_path.exists()

    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "handler finished after the SECOND cancel — worker abandoned"
        assert physical_path.exists(), "physical file removed while the indexing worker was still active"

    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert call_count["n"] == 1  # ingestion happened exactly once
    assert physical_path.exists()  # never delete successfully ingested data

    sent_texts = [c.args[1] for c in send_message_mock.await_args_list]
    assert not any("успешно загружен" in t for t in sent_texts)


# ---------------------------------------------------------------------------
# C. Worker failure observed AFTER repeated cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_exception_after_repeated_cancel_is_retrieved_and_stays_private(
    monkeypatch, tmp_path, caplog
):
    """Caller is cancelled twice; the worker only fails afterward, once it
    is finally released. Proves: the exception is retrieved (no asyncio
    "exception was never retrieved" warning after a forced gc.collect()),
    its raw content never reaches the logs, reconciliation (file cleanup)
    still runs, and the caller ends up seeing CancelledError — never the
    worker's own exception in its place."""
    import handlers.document_upload as document_upload

    started = threading.Event()
    release = threading.Event()
    sensitive_detail = (
        "chroma write failed for C:\\Users\\confidential_employee\\secret_report.pdf: "
        "Authorization: Bearer sk-FAKE-DOUBLE-CANCEL-SECRET"
    )

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    def fake_load_and_index(physical_path, display_name):
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
    assert len(created_before_cancel) == 1
    physical_path = created_before_cancel[0]

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done()
        assert physical_path.exists(), "physical file removed before the failing worker terminated"

    release.set()

    with caplog.at_level(logging.WARNING):
        # The caller must see CancelledError, never the worker's RuntimeError.
        with pytest.raises(asyncio.CancelledError):
            await task
        # Force any pending Future.__del__-triggered "exception was never
        # retrieved" reporting to fire deterministically now, rather than
        # relying on incidental refcount-drop timing.
        gc.collect()

    # Removed only after the worker actually terminated (with a failure).
    assert not physical_path.exists()

    log_text = caplog.text
    assert "was never retrieved" not in log_text
    assert sensitive_detail not in log_text
    assert "C:\\Users\\confidential_employee" not in log_text
    assert "sk-FAKE-DOUBLE-CANCEL-SECRET" not in log_text
    assert "RuntimeError" in log_text  # safe: exception type only

    sent_texts = [c.args[1] for c in send_message_mock.await_args_list]
    assert not any("ошибка при загрузке" in t.lower() for t in sent_texts)


# ---------------------------------------------------------------------------
# D. Double cancellation during OGG->WAV conversion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_twice_during_ogg_conversion_does_not_abandon_worker(monkeypatch, tmp_path):
    """End-to-end through handlers/voice.py: two cancellations in a row must
    not let the OGG input or the produced WAV be touched before the (fake)
    ffmpeg worker genuinely terminates; only then does reconciliation happen
    and outer OGG cleanup run."""
    import handlers.voice as voice_handler
    from services import stt as stt_module

    started = threading.Event()
    release = threading.Event()
    wav_holder = {}

    async def fake_save_file_async(content, extension="tmp"):
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
        from_user=SimpleNamespace(id=556),
        chat=SimpleNamespace(id=556),
        voice=SimpleNamespace(file_id="fake-voice-id"),
    )

    task = asyncio.create_task(voice_handler.handle_voice_message(message))

    await _wait_until(started.is_set)

    ogg_path = tmp_path / "voice_input.ogg"
    assert ogg_path.exists()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "handler finished after the FIRST cancel — worker abandoned"
    assert ogg_path.exists()

    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done(), "handler finished after the SECOND cancel — worker abandoned"
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


# ---------------------------------------------------------------------------
# E. Shutdown-style broad cancellation (no separate internal Task to sweep)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shutdown_style_broad_cancellation_does_not_abandon_storage_worker(monkeypatch, tmp_path):
    """Simulates a shutdown sweep (`for t in asyncio.all_tasks(): t.cancel()`)
    firing twice against every live Task while a storage worker is
    deliberately held open. Also proves the structural precondition for
    this to matter: while the worker is running, the ONLY asyncio.Task in
    flight for this operation is the handler's own Task — submit_worker()
    never creates a second, independently-cancellable Task wrapping the
    executor work (see utils/helpers.py)."""
    import handlers.document_upload as document_upload

    started = threading.Event()
    release = threading.Event()
    created_path_holder = {}

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    def fake_store(file_bytes, extension):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        path = tmp_path / f"owned_upload{extension}"
        path.write_bytes(file_bytes)
        created_path_holder["path"] = path
        return path

    load_mock = Mock()
    add_mock = Mock()
    monkeypatch.setattr(document_upload, "_store_document_exclusively", fake_store)
    monkeypatch.setattr(document_upload.document_loader, "load_document", load_mock)
    monkeypatch.setattr(document_upload.vector_index, "add_documents", add_mock)

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"hello world"))
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=43),
        chat=SimpleNamespace(id=43),
        document=SimpleNamespace(file_name="notes.txt", mime_type="text/plain", file_id="fid", file_size=11),
    )

    handler_task = asyncio.create_task(document_upload.process_document_upload(message, message.document))
    this_test_task = asyncio.current_task()

    await _wait_until(started.is_set)

    # Structural proof: no extra Task exists to independently abandon the
    # worker — the only other live Task is this test coroutine itself.
    other_tasks = {t for t in asyncio.all_tasks() if t is not this_test_task}
    assert other_tasks == {handler_task}, (
        "an extra internal Task exists for the worker — it would be "
        "independently reachable (and cancellable) by a shutdown sweep"
    )

    def shutdown_sweep():
        for t in asyncio.all_tasks():
            if t is not this_test_task:
                t.cancel()

    shutdown_sweep()  # first broad sweep
    await asyncio.sleep(0)
    assert not handler_task.done(), "handler finished after the first shutdown-style sweep — worker abandoned"
    assert "path" not in created_path_holder

    shutdown_sweep()  # second broad sweep, while still reconciling the first
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not handler_task.done(), "handler finished during a shutdown-style sweep — worker abandoned"
    assert "path" not in created_path_holder

    release.set()

    with pytest.raises(asyncio.CancelledError):
        await handler_task

    assert "path" in created_path_holder
    assert not created_path_holder["path"].exists()
    load_mock.assert_not_called()
    add_mock.assert_not_called()
