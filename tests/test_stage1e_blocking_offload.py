"""
Stage 1E regression tests: blocking work is offloaded off the async event
loop, without changing user-facing behavior, privacy guarantees (Stage 1D),
or upload safety guarantees (Stage 1B).

Each test proves the *mechanism* (the offloaded call actually executed on a
different OS thread than the event loop / test coroutine), not merely that a
mocked function's return value flows through correctly — that part is
already covered by the existing Stage 1B/1D test modules and is left
unchanged here.

All external calls (Telegram, OpenAI, Chroma/embeddings, ffmpeg/pydub) are
mocked. No network, no live ffmpeg process, and no mutation of the real
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


def _main_thread_id() -> int:
    return threading.get_ident()


# ---------------------------------------------------------------------------
# A. Blocking RAG work is offloaded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_similarity_search_runs_off_event_loop_thread(monkeypatch):
    """query_knowledge_base() must run vector_index.similarity_search_with_score()
    on a worker thread, not the coroutine's own (event-loop) thread."""
    import rag.query as rag_query
    from services.openai_client import openai_client

    caller_thread_id = _main_thread_id()
    recorded_thread_id = {}

    fake_doc = SimpleNamespace(metadata={"source": "notes.txt"}, page_content="Some content.")

    def synthetic_blocking_search(query, k=3):
        recorded_thread_id["id"] = threading.get_ident()
        return [(fake_doc, 0.1)]

    monkeypatch.setattr(rag_query.vector_index, "similarity_search_with_score", synthetic_blocking_search)
    monkeypatch.setattr(
        openai_client.client.chat.completions, "create",
        AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Here is the answer."))],
            usage=None,
        )),
    )

    response = await rag_query.query_knowledge_base("What is a list comprehension?")

    assert "id" in recorded_thread_id, "synthetic_blocking_search was never called"
    assert recorded_thread_id["id"] != caller_thread_id
    assert "Here is the answer." in response
    assert "notes.txt" in response  # source attribution preserved


# ---------------------------------------------------------------------------
# B. Document parsing/indexing is offloaded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_upload_pipeline_runs_off_event_loop_thread(monkeypatch, tmp_path):
    """_store_document_exclusively (disk write) and the combined load+index
    helper must both execute on a worker thread, not the handler coroutine's
    own thread.

    Stage 1E.1 strengthening: the load+index step is proven to run as ONE
    offloaded unit by spying on `_load_and_index_document` itself — the
    actual callable passed to `asyncio.to_thread()` — rather than merely
    comparing the thread IDs the loader/add_documents calls happened to run
    on. Two independently-offloaded `to_thread` calls could coincidentally
    land on the same default-executor thread, so thread-ID equality alone
    does not prove a single-callable boundary; call-count on the wrapper
    itself does.

    Preserves Stage 1B guarantees: exclusive write, cleanup untouched on
    success, source attribution."""
    import handlers.document_upload as document_upload

    caller_thread_id = _main_thread_id()
    store_thread_id = {}
    helper_thread_id = {}
    helper_call_count = {"n": 0}
    load_call_count = {"n": 0}
    add_call_count = {"n": 0}

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    real_store = document_upload._store_document_exclusively

    def spy_store(file_bytes, extension, attempts=5):
        store_thread_id["id"] = threading.get_ident()
        return real_store(file_bytes, extension, attempts=attempts)

    def fake_load_document(physical_path, display_name=None):
        load_call_count["n"] += 1
        return [SimpleNamespace(metadata={"source": display_name})]

    def fake_add_documents(chunks):
        add_call_count["n"] += 1

    real_load_and_index = document_upload._load_and_index_document

    def spy_load_and_index(physical_path, display_name):
        # This IS the callable handed to asyncio.to_thread() in
        # handlers/document_upload.py — proving it runs exactly once, off
        # the caller's thread, proves both load_document() and
        # add_documents() happened inside that single offloaded unit.
        helper_call_count["n"] += 1
        helper_thread_id["id"] = threading.get_ident()
        return real_load_and_index(physical_path, display_name)

    monkeypatch.setattr(document_upload, "_store_document_exclusively", spy_store)
    monkeypatch.setattr(document_upload, "_load_and_index_document", spy_load_and_index)
    monkeypatch.setattr(document_upload.document_loader, "load_document", fake_load_document)
    monkeypatch.setattr(document_upload.vector_index, "add_documents", fake_add_documents)

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

    await document_upload.process_document_upload(message, message.document)

    assert store_thread_id["id"] != caller_thread_id
    assert helper_thread_id["id"] != caller_thread_id
    # The combined helper is invoked exactly once through the offloaded
    # boundary — the actual proof of "one unit", independent of thread IDs.
    assert helper_call_count["n"] == 1
    assert load_call_count["n"] == 1
    assert add_call_count["n"] == 1

    created = list(tmp_path.iterdir())
    assert len(created) == 1  # exclusive write still happened exactly once

    success_text = send_message_mock.await_args.args[1]
    assert "успешно загружен" in success_text


@pytest.mark.asyncio
async def test_document_upload_offloaded_failure_preserves_privacy_and_cleanup(monkeypatch, tmp_path, caplog):
    """D. Exceptions propagate: when the offloaded load+index pipeline
    raises, the exception must reach process_document_upload's existing
    error boundary — generic user message, sanitized (Stage 1D) log, and the
    partially-created file cleaned up — exactly as before offloading."""
    import handlers.document_upload as document_upload

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    sensitive_detail = "corrupt PDF stream at absolute path C:\\Users\\confidential\\report.pdf"
    monkeypatch.setattr(
        document_upload.document_loader, "load_document",
        Mock(side_effect=ValueError(sensitive_detail)),
    )
    add_mock = Mock()
    monkeypatch.setattr(document_upload.vector_index, "add_documents", add_mock)

    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/report.pdf")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"%PDF-1.4 fake bytes"))
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=42),
        document=SimpleNamespace(file_name="report.pdf", mime_type="application/pdf", file_id="fid", file_size=20),
    )

    with caplog.at_level(logging.ERROR):
        await document_upload.process_document_upload(message, message.document)

    add_mock.assert_not_called()
    assert list(tmp_path.iterdir()) == []  # orphan file cleaned up

    log_text = caplog.text
    assert sensitive_detail not in log_text
    assert "Document upload failed" in log_text
    assert "ValueError" in log_text

    user_text = send_message_mock.await_args.args[1]
    assert sensitive_detail not in user_text
    assert "ошибка" in user_text.lower()


# ---------------------------------------------------------------------------
# C. Voice conversion is offloaded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ogg_to_wav_conversion_runs_off_event_loop_thread(monkeypatch, tmp_path):
    """transcribe_voice_message() must run convert_ogg_to_wav() (pydub/ffmpeg)
    on a worker thread. STT (Whisper) itself stays on the async provider path
    and is unaffected."""
    from services import stt
    from services.openai_client import openai_client

    caller_thread_id = _main_thread_id()
    conversion_thread_id = {}

    ogg_path = tmp_path / "voice.ogg"
    ogg_path.write_bytes(b"OggS fake ogg bytes")
    wav_path = tmp_path / "voice.wav"

    def fake_convert(path):
        conversion_thread_id["id"] = threading.get_ident()
        wav_path.write_bytes(b"RIFF....WAVEfmt fake wav bytes")
        return wav_path

    monkeypatch.setattr(stt, "convert_ogg_to_wav", fake_convert)
    monkeypatch.setattr(
        openai_client.client.audio.transcriptions, "create",
        AsyncMock(return_value="transcribed text"),
    )
    cleanup_mock = Mock()
    monkeypatch.setattr(stt, "cleanup_file", cleanup_mock)

    result = await stt.transcribe_voice_message(ogg_path)

    assert conversion_thread_id["id"] != caller_thread_id
    assert result == "transcribed text"
    cleanup_mock.assert_called_once_with(wav_path)


# ---------------------------------------------------------------------------
# E. Event-loop responsiveness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_offloaded_rag_query(monkeypatch):
    """While an offloaded RAG similarity search is still blocked (deliberately
    held open via a threading.Event, never a sleep-based race), a concurrent
    lightweight coroutine must still be able to run and complete on the event
    loop."""
    import rag.query as rag_query
    from services.openai_client import openai_client

    started = threading.Event()
    release = threading.Event()
    fake_doc = SimpleNamespace(metadata={"source": "notes.txt"}, page_content="Some content.")

    def blocking_search(query, k=3):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        return [(fake_doc, 0.1)]

    monkeypatch.setattr(rag_query.vector_index, "similarity_search_with_score", blocking_search)
    monkeypatch.setattr(
        openai_client.client.chat.completions, "create",
        AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=None,
        )),
    )

    task = asyncio.create_task(rag_query.query_knowledge_base("question"))

    # Poll (cheaply) for the worker thread to signal it has actually entered
    # the blocking call — bounded, not an arbitrary fixed sleep used as the
    # correctness assertion itself.
    for _ in range(500):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "worker thread never started the blocking call"
    assert not task.done()

    # The event loop must still be free to run other coroutines right now.
    progressed = []

    async def light_task():
        progressed.append(True)

    await asyncio.wait_for(light_task(), timeout=1)
    assert progressed == [True]
    assert not task.done()  # the offloaded call is still blocked

    release.set()
    response = await asyncio.wait_for(task, timeout=5)
    assert "answer" in response


# ---------------------------------------------------------------------------
# F. Session safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_thread_never_mutates_user_session(monkeypatch):
    """route_text_request() in RAG mode exercises the offloaded similarity
    search on a worker thread and then mutates UserSession on completion.
    Every UserSession.add_message() call must happen on the event-loop
    (caller's) thread — never from inside the offloaded worker thread."""
    import services.router as router_module
    from services.router import route_text_request
    from utils.helpers import user_sessions
    import rag.query as rag_query
    from services.openai_client import openai_client
    from config import BotMode

    caller_thread_id = _main_thread_id()
    user_id = 777
    user_sessions.sessions.pop(user_id, None)
    user_sessions.set_mode(user_id, BotMode.RAG)

    search_thread_id = {}
    fake_doc = SimpleNamespace(metadata={"source": "notes.txt"}, page_content="Some content.")

    def blocking_search(query, k=3):
        search_thread_id["id"] = threading.get_ident()
        return [(fake_doc, 0.1)]

    monkeypatch.setattr(rag_query.vector_index, "similarity_search_with_score", blocking_search)
    monkeypatch.setattr(
        openai_client.client.chat.completions, "create",
        AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=None,
        )),
    )
    # Also bypasses the image-generation-intent detection's own LLM call.
    monkeypatch.setattr(
        router_module, "detect_image_generation_intent",
        AsyncMock(return_value={"needs_generation": False, "confidence": 0.0}),
    )

    add_message_call_threads = []
    real_add_message = user_sessions.add_message

    def spy_add_message(uid, role, content):
        add_message_call_threads.append(threading.get_ident())
        return real_add_message(uid, role, content)

    monkeypatch.setattr(user_sessions, "add_message", spy_add_message)

    try:
        await route_text_request(user_id, "What is a Python decorator?")
    finally:
        user_sessions.sessions.pop(user_id, None)
        user_sessions.sessions.pop(f"{user_id}_mode", None)

    assert search_thread_id["id"] != caller_thread_id  # the blocking work really was offloaded
    assert len(add_message_call_threads) == 2  # user message + assistant response
    assert all(t == caller_thread_id for t in add_message_call_threads), \
        "UserSession.add_message() must only ever be called on the event-loop thread"


# ---------------------------------------------------------------------------
# Thread-safety: shared VectorIndex/Chroma access is serialized
# ---------------------------------------------------------------------------

def test_vector_index_add_documents_acquires_lock_around_underlying_call(monkeypatch):
    """Deterministic (non-probabilistic) proof that
    `VectorIndex.add_documents()` acquires `self._lock` strictly BEFORE
    entering the underlying `vectorstore.add_documents()` call, holds it
    for the call's entire duration, and releases it only after.

    Stage 1E.3 correction: Stage 1E.2's version (6 threads + a
    `threading.Barrier` + a polled `attempting_count`) inferred locking
    behavior from concurrent-thread *timing* — every caller incremented
    `attempting_count` before actually contending for the lock, so a
    schedule where the lock is genuinely absent but every thread simply
    hasn't reached the underlying call yet by the time the assertions run
    is still logically possible, even if practically unlikely. That is a
    probabilistic proof, not a logical one.

    This version replaces timing inference with two independent
    *mechanical* guarantees instead:

    1. `self._lock` is replaced with a thin recording wrapper around a
       REAL `threading.RLock`, which records the exact order of
       `lock_enter` / `vectorstore_entered` / `lock_exit` events. If
       production `add_documents()` ever stops using `with self._lock:`,
       the `lock_enter`/`lock_exit` events simply never appear — this is
       a structural fact about which code path ran, not a timing
       coincidence.
    2. While the (single) worker thread is deterministically known to be
       inside the underlying call (via a `threading.Event`, never a
       sleep), THIS thread attempts to acquire that same real `RLock`
       directly with a short bounded timeout. `threading.RLock` mutual
       exclusion between different threads is a mechanical guarantee of
       CPython's lock implementation, not something that can pass by
       scheduler luck: if the production code path really held the lock,
       this second acquire CANNOT succeed within the timeout; if the lock
       were silently missing from the call path, this acquire would
       trivially succeed.

    Together these prove the real nested-call structure
    (`with self._lock: ... self.vectorstore.add_documents(...)`) is
    exactly what runs — not merely that it's statistically likely."""
    import rag.index as rag_index

    events = []
    events_lock = threading.Lock()

    class RecordingRLock:
        """Wraps a real threading.RLock, recording enter/exit order. Also
        exposes acquire()/release() directly so the test thread can attempt
        a genuine, independently-timed second acquisition against the very
        same underlying lock object the production code uses."""

        def __init__(self):
            self._real_lock = threading.RLock()

        def __enter__(self):
            self._real_lock.acquire()
            with events_lock:
                events.append("lock_enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            with events_lock:
                events.append("lock_exit")
            self._real_lock.release()
            return False

        def acquire(self, blocking=True, timeout=-1):
            return self._real_lock.acquire(blocking, timeout)

        def release(self):
            self._real_lock.release()

    recording_lock = RecordingRLock()
    monkeypatch.setattr(rag_index.vector_index, "_lock", recording_lock)

    entered_vectorstore = threading.Event()
    release_vectorstore = threading.Event()

    def fake_chroma_add_documents(documents):
        with events_lock:
            events.append("vectorstore_entered")
        entered_vectorstore.set()
        assert release_vectorstore.wait(timeout=5), "test setup: release_vectorstore was never set"

    monkeypatch.setattr(rag_index.vector_index.vectorstore, "add_documents", fake_chroma_add_documents)

    worker_thread = threading.Thread(target=lambda: rag_index.vector_index.add_documents(["chunk"]))
    worker_thread.start()

    assert entered_vectorstore.wait(timeout=5), "worker never reached the underlying vectorstore call"

    # Mechanical mutual-exclusion proof (not timing inference): the worker
    # is deterministically known (via the Event above) to be inside the
    # underlying call right now. A second acquisition attempt against the
    # SAME real RLock, from this (different) thread, must fail within a
    # short bounded timeout — RLock ownership is per-thread, so this can
    # only succeed if nothing is actually holding the lock.
    acquired = recording_lock.acquire(blocking=True, timeout=0.2)
    assert not acquired, "lock was not held during the underlying vectorstore call"

    release_vectorstore.set()
    worker_thread.join(timeout=5)
    assert not worker_thread.is_alive(), "worker thread never terminated"

    # Exact event order: the lock was entered before, and exited after,
    # the underlying call — nothing more, nothing less.
    assert events == ["lock_enter", "vectorstore_entered", "lock_exit"], (
        f"add_documents() did not acquire the lock strictly around the "
        f"underlying vectorstore call: {events!r}"
    )

    # The lock is free again afterward — released, not leaked.
    assert recording_lock.acquire(blocking=True, timeout=0.2), "lock was not released after add_documents() returned"
    recording_lock.release()


def test_vector_index_nested_locked_methods_do_not_deadlock(monkeypatch, tmp_path):
    """index_documents_directory() calls clear_index() and add_documents()
    on itself while already holding VectorIndex._lock.

    Stage 1E.1 strengthening: merely asserting `_lock` is an RLock (as the
    original Stage 1E test did) doesn't prove the actual nested production
    path avoids deadlock — it only proves the lock TYPE is reentrant. This
    exercises the real path end-to-end (index_documents_directory ->
    clear_index -> add_documents, each reacquiring the lock) with
    loader/vectorstore mocked out (no Chroma/network/filesystem side
    effects beyond an isolated tmp_path), run on a background thread with a
    bounded `join(timeout=...)` — a regression to a non-reentrant lock would
    hang that thread forever and fail this test instead of hanging the
    whole suite.

    Stage 1E.2 process-safety fix: `join(timeout=...)` only bounds how long
    THIS test waits — if the production lock ever regressed to a
    non-reentrant `Lock`, the thread below would block forever trying to
    reacquire it, and a non-daemon thread left alive past the timeout keeps
    the whole pytest process from exiting even after this test reports
    failure. Marking it `daemon=True` means a genuine deadlock here fails
    just this test (via the `is_alive()` assertion after the bounded join)
    instead of hanging the entire suite."""
    import rag.index as rag_index

    fake_documents = ["chunk-a", "chunk-b"]

    load_directory_mock = Mock(return_value=fake_documents)
    monkeypatch.setattr(rag_index.document_loader, "load_directory", load_directory_mock)
    add_documents_mock = Mock()
    monkeypatch.setattr(rag_index.vector_index.vectorstore, "add_documents", add_documents_mock)

    # clear_index() deletes/recreates persist_directory and calls
    # _load_or_create_vectorstore() — isolate to a fresh tmp_path and stub
    # vectorstore reconstruction so no real Chroma/network call happens.
    monkeypatch.setattr(rag_index.vector_index, "persist_directory", tmp_path / "chroma_isolated")
    reload_vectorstore_mock = Mock()
    monkeypatch.setattr(rag_index.vector_index, "_load_or_create_vectorstore", reload_vectorstore_mock)

    result_holder = {}
    error_holder = {}

    def run_nested_path():
        try:
            result_holder["count"] = rag_index.vector_index.index_documents_directory(
                directory=tmp_path, force_reindex=True
            )
        except BaseException as e:
            error_holder["error"] = e

    # daemon=True: see docstring above — a genuine deadlock must fail only
    # this test, never hang the pytest process.
    t = threading.Thread(target=run_nested_path, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "index_documents_directory -> clear_index -> add_documents deadlocked"
    assert "error" not in error_holder, f"nested locked path raised: {error_holder.get('error')!r}"
    # clear_index() actually ran (first nested lock reacquisition)...
    reload_vectorstore_mock.assert_called_once()
    # ...and the load+add pipeline actually ran too (second nested lock
    # reacquisition, via self.add_documents() called from inside
    # index_documents_directory() while it still holds the outer lock).
    load_directory_mock.assert_called_once_with(tmp_path)
    add_documents_mock.assert_called_once_with(fake_documents)
    assert result_holder.get("count") == len(fake_documents)
