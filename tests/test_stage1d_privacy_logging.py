"""
Stage 1D regression tests: privacy-safe logging + safe Telegram error
responses.

Covers:
- text/provider failures never leak API keys, token-bearing Telegram URLs,
  or the raw user message into logs; the user receives a generic message
- vision/photo processing failures (not just download failures, already
  covered by Stage 1A) never leak sensitive exception content
- voice/STT failures never leak the raw transcript or raw exception text
- document upload: a confidential-looking filename is never logged, and a
  token-bearing download-stage exception is fully sanitized end to end
  (Stage 1B's own pinned tests separately cover the loader/indexing-failure
  exception-retention carve-out documented in handlers/document_upload.py)
- a distinctive "secret-looking" user message never appears in logs on a
  successful request either
- /stats never forwards a raw internal exception string to the user
  (rag/index.py's VectorIndex.get_stats())

All external calls (Telegram, OpenAI, Chroma/embeddings) are mocked. No
network access is performed by this test module. Nothing here weakens any
assertion in tests/test_stage1a_security.py, tests/test_stage1b_document_upload.py,
or tests/test_stage1c_access_control.py — all three continue to pass.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from utils.helpers import user_sessions

FAKE_OPENAI_KEY = "sk-FAKE1234567890ABCDEFSECRETKEYDONOTUSE"
FAKE_TELEGRAM_TOKEN = "123456789:FAKE-STAGE1D-TOKEN-FOR-LOG-LEAK-TEST"
FAKE_TELEGRAM_URL = f"https://api.telegram.org/file/bot{FAKE_TELEGRAM_TOKEN}/photos/file_1.jpg"


def _leaking_exception_message(raw_user_text: str = "") -> str:
    """A synthetic provider exception embedding everything Stage 1D's
    threat model says must never reach logs or the user."""
    parts = [
        f"Authorization: Bearer {FAKE_OPENAI_KEY}",
        f"request to {FAKE_TELEGRAM_URL} failed: 401 Unauthorized",
    ]
    if raw_user_text:
        parts.append(f"payload: {raw_user_text}")
    return " | ".join(parts)


def _fake_sdk_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


@pytest.fixture(autouse=True)
def _clean_sessions():
    user_sessions.sessions.clear()
    yield
    user_sessions.sessions.clear()


def _make_text_message(user_id: int, text: str):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        text=text,
        content_type="text",
    )


def _make_photo_message(user_id: int, caption: str = "", file_id: str = "fake-file-id"):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        caption=caption,
        photo=[SimpleNamespace(file_id=file_id)],
    )


# ---------------------------------------------------------------------------
# A. Text/provider failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_provider_failure_leaks_nothing_and_returns_generic_message(monkeypatch, caplog):
    import handlers.text as text_handler
    from services.openai_client import openai_client

    user_id = 9001
    raw_user_text = "MySuperSecretDiaryEntry_do_not_share_2026"
    leaking_message = _leaking_exception_message(raw_user_text)

    create_mock = AsyncMock(side_effect=Exception(leaking_message))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    send_message_mock = AsyncMock()
    monkeypatch.setattr(text_handler.bot, "send_message", send_message_mock)
    monkeypatch.setattr(text_handler.bot, "send_chat_action", AsyncMock())

    message = _make_text_message(user_id, raw_user_text)

    with caplog.at_level(logging.DEBUG):
        await text_handler.handle_text_message(message)

    # Generic response only.
    sent_text = send_message_mock.await_args.args[1]
    assert leaking_message not in sent_text
    assert FAKE_OPENAI_KEY not in sent_text
    assert FAKE_TELEGRAM_TOKEN not in sent_text
    assert raw_user_text not in sent_text
    assert "ошибка" in sent_text.lower()

    log_text = caplog.text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert "api.telegram.org" not in log_text
    assert leaking_message not in log_text
    assert raw_user_text not in log_text
    # Safe, structured diagnostics must still be observable.
    assert "route_text_request failed" in log_text
    assert "Exception" in log_text


# ---------------------------------------------------------------------------
# B. Vision/photo failure (processing failure, not the download failure
# already covered by tests/test_stage1a_security.py)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vision_processing_failure_leaks_nothing_and_returns_generic_message(monkeypatch, caplog):
    import handlers.image as image_handler
    from services.openai_client import openai_client

    user_id = 9002
    leaking_message = _leaking_exception_message("What is in this diagram?")

    monkeypatch.setattr(
        image_handler.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="photos/file_1.jpg")),
    )
    download_mock = AsyncMock(return_value=b"\xff\xd8\xff\xe0fakejpegbytes")
    monkeypatch.setattr(image_handler.bot, "download_file", download_mock)
    monkeypatch.setattr(image_handler.bot, "send_chat_action", AsyncMock())
    send_message_mock = AsyncMock()
    monkeypatch.setattr(image_handler.bot, "send_message", send_message_mock)

    create_mock = AsyncMock(side_effect=Exception(leaking_message))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    message = _make_photo_message(user_id, caption="What is in this diagram?")

    with caplog.at_level(logging.DEBUG):
        await image_handler.handle_photo_message(message)

    # Stage 1A's download boundary is untouched: the bot's own token-scoped
    # calls were still used to fetch the file before the (mocked) provider
    # call failed.
    download_mock.assert_awaited_once()

    sent_text = send_message_mock.await_args.args[1]
    assert leaking_message not in sent_text
    assert FAKE_OPENAI_KEY not in sent_text
    assert FAKE_TELEGRAM_TOKEN not in sent_text
    assert "ошибка" in sent_text.lower()

    log_text = caplog.text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert "api.telegram.org" not in log_text
    assert leaking_message not in log_text
    # route_image_request() (services/router.py) absorbs the exception and
    # returns a generic response, so handlers/image.py's own try succeeds —
    # the safe, structured event fires one layer down the call stack.
    assert "route_image_request failed" in log_text
    assert "Exception" in log_text


# ---------------------------------------------------------------------------
# C. Voice/STT/TTS failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_stt_failure_leaks_nothing_and_returns_generic_message(monkeypatch, caplog):
    """End-to-end handler test: route_voice_request's transcription step
    fails with a sensitive exception. The audio-conversion step (pydub/
    ffmpeg) is bypassed by mocking services.router.transcribe_voice_message
    directly, so this test stays offline and independent of any local
    ffmpeg installation."""
    import handlers.voice as voice_handler
    import services.router as router_module

    user_id = 9003
    raw_transcript_like_text = "I have a heart condition called SecretDiagnosisXYZ"
    leaking_message = _leaking_exception_message(raw_transcript_like_text)

    monkeypatch.setattr(
        voice_handler.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="voice/file_1.oga")),
    )
    monkeypatch.setattr(voice_handler.bot, "download_file", AsyncMock(return_value=b"fake ogg bytes"))
    monkeypatch.setattr(voice_handler.bot, "send_chat_action", AsyncMock())
    send_message_mock = AsyncMock()
    monkeypatch.setattr(voice_handler.bot, "send_message", send_message_mock)

    monkeypatch.setattr(
        router_module, "transcribe_voice_message",
        AsyncMock(side_effect=Exception(leaking_message)),
    )

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        voice=SimpleNamespace(file_id="fake-voice-id"),
    )

    with caplog.at_level(logging.DEBUG):
        await voice_handler.handle_voice_message(message)

    sent_text = send_message_mock.await_args.args[1]
    assert leaking_message not in sent_text
    assert raw_transcript_like_text not in sent_text
    assert "ошибка" in sent_text.lower()

    log_text = caplog.text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert leaking_message not in log_text
    assert raw_transcript_like_text not in log_text
    assert "Voice message failed" in log_text
    assert "Exception" in log_text


@pytest.mark.asyncio
async def test_stt_service_itself_never_logs_raw_provider_exception(monkeypatch, caplog, tmp_path):
    """Unit-level check on services/stt.py directly: a raw Whisper API
    exception must not be logged verbatim at the source, independent of
    which handler eventually calls it. Uses a .wav path to skip the
    ogg->wav pydub conversion step entirely; the file must actually exist
    on disk since openai_client.transcribe_audio() opens it before making
    the (mocked) API call."""
    from services import stt
    from services.openai_client import openai_client

    audio_path = tmp_path / "fake_audio.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt fake wav bytes")

    leaking_message = _leaking_exception_message("this is my private voice memo content")
    monkeypatch.setattr(
        openai_client.client.audio.transcriptions, "create",
        AsyncMock(side_effect=Exception(leaking_message)),
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception):
            await stt.transcribe_voice_message(audio_path)

    log_text = caplog.text
    assert leaking_message not in log_text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert "STT transcription failed" in log_text
    assert "Exception" in log_text


# ---------------------------------------------------------------------------
# D. Document processing failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_confidential_filename_and_token_exception_fully_sanitized(monkeypatch, tmp_path, caplog):
    """
    Exercises the download stage of document upload (the stage that can
    genuinely carry the Telegram token in its exception, per Stage 1A/1B's
    own design) together with a confidential-looking user-controlled
    filename, and asserts BOTH are absent everywhere: the filename is never
    logged (Stage 1D's new filename-privacy contract) and the token-bearing
    exception text is never logged (Stage 1B's pre-existing token-safety
    contract, still intact). Nothing is written to disk, matching Stage 1B's
    cleanup guarantee (there is nothing to clean up because ingestion never
    started), and the user receives a generic message.
    """
    import handlers.document_upload as document_upload

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    confidential_filename = "Ivanov_passport_scan_CONFIDENTIAL.pdf"
    leaking_message = _leaking_exception_message()

    async def raise_leaking_error(file_id):
        raise Exception(leaking_message)

    monkeypatch.setattr(document_upload.bot, "get_file", raise_leaking_error)
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    document = SimpleNamespace(
        file_name=confidential_filename,
        mime_type="application/pdf",
        file_id="fake-doc-id",
        file_size=1234,
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=9004),
        chat=SimpleNamespace(id=9004),
        document=document,
    )

    with caplog.at_level(logging.DEBUG):
        await document_upload.handle_document_message(message)

    # Stage 1B cleanup guarantee: nothing was ever written to disk.
    assert list(tmp_path.iterdir()) == []

    sent_texts = [c.args[1] for c in send_message_mock.await_args_list]
    assert not any(confidential_filename in t for t in sent_texts)
    assert not any(leaking_message in t for t in sent_texts)
    assert not any(FAKE_OPENAI_KEY in t for t in sent_texts)
    assert any("ошибка" in t.lower() for t in sent_texts)

    log_text = caplog.text
    assert confidential_filename not in log_text
    assert leaking_message not in log_text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert "api.telegram.org" not in log_text
    assert "document_download" in log_text


@pytest.mark.asyncio
async def test_document_upload_received_log_never_contains_filename(monkeypatch, caplog):
    """The 'Document received' event fires before any download/processing
    is attempted, and must never carry the raw Telegram display filename."""
    import handlers.document_upload as document_upload

    confidential_filename = "board_meeting_minutes_SECRET.docx"
    document = SimpleNamespace(
        file_name=confidential_filename,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_id="fake-doc-id",
        file_size=42,
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=9005),
        chat=SimpleNamespace(id=9005),
        document=document,
    )

    async def _fake_process(msg, doc):
        return None

    monkeypatch.setattr(document_upload, "process_document_upload", _fake_process)
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())

    with caplog.at_level(logging.DEBUG):
        await document_upload.handle_document_message(message)

    assert confidential_filename not in caplog.text


# ---------------------------------------------------------------------------
# E. User text privacy (successful flow)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_text_flow_never_logs_raw_user_message(monkeypatch, caplog):
    import handlers.text as text_handler
    from services.openai_client import openai_client

    user_id = 9006
    secret_text = "MyBankAccountNumberIs 1234-5678-9012-SECRET"

    create_mock = AsyncMock(return_value=_fake_sdk_response("Here is a general Python tip."))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    monkeypatch.setattr(text_handler.bot, "send_chat_action", AsyncMock())
    send_message_mock = AsyncMock()
    monkeypatch.setattr(text_handler.bot, "send_message", send_message_mock)

    message = _make_text_message(user_id, secret_text)

    with caplog.at_level(logging.DEBUG):
        await text_handler.handle_text_message(message)

    assert secret_text not in caplog.text
    # Safe metadata (length, not content) is still observable.
    assert f"text_len={len(secret_text)}" in caplog.text


# ---------------------------------------------------------------------------
# F. /stats never forwards a raw internal exception to the user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stats_command_never_forwards_raw_exception_to_user(monkeypatch, caplog):
    import handlers.start as start_handler
    import rag.index as rag_index

    # Exercise the real VectorIndex.get_stats() implementation (not a
    # replacement mock) by making the underlying Chroma collection access
    # itself fail, so the sanitization inside get_stats() is what's tested.
    leaking_message = _leaking_exception_message()
    broken_collection = Mock()
    broken_collection.count = Mock(side_effect=RuntimeError(leaking_message))
    monkeypatch.setattr(rag_index.vector_index.vectorstore, "_collection", broken_collection)

    send_message_mock = AsyncMock()
    monkeypatch.setattr(start_handler.bot, "send_message", send_message_mock)

    message = SimpleNamespace(from_user=SimpleNamespace(id=9007), chat=SimpleNamespace(id=9007))

    with caplog.at_level(logging.DEBUG):
        await start_handler.cmd_stats(message)

    sent_text = send_message_mock.await_args.args[1]
    assert leaking_message not in sent_text
    assert FAKE_OPENAI_KEY not in sent_text
    assert FAKE_TELEGRAM_TOKEN not in sent_text

    log_text = caplog.text
    assert leaking_message not in log_text
    assert FAKE_OPENAI_KEY not in log_text
    assert "RAG get_stats failed" in log_text
    assert "RuntimeError" in log_text
