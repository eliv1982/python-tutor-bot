"""
Stage 1D.1 regression tests: targeted corrections after independent Codex
audit of Stage 1D.

Covers (see PR/task description for the numbered findings):
1. BLOCKER — central AsyncTeleBot exception boundary (dispatcher-level test,
   not a direct handler-function call)
2. Document-upload loader/indexing failure no longer logs raw exception text
   (the pinned-test carve-out from Stage 1D was removed; see updated
   assertions in tests/test_stage1b_document_upload.py) — this file adds a
   provider/embedding-shaped exception through the same boundary
3. RAG source filenames (rag/query.py) are not logged
4. /start does not log the user's first_name
5. ffmpeg/Pydub conversion exceptions are sanitized
6. /stats and RAG init/index logs no longer carry absolute filesystem paths
7. TTS external-failure privacy + DALL-E non-200 body privacy (test gaps
   Codex identified that Stage 1D's original 8 tests did not cover)

All external calls (Telegram, OpenAI, Chroma/embeddings, ffmpeg subprocess)
are mocked/monkeypatched. No network access, no live ffmpeg/pydub decoding,
and no mutation of the real data/documents, data/documents/uploads,
data/chroma_db, or bot.log paths (same tests/conftest.py isolation as every
other test module). Nothing here weakens tests/test_stage1a_security.py,
tests/test_stage1b_document_upload.py, tests/test_stage1c_access_control.py,
or tests/test_stage1d_privacy_logging.py.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from telebot import types
from telebot.async_telebot import AsyncTeleBot

from utils.helpers import user_sessions

FAKE_OPENAI_KEY = "sk-FAKE1234567890ABCDEFSECRETKEYDONOTUSE"
FAKE_TELEGRAM_TOKEN = "123456789:FAKE-STAGE1D1-TOKEN-FOR-LOG-LEAK-TEST"
FAKE_TELEGRAM_URL = f"https://api.telegram.org/file/bot{FAKE_TELEGRAM_TOKEN}/photos/file_1.jpg"


def _leaking_exception_message(raw_user_text: str = "") -> str:
    parts = [
        f"Authorization: Bearer {FAKE_OPENAI_KEY}",
        f"request to {FAKE_TELEGRAM_URL} failed: 401 Unauthorized",
    ]
    if raw_user_text:
        parts.append(f"payload: {raw_user_text}")
    return " | ".join(parts)


@pytest.fixture(autouse=True)
def _clean_sessions():
    user_sessions.sessions.clear()
    yield
    user_sessions.sessions.clear()


# ---------------------------------------------------------------------------
# 1. BLOCKER — central AsyncTeleBot exception boundary (dispatcher-level)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_level_unhandled_exception_is_sanitized(caplog):
    """
    Exercises the ACTUAL pyTelegramBotAPI async dispatcher path —
    AsyncTeleBot.process_new_messages() -> _process_updates() ->
    _run_middlewares_and_handlers() — not a direct call to a handler
    function. A registered handler raises a synthetic exception carrying a
    fake bot token, a token-bearing Telegram request URL, and distinctive
    raw text (mimicking a real telebot.apihelper.ApiException, which can
    embed the token-bearing request URL). Uses a throwaway AsyncTeleBot
    instance (not the shared production `bot`) so this test never touches
    real handler registrations. No network call is made: process_new_messages
    dispatches a locally constructed message, it never calls getUpdates.
    """
    import bot as bot_module

    test_bot = AsyncTeleBot(
        "111111111:THROWAWAY-TEST-BOT-TOKEN",
        exception_handler=bot_module._SafeDispatcherExceptionHandler(),
    )

    leaking_message = _leaking_exception_message("distinctive-raw-exception-text-9d8e7f")

    @test_bot.message_handler(content_types=["text"])
    async def _boom(message):
        raise RuntimeError(leaking_message)

    message = types.Message.__new__(types.Message)
    message.from_user = types.User(id=42, is_bot=False, first_name="Dispatcher")
    message.chat = types.Chat(id=42, type="private")
    message.text = "trigger"
    message.content_type = "text"

    with caplog.at_level(logging.DEBUG):
        # process_new_messages() must complete without the handler's
        # exception escaping into the test — the dispatcher's own
        # try/except plus our exception_handler absorb it.
        await test_bot.process_new_messages([message])

    log_text = caplog.text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert "api.telegram.org" not in log_text
    assert leaking_message not in log_text
    assert "distinctive-raw-exception-text-9d8e7f" not in log_text
    assert "Traceback" not in log_text
    # Safe, structured diagnostics remain.
    assert "Unhandled dispatcher exception" in log_text
    assert "RuntimeError" in log_text


@pytest.mark.asyncio
async def test_shared_bot_instance_has_exception_handler_installed():
    """Regression guard: the shared production `bot` (bot.py) must have the
    safe exception handler installed, not just a throwaway test instance."""
    import bot as bot_module

    assert bot_module.bot.exception_handler is not None
    assert isinstance(bot_module.bot.exception_handler, bot_module._SafeDispatcherExceptionHandler)


@pytest.mark.asyncio
async def test_dispatcher_exception_handler_does_not_send_telegram_message(monkeypatch, caplog):
    """The central handler must not attempt any Telegram send of its own."""
    import bot as bot_module

    test_bot = AsyncTeleBot(
        "111111111:THROWAWAY-TEST-BOT-TOKEN-2",
        exception_handler=bot_module._SafeDispatcherExceptionHandler(),
    )
    send_message_mock = AsyncMock()
    monkeypatch.setattr(test_bot, "send_message", send_message_mock)

    @test_bot.message_handler(content_types=["text"])
    async def _boom(message):
        raise RuntimeError("boom")

    message = types.Message.__new__(types.Message)
    message.from_user = types.User(id=43, is_bot=False, first_name="NoSend")
    message.chat = types.Chat(id=43, type="private")
    message.text = "trigger"
    message.content_type = "text"

    with caplog.at_level(logging.DEBUG):
        await test_bot.process_new_messages([message])

    send_message_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Document indexing/provider exception through the upload boundary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_indexing_provider_exception_fully_sanitized(monkeypatch, tmp_path, caplog):
    """
    vector_index.reconcile_document() (Stage 2B-F: the call
    _load_and_index_document() now makes, superseding the old direct
    add_documents() call) embeds chunks via OpenAIEmbeddings before
    writing to Qdrant, so a provider/HTTP exception can genuinely surface
    here. Simulates exactly that shape (token/API-key/URL-bearing) and
    proves it never reaches the logs, while cleanup/generic-response
    guarantees from Stage 1B still hold.
    """
    import handlers.document_upload as document_upload

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)

    leaking_message = _leaking_exception_message()
    monkeypatch.setattr(
        document_upload.get_vector_index(), "reconcile_document",
        Mock(side_effect=Exception(leaking_message)),
    )

    async def fake_get_file(file_id):
        return SimpleNamespace(file_path="documents/file_1.pdf")

    monkeypatch.setattr(document_upload.bot, "get_file", fake_get_file)
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"%PDF-1.4 fake"))
    send_message_mock = AsyncMock()
    monkeypatch.setattr(document_upload.bot, "send_message", send_message_mock)

    document = SimpleNamespace(
        file_name="report.pdf", mime_type="application/pdf", file_id="fake-id", file_size=100,
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=9101), chat=SimpleNamespace(id=9101), document=document,
    )

    with caplog.at_level(logging.DEBUG):
        await document_upload.process_document_upload(message, document)

    # Stage 1B cleanup guarantee holds: the newly created file is removed.
    assert list(tmp_path.iterdir()) == []

    sent_text = send_message_mock.await_args.args[1]
    assert leaking_message not in sent_text
    assert FAKE_OPENAI_KEY not in sent_text
    assert "ошибка" in sent_text.lower()

    log_text = caplog.text
    assert leaking_message not in log_text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert "api.telegram.org" not in log_text
    assert "Document upload failed" in log_text
    assert "Exception" in log_text


# ---------------------------------------------------------------------------
# 3. RAG source filenames must not be logged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_source_filename_not_logged_but_attribution_preserved(monkeypatch, caplog):
    """
    A confidential-looking filename stored as RAG source metadata (Stage 1B
    display_name) must reach the user (source attribution is a legitimate,
    intentional feature) but must never appear in operational logs.
    """
    import rag.query as rag_query

    confidential_source = "Ivanov_passport_CONFIDENTIAL.pdf"

    fake_doc = SimpleNamespace(
        metadata={"source": confidential_source},
        page_content="Some retrieved passage.",
    )
    monkeypatch.setattr(
        rag_query.get_vector_index(), "similarity_search_with_score",
        Mock(return_value=[(fake_doc, 0.1)]),
    )

    from services.openai_client import openai_client
    create_mock = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Here is the answer."))],
        usage=None,
    ))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    with caplog.at_level(logging.DEBUG):
        response = await rag_query.query_knowledge_base("What does my document say?")

    # Source attribution is preserved for the user.
    assert confidential_source in response

    # But it never appears in the logs.
    log_text = caplog.text
    assert confidential_source not in log_text
    assert "source_count=1" in log_text


def test_rag_add_document_helper_does_not_log_filename(monkeypatch, caplog, tmp_path):
    """The currently-unused add_document_to_knowledge_base() helper must not
    reintroduce a filename leak if it's ever wired up in the future."""
    import asyncio
    import rag.query as rag_query
    import rag.loader as rag_loader

    confidential_path = tmp_path / "board_meeting_minutes_SECRET.docx"
    confidential_path.write_bytes(b"fake docx bytes")

    # add_document_to_knowledge_base() imports document_loader locally
    # (`from rag.loader import document_loader`) inside the function body,
    # so the module-level singleton on rag.loader is the patch target.
    monkeypatch.setattr(rag_loader.document_loader, "load_document", Mock(return_value=["chunk-a", "chunk-b"]))
    monkeypatch.setattr(rag_query.get_vector_index(), "add_documents", Mock())

    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(rag_query.add_document_to_knowledge_base(str(confidential_path)))

    assert result["success"] is True
    assert "board_meeting_minutes_SECRET" not in caplog.text
    assert "chunks=2" in caplog.text


# ---------------------------------------------------------------------------
# 4. /start must not log first_name
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_command_does_not_log_first_name(monkeypatch, caplog):
    import handlers.start as start_handler

    distinctive_name = "XiomaraQuetzalcoatl9182"
    send_message_mock = AsyncMock()
    monkeypatch.setattr(start_handler.bot, "send_message", send_message_mock)

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=9102, first_name=distinctive_name),
        chat=SimpleNamespace(id=9102),
    )

    with caplog.at_level(logging.DEBUG):
        await start_handler.cmd_start(message)

    # The greeting sent back to this same user still uses their name (UX).
    sent_text = send_message_mock.await_args.args[1]
    assert distinctive_name in sent_text

    # But it is never logged.
    assert distinctive_name not in caplog.text
    assert "Command /start" in caplog.text


# ---------------------------------------------------------------------------
# 5. ffmpeg/Pydub conversion errors must be sanitized
# ---------------------------------------------------------------------------

def test_ogg_to_wav_conversion_failure_leaks_nothing(monkeypatch, tmp_path, caplog):
    """Simulates a pydub/ffmpeg decoding failure whose message embeds a
    confidential path, distinctive media metadata, and a fake sensitive
    token/text — none of it may reach the logs."""
    from pydub import AudioSegment
    import utils.helpers as helpers

    confidential_path_fragment = "C:\\Users\\confidential_employee\\Desktop\\salary_review.ogg"
    fake_ffmpeg_stderr = (
        f"ffmpeg error decoding {confidential_path_fragment}: "
        f"stream #0:0: codec_tag=0x6134706d, bitrate=64000, "
        f"Authorization: Bearer {FAKE_OPENAI_KEY}"
    )

    monkeypatch.setattr(
        AudioSegment, "from_ogg",
        Mock(side_effect=Exception(fake_ffmpeg_stderr)),
    )

    ogg_path = tmp_path / "voice_message.ogg"
    ogg_path.write_bytes(b"OggS fake ogg bytes")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception):
            helpers.convert_ogg_to_wav(ogg_path)

    log_text = caplog.text
    assert fake_ffmpeg_stderr not in log_text
    assert confidential_path_fragment not in log_text
    assert FAKE_OPENAI_KEY not in log_text
    assert "salary_review" not in log_text
    assert "Audio conversion failed" in log_text
    assert "Exception" in log_text


# ---------------------------------------------------------------------------
# 6. No absolute filesystem paths in /stats or RAG init/index logs
# ---------------------------------------------------------------------------

def test_get_stats_never_returns_absolute_path(monkeypatch, tmp_path):
    import rag.index as rag_index

    sensitive_dir = tmp_path / "C_Users_confidential_deploy_user" / "qdrant"
    sensitive_dir.mkdir(parents=True)
    monkeypatch.setattr(rag_index.get_vector_index(), "persist_directory", sensitive_dir)

    fake_count_result = SimpleNamespace(count=3)
    monkeypatch.setattr(rag_index.get_vector_index().client, "count", Mock(return_value=fake_count_result))

    stats = rag_index.get_vector_index().get_stats()

    assert "persist_directory" not in stats
    assert str(sensitive_dir) not in str(stats)
    assert "confidential_deploy_user" not in str(stats)
    assert stats["total_documents"] == 3


@pytest.mark.asyncio
async def test_stats_command_output_never_contains_absolute_path(monkeypatch, tmp_path):
    import handlers.start as start_handler
    import rag.index as rag_index

    sensitive_dir = tmp_path / "C_Users_confidential_deploy_user" / "qdrant"
    sensitive_dir.mkdir(parents=True)
    monkeypatch.setattr(rag_index.get_vector_index(), "persist_directory", sensitive_dir)

    fake_count_result = SimpleNamespace(count=5)
    monkeypatch.setattr(rag_index.get_vector_index().client, "count", Mock(return_value=fake_count_result))

    send_message_mock = AsyncMock()
    monkeypatch.setattr(start_handler.bot, "send_message", send_message_mock)

    message = SimpleNamespace(from_user=SimpleNamespace(id=9103), chat=SimpleNamespace(id=9103))
    await start_handler.cmd_stats(message)

    sent_text = send_message_mock.await_args.args[1]
    assert str(sensitive_dir) not in sent_text
    assert "confidential_deploy_user" not in sent_text
    assert "5" in sent_text


@pytest.mark.asyncio
async def test_rag_init_and_index_logs_have_no_absolute_path(monkeypatch, tmp_path, caplog):
    """rag/index.py's own operational logs (Qdrant client init,
    index_documents_directory) must not carry self.persist_directory."""
    import rag.index as rag_index

    sensitive_dir = tmp_path / "C_Users_confidential_deploy_user" / "qdrant"
    monkeypatch.setattr(rag_index.get_vector_index(), "persist_directory", sensitive_dir)
    # index_documents_directory() now reconciles each file via
    # reconcile_document() (Stage 2B-C Blocker 2) rather than calling
    # add_documents() directly — mock the higher-level method instead so
    # this log-redaction test never reaches the real (production)
    # OpenAIEmbeddings client.
    monkeypatch.setattr(rag_index.get_vector_index(), "reconcile_document", Mock(return_value=("reindexed", 1)))

    docs_root = tmp_path / "C_Users_confidential_deploy_user" / "documents"
    docs_root.mkdir(parents=True)
    (docs_root / "notes.md").write_text("hello", encoding="utf-8")

    with caplog.at_level(logging.DEBUG):
        # reference_filenames=None: this test exercises generic
        # directory-scan log-redaction, not the BUILTIN_REFERENCE_FILES
        # manifest gate — "notes.md" isn't one of the real manifest names.
        rag_index.get_vector_index().index_documents_directory(directory=docs_root, reference_filenames=None)

    assert "confidential_deploy_user" not in caplog.text
    assert str(sensitive_dir) not in caplog.text


# ---------------------------------------------------------------------------
# 7. Additional Stage 1D test gaps: TTS external failure, DALL-E non-200 body
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_external_failure_leaks_nothing(monkeypatch, caplog):
    from services import tts
    from services.openai_client import openai_client

    leaking_message = _leaking_exception_message("my private diary text")
    monkeypatch.setattr(
        openai_client.client.audio.speech, "create",
        AsyncMock(side_effect=Exception(leaking_message)),
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception):
            await tts.generate_voice_response("some safe response text", voice="alloy")

    log_text = caplog.text
    assert leaking_message not in log_text
    assert FAKE_OPENAI_KEY not in log_text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert "my private diary text" not in log_text
    assert "TTS failed" in log_text
    assert "Exception" in log_text


@pytest.mark.asyncio
async def test_tts_external_failure_generic_telegram_response(monkeypatch, caplog):
    """End-to-end: a TTS failure inside the voice-mode text flow must reach
    the user only as a generic message."""
    import handlers.text as text_handler
    from services.openai_client import openai_client
    from utils.helpers import user_sessions as sessions

    user_id = 9104
    sessions.set_mode(user_id, "voice")

    create_mock = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A short Python tip."))],
        usage=None,
    ))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    leaking_message = _leaking_exception_message()
    monkeypatch.setattr(
        openai_client.client.audio.speech, "create",
        AsyncMock(side_effect=Exception(leaking_message)),
    )

    monkeypatch.setattr(text_handler.bot, "send_chat_action", AsyncMock())
    send_message_mock = AsyncMock()
    monkeypatch.setattr(text_handler.bot, "send_message", send_message_mock)

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=user_id),
        text="Tell me a Python tip", content_type="text",
    )

    with caplog.at_level(logging.DEBUG):
        await text_handler.handle_text_message(message)

    sent_text = send_message_mock.await_args.args[1]
    assert leaking_message not in sent_text
    assert FAKE_OPENAI_KEY not in sent_text
    assert "ошибка" in sent_text.lower()

    log_text = caplog.text
    assert leaking_message not in log_text
    assert FAKE_OPENAI_KEY not in log_text


@pytest.mark.asyncio
async def test_dalle_non_200_body_not_logged_or_surfaced(monkeypatch, tmp_path, caplog):
    """A DALL-E non-200 response body containing a fake key/token/
    confidential payload must not be logged or surfaced raw."""
    from services import image_generation

    monkeypatch.setattr(image_generation, "GENERATED_IMAGES_DIR", tmp_path)

    leaking_body = (
        f'{{"error": {{"message": "Invalid Authorization: Bearer {FAKE_OPENAI_KEY}, '
        f'confidential_customer_note: my SSN is 123-45-6789"}}}}'
    )

    class FakeResponse:
        status = 401

        async def json(self):
            return {}

        async def text(self):
            return leaking_body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeSession:
        def post(self, url, headers=None, json=None):
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(image_generation.aiohttp, "ClientSession", lambda: FakeSession())

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as exc_info:
            await image_generation.generate_image(prompt="a friendly robot")

    # The exception itself (which route_image_generation_request only ever
    # logs the type of, never str()) must not carry the leaked body either.
    assert leaking_body not in str(exc_info.value)
    assert FAKE_OPENAI_KEY not in str(exc_info.value)

    log_text = caplog.text
    assert leaking_body not in log_text
    assert FAKE_OPENAI_KEY not in log_text
    assert "123-45-6789" not in log_text
    assert "DALL-E API error" in log_text
    assert "status=401" in log_text
