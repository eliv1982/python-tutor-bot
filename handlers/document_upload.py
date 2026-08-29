"""
Document Upload Handler.
Allows users to upload documents (PDF, TXT, MD, DOCX) for the RAG knowledge base.

Stage 5B: this module is now a thin Telegram adapter. The hardened
storage/index transaction (validation, managed storage, secure read,
hashing, parsing/indexing, rollback/cleanup, cancellation resolution) has
moved to app/documents.py — an adapter-independent module reachable by a
future FastAPI adapter without importing telebot or constructing a
Telegram bot instance. This module's own responsibility is now limited to
genuinely Telegram-specific concerns: routing by MIME type, downloading
the file via the bot's own token-scoped call, sending progress/result
messages, and translating app.documents.DocumentIngestResult back into a
Telegram response.
"""

from telebot import types
from pathlib import Path

from bot import bot
from config import MAX_DOCUMENT_SIZE_BYTES
from rag.loader import SUPPORTED_EXTENSIONS
from utils.logging import logger
from utils.helpers import download_telegram_file
from utils.access_control import require_authorized
from app import documents as document_pipeline


# Supported MIME types for RAG (early routing only — the authoritative
# format gate is the SUPPORTED_EXTENSIONS check in process_document_upload).
SUPPORTED_DOC_MIMES = [
    'application/pdf',
    'text/plain',
    'text/markdown',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
]


@bot.message_handler(content_types=['document'])
@require_authorized
async def handle_document_message(message: types.Message):
    """Route document messages: RAG upload for PDF/TXT/MD/DOCX, info for images."""
    document = message.document
    user_id = message.from_user.id
    # document.file_name is a user-controlled Telegram display filename and
    # may carry personal/confidential information — never logged raw.
    logger.info("Document received | user_id=%s, mime=%s, size=%s", user_id, document.mime_type, getattr(document, "file_size", None))
    if not document.mime_type:
        logger.warning("Document: unknown mime_type | user_id=%s", user_id)
        await bot.send_message(message.chat.id, "❌ Не удалось определить тип файла.")
        return
    if document.mime_type.startswith("image/"):
        await bot.send_message(
            message.chat.id,
            "📸 Отправьте изображение как фото для анализа (не как документ)."
        )
        return
    if document.mime_type in SUPPORTED_DOC_MIMES:
        logger.debug("Document: supported type, processing upload | user_id=%s", user_id)
        await process_document_upload(message, document)
        return
    logger.debug("Document: unsupported mime | user_id=%s, mime=%s", user_id, document.mime_type)
    await bot.send_message(
        message.chat.id,
        f"ℹ️ Формат не поддерживается: {document.mime_type}\n\n"
        "Поддерживаются: PDF, TXT, MD, DOCX — для базы знаний."
    )


async def process_document_upload(message: types.Message, document: types.Document):
    """
    Telegram adapter around app.documents.ingest_document(): downloads the
    file via Telegram, delegates the entire hardened storage/index
    transaction to the application layer, and translates the returned
    DocumentIngestResult into Telegram messages.

    The extension pre-check below stays here (rather than solely inside
    the transaction) so an unsupported extension is rejected WITHOUT ever
    downloading the file — matching the original behavior exactly.
    ingest_document() re-validates the extension itself too, so this is
    defense in depth, not the only gate: a future adapter that skips this
    pre-check still gets a safe, structured rejection instead of an
    unvalidated upload attempt.

    Ingestion (download -> store+sidecar -> load -> index) and the final
    success notification remain separate lifecycle phases: once
    ingest_document() reports success, the upload is considered committed,
    and nothing after that point — including a failure to send the
    confirmation message — rolls back or reports the upload as failed.

    Cancellation safety (Stage 1E/2B-C): the storage+sidecar write and the
    load/index pipeline both run inside ingest_document(), which resolves
    any cancellation landing during or after that work (including at the
    "Индексирую документ..." progress message, sent here via the
    before_indexing hook) before re-raising — see app/documents.py. A
    cancellation during the download step above has nothing durable to
    clean up yet, so it simply propagates.
    """
    user_id = message.from_user.id
    original_filename = document.file_name or "document"
    extension = Path(original_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        logger.debug("Document upload: unsupported extension | user_id=%s, extension=%s", user_id, extension)
        await bot.send_message(
            message.chat.id,
            f"❌ Неподдерживаемое расширение файла: {extension or '(нет расширения)'}\n\n"
            "Поддерживаемые форматы: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
        return

    try:
        await bot.send_message(message.chat.id, "⏳ Загружаю документ...")
        file_bytes, _ = await download_telegram_file(bot, document.file_id, operation="document_download")
    except Exception as e:
        # Wraps Stage 1A's download_telegram_file() (already privacy-safe
        # on its own) and this handler's own Telegram sends — never log
        # raw exception text.
        logger.error("Document upload failed | user_id=%s, extension=%s, error_type=%s", user_id, extension, type(e).__name__)
        await bot.send_message(
            message.chat.id,
            "❌ Ошибка при загрузке документа. Попробуйте ещё раз позже."
        )
        return

    async def _notify_indexing_started() -> None:
        await bot.send_message(message.chat.id, "📄 Индексирую документ...")

    result = await document_pipeline.ingest_document(
        file_bytes=file_bytes,
        extension=extension,
        display_name=original_filename,
        owner_user_id=user_id,
        before_indexing=_notify_indexing_started,
    )

    if not result.success:
        if result.rejected_reason == "oversized":
            logger.warning(
                "Document upload rejected: oversized | user_id=%s, extension=%s, size_bytes=%s",
                user_id, extension, result.file_size_bytes
            )
            await bot.send_message(
                message.chat.id,
                f"❌ Файл слишком большой: {result.file_size_bytes / 1024 / 1024:.1f} MB\n"
                f"Максимальный размер: {MAX_DOCUMENT_SIZE_BYTES / 1024 / 1024:.0f} MB"
            )
            return
        # Storage/indexing failure: already logged and rolled back inside
        # ingest_document() — only a generic message here, same as before
        # this extraction.
        await bot.send_message(
            message.chat.id,
            "❌ Ошибка при загрузке документа. Попробуйте ещё раз позже."
        )
        return

    try:
        await bot.send_message(
            message.chat.id,
            f"✅ Документ успешно загружен!\n\n"
            f"📄 Файл: {original_filename}\n"
            f"📊 Фрагментов: {result.chunk_count}\n"
            f"💾 Размер: {result.file_size_bytes / 1024:.1f} KB\n\n"
            f"Теперь вы можете задавать вопросы по этому документу:\n"
            f"/mode rag"
        )
    except Exception as e:
        # pyTelegramBotAPI HTTP exceptions can carry request metadata that
        # includes the token-bearing Telegram API URL, so — like Stage 1A's
        # download_telegram_file() — only a fixed operation name, the
        # user_id, and the exception's type are ever logged here: never
        # str(e), never exc_info=True.
        logger.error(
            "Document upload: success notification failed (ingestion already committed) | user_id=%s, error_type=%s",
            user_id, type(e).__name__
        )
