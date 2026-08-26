"""
Document Upload Handler.
Allows users to upload documents (PDF, TXT, MD, DOCX) for the RAG knowledge base.
"""

import uuid
from telebot import types
from pathlib import Path

from bot import bot
from config import MANAGED_UPLOADS_DIR, MAX_DOCUMENT_SIZE_BYTES
from rag.loader import document_loader, SUPPORTED_EXTENSIONS
from rag.index import vector_index
from utils.logging import logger
from utils.helpers import download_telegram_file, cleanup_file
from utils.access_control import require_authorized


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
    logger.info("Document received | user_id=%s, file_name=%s, mime=%s, size=%s", user_id, document.file_name, document.mime_type, getattr(document, "file_size", None))
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


def _store_document_exclusively(file_bytes: bytes, extension: str, attempts: int = 5) -> Path:
    """
    Atomically claim a fresh, opaque, application-generated storage path and
    write the document bytes into it in the same exclusive-create operation.

    Each candidate is opened with 'xb' (O_CREAT | O_EXCL). Ownership
    boundary: a FileExistsError from that open call means nothing was
    created, so it's safe to just retry with a new UUID without touching
    whatever already occupies that path. Once the open call itself
    succeeds, this call is the sole owner of `candidate` — no other upload
    could have created it — so any later failure (write or close) triggers
    best-effort cleanup of exactly that file before the original exception
    is re-raised, never masked by a cleanup failure.
    """
    MANAGED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    last_collision_error = None
    for _ in range(attempts):
        candidate = MANAGED_UPLOADS_DIR / f"{uuid.uuid4().hex}{extension}"
        try:
            handle = open(candidate, 'xb')
        except FileExistsError as e:
            last_collision_error = e
            continue

        try:
            with handle:
                handle.write(file_bytes)
        except Exception:
            # Ownership of `candidate` was established by the open() above,
            # so cleaning it up here can never remove another attempt's
            # file. cleanup_file() already swallows its own errors, so this
            # cannot mask the original write/close exception re-raised
            # below.
            cleanup_file(candidate)
            raise

        return candidate

    raise RuntimeError("Could not allocate a unique document storage path") from last_collision_error


async def process_document_upload(message: types.Message, document: types.Document):
    """
    Process document upload for RAG.

    The user-controlled original filename is treated as metadata only: it
    is validated for its extension and preserved for display/source
    attribution, but it never determines the physical storage path. The
    physical path is always an exclusively-created
    `MANAGED_UPLOADS_DIR / f"{uuid4().hex}{extension}"`, where `extension`
    has already been checked against the fixed SUPPORTED_EXTENSIONS set —
    so the resulting path can never contain a path separator or `..`
    regardless of what the original filename was.

    Ingestion (download -> store -> load -> index) and the final success
    notification are deliberately separate lifecycle phases: once
    `vector_index.add_documents()` returns without raising, the upload is
    considered successfully ingested, and nothing after that point —
    including a failure to send the confirmation message — rolls back or
    deletes the stored file, or reports the upload as failed.
    """
    user_id = message.from_user.id
    original_filename = document.file_name or "document"
    extension = Path(original_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        logger.debug("Document upload: unsupported extension | user_id=%s, original_name=%s, extension=%s", user_id, original_filename, extension)
        await bot.send_message(
            message.chat.id,
            f"❌ Неподдерживаемое расширение файла: {extension or '(нет расширения)'}\n\n"
            "Поддерживаемые форматы: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
        return

    physical_path = None
    chunks = None
    file_bytes = b""
    try:
        await bot.send_message(message.chat.id, "⏳ Загружаю документ...")

        # Download file
        file_bytes, _ = await download_telegram_file(bot, document.file_id, operation="document_download")

        # Enforce the size limit against the bytes actually downloaded,
        # not Telegram-reported metadata, before any disk write/parsing.
        if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
            logger.warning(
                "Document upload rejected: oversized | user_id=%s, original_name=%s, size_bytes=%s",
                user_id, original_filename, len(file_bytes)
            )
            await bot.send_message(
                message.chat.id,
                f"❌ Файл слишком большой: {len(file_bytes) / 1024 / 1024:.1f} MB\n"
                f"Максимальный размер: {MAX_DOCUMENT_SIZE_BYTES / 1024 / 1024:.0f} MB"
            )
            return

        physical_path = _store_document_exclusively(file_bytes, extension)

        logger.info(
            "Document upload: file saved | user_id=%s, original_name=%s, storage_name=%s, size_bytes=%s",
            user_id, original_filename, physical_path.name, len(file_bytes)
        )
        await bot.send_message(message.chat.id, "📄 Индексирую документ...")
        chunks = document_loader.load_document(physical_path, display_name=original_filename)
        vector_index.add_documents(chunks)
        logger.info("Document indexed | user_id=%s, original_name=%s, chunks=%s", user_id, original_filename, len(chunks))
    except Exception as e:
        logger.error("Document upload failed | user_id=%s, original_name=%s, error=%s", user_id, original_filename, e, exc_info=True)
        cleanup_file(physical_path)
        await bot.send_message(
            message.chat.id,
            "❌ Ошибка при загрузке документа. Попробуйте ещё раз позже."
        )
        return

    # Ingestion has already committed (file stored, chunks indexed into
    # Chroma). Everything below is best-effort notification only — a
    # failure here must not roll back storage or claim the upload failed.
    try:
        await bot.send_message(
            message.chat.id,
            f"✅ Документ успешно загружен!\n\n"
            f"📄 Файл: {original_filename}\n"
            f"📊 Фрагментов: {len(chunks)}\n"
            f"💾 Размер: {len(file_bytes) / 1024:.1f} KB\n\n"
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
