"""
Document Upload Handler.
Allows users to upload documents (PDF, TXT, MD, DOCX) for the RAG knowledge base.
"""

import asyncio
import uuid
from telebot import types
from pathlib import Path

from bot import bot
from config import MANAGED_UPLOADS_DIR, MAX_DOCUMENT_SIZE_BYTES
from rag.loader import document_loader, SUPPORTED_EXTENSIONS
from rag.index import vector_index
from utils.logging import logger
from utils.helpers import download_telegram_file, cleanup_file, submit_worker, await_worker
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


def _load_and_index_document(physical_path: Path, display_name: str) -> list:
    """
    Runs document parsing (PyPDFLoader/TextLoader/Docx2txtLoader, CPU/disk
    bound) and Chroma indexing (OpenAIEmbeddings network call + vector-store
    write) as a single blocking unit, so the whole pipeline can be offloaded
    to a worker thread in one `asyncio.to_thread()` call rather than
    thread-hopping per step. No async/Telegram/UserSession work happens
    between these two calls in the original code, so combining them changes
    no ordering or behavior.
    """
    chunks = document_loader.load_document(physical_path, display_name=display_name)
    vector_index.add_documents(chunks)
    return chunks


def _resolve_cancelled_storage(storage_future: "asyncio.Future", user_id: int) -> None:
    """
    Called from `process_document_upload`'s `except asyncio.CancelledError:`
    handler around the storage step, after `await_worker(storage_future)`
    has already blocked until `storage_future` reached a terminal state —
    genuine terminal state of the executor thread itself (see
    `utils.helpers.submit_worker()`), not merely of an asyncio Task wrapping
    it — so it is always safe here to inspect/act on its outcome without
    racing the worker thread.

    Cleans up ONLY a file the worker itself newly created (never a
    bystander); sends no Telegram message (the caller is already being
    cancelled); logs only safe, sanitized metadata (Stage 1D).
    """
    if storage_future.cancelled():
        return
    exc = storage_future.exception()
    if exc is not None:
        # _store_document_exclusively() already cleans up its own partial
        # write before raising — nothing new to delete. Retrieving the
        # exception here just keeps asyncio from ever reporting it as
        # "exception was never retrieved".
        logger.warning(
            "Document upload: cancelled during storage, worker failed | user_id=%s, error_type=%s",
            user_id, type(exc).__name__
        )
        return
    created_path = storage_future.result()
    logger.warning(
        "Document upload: cancelled during storage, removing orphaned file | user_id=%s",
        user_id
    )
    cleanup_file(created_path)


def _resolve_cancelled_indexing(index_future: "asyncio.Future", physical_path: Path, user_id: int) -> None:
    """
    Called from `process_document_upload`'s `except asyncio.CancelledError:`
    handler around the load/index step, after `await_worker(index_future)`
    has already blocked until `index_future` reached a terminal state —
    genuine terminal state of the executor thread itself, not merely of an
    asyncio Task wrapping it — so it is always safe here to touch
    `physical_path`: the parser/indexer worker is guaranteed to be done with
    it, never still reading it.

    Sends no Telegram message; logs only safe, sanitized metadata (Stage 1D).
    """
    if index_future.cancelled():
        cleanup_file(physical_path)
        return
    exc = index_future.exception()
    if exc is not None:
        # Ingestion did not commit — remove the newly-owned upload, same as
        # the non-cancelled failure path.
        logger.warning(
            "Document upload: cancelled during indexing, ingestion did not commit | user_id=%s, error_type=%s",
            user_id, type(exc).__name__
        )
        cleanup_file(physical_path)
        return
    # Indexing succeeded despite cancellation: chunks are already committed
    # into Chroma. Never delete successfully ingested data, and skip the
    # normal success notification — the request itself was cancelled.
    logger.warning(
        "Document upload: cancelled after indexing already committed, retaining file | user_id=%s",
        user_id
    )


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

    Stage 1E.2 cancellation safety: the storage write and the load/index
    pipeline each run via `submit_worker()` (a bare default-executor Future,
    never a Task — see utils/helpers.py), awaited through `await_worker()`.
    If this coroutine is cancelled — even repeatedly, or as part of a
    broad/shutdown-style sweep — the worker is never abandoned:
    `_resolve_cancelled_storage()` / `_resolve_cancelled_indexing()` only
    run once the worker has genuinely reached a terminal state, then the
    original `asyncio.CancelledError` is re-raised. No Telegram message is
    sent while resolving a cancellation.
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
                "Document upload rejected: oversized | user_id=%s, extension=%s, size_bytes=%s",
                user_id, extension, len(file_bytes)
            )
            await bot.send_message(
                message.chat.id,
                f"❌ Файл слишком большой: {len(file_bytes) / 1024 / 1024:.1f} MB\n"
                f"Максимальный размер: {MAX_DOCUMENT_SIZE_BYTES / 1024 / 1024:.0f} MB"
            )
            return

        # Disk write of the downloaded bytes (up to MAX_DOCUMENT_SIZE_BYTES)
        # is blocking I/O — run it off the event loop via submit_worker(),
        # awaited through await_worker(): if this coroutine is cancelled
        # (even repeatedly) while the write is in flight, the worker thread
        # is never abandoned (see _resolve_cancelled_storage()).
        storage_future = submit_worker(_store_document_exclusively, file_bytes, extension)
        try:
            physical_path = await await_worker(storage_future)
        except asyncio.CancelledError:
            _resolve_cancelled_storage(storage_future, user_id)
            raise

        logger.info(
            "Document upload: file saved | user_id=%s, storage_name=%s, size_bytes=%s",
            user_id, physical_path.name, len(file_bytes)
        )
        await bot.send_message(message.chat.id, "📄 Индексирую документ...")
        # Parsing (PDF/DOCX/TXT) + Chroma/embeddings indexing is a blocking
        # pipeline — same submit_worker()/await_worker() pattern: repeated
        # cancellation here must not race the worker for ownership of
        # `physical_path` (see _resolve_cancelled_indexing()).
        index_future = submit_worker(_load_and_index_document, physical_path, original_filename)
        try:
            chunks = await await_worker(index_future)
        except asyncio.CancelledError:
            _resolve_cancelled_indexing(index_future, physical_path, user_id)
            raise
        logger.info("Document indexed | user_id=%s, chunks=%s", user_id, len(chunks))
    except Exception as e:
        # document_loader.load_document() can fail on local PDF/TXT/DOCX
        # parsing, and vector_index.add_documents() reaches Chroma +
        # OpenAIEmbeddings (a network call to OpenAI) — either can surface a
        # provider/HTTP exception, so only the exception's class name is
        # logged here, never its text, a traceback, or the user-controlled
        # original filename.
        logger.error("Document upload failed | user_id=%s, extension=%s, error_type=%s", user_id, extension, type(e).__name__)
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
