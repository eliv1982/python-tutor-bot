"""
Document Upload Handler.
Allows users to upload documents (PDF, TXT, MD, DOCX) for the RAG knowledge base.
"""

from telebot import types
from pathlib import Path

from bot import bot
from config import DOCUMENTS_DIR
from rag.loader import document_loader
from rag.index import vector_index
from utils.logging import logger
from utils.helpers import download_telegram_file


# Supported MIME types for RAG
SUPPORTED_DOC_MIMES = [
    'application/pdf',
    'text/plain',
    'text/markdown',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
]


@bot.message_handler(content_types=['document'])
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


async def process_document_upload(message: types.Message, document: types.Document):
    """Process document upload for RAG."""
    user_id = message.from_user.id
    
    # Check file type
    if document.mime_type not in SUPPORTED_DOC_MIMES:
        await bot.send_message(
            message.chat.id,
            f"❌ Неподдерживаемый тип файла: {document.mime_type}\n\n"
            "Поддерживаемые форматы: PDF, TXT, MD, DOCX."
        )
        return
    
    # Check file size (max 20 MB)
    max_size = 20 * 1024 * 1024  # 20 MB
    if document.file_size > max_size:
        await bot.send_message(
            message.chat.id,
            f"❌ Файл слишком большой: {document.file_size / 1024 / 1024:.1f} MB\n"
            f"Максимальный размер: 20 MB"
        )
        return
    
    try:
        await bot.send_message(message.chat.id, "⏳ Загружаю документ...")
        
        # Download file
        file_bytes, _ = await download_telegram_file(bot, document.file_id, operation="document_download")
        file_path = DOCUMENTS_DIR / document.file_name
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        logger.info("Document upload: file saved | user_id=%s, file=%s, size_bytes=%s", user_id, document.file_name, document.file_size)
        await bot.send_message(message.chat.id, "📄 Индексирую документ...")
        chunks = document_loader.load_document(file_path)
        vector_index.add_documents(chunks)
        logger.info("Document indexed | user_id=%s, file=%s, chunks=%s", user_id, document.file_name, len(chunks))
        
        # Success message
        await bot.send_message(
            message.chat.id,
            f"✅ Документ успешно загружен!\n\n"
            f"📄 Файл: {document.file_name}\n"
            f"📊 Фрагментов: {len(chunks)}\n"
            f"💾 Размер: {document.file_size / 1024:.1f} KB\n\n"
            f"Теперь вы можете задавать вопросы по этому документу:\n"
            f"/mode rag"
        )
        
    except Exception as e:
        logger.error("Document upload failed | user_id=%s, file=%s, error=%s", user_id, document.file_name, e, exc_info=True)
        await bot.send_message(
            message.chat.id,
            f"❌ Ошибка при загрузке документа:\n{str(e)}\n\n"
            "Попробуйте загрузить файл вручную в папку data/documents/"
        )
