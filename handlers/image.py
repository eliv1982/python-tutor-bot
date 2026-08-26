"""
Image Handler.
Handles image analysis with GPT-4 Vision using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from services.router import route_image_request
from services.vision import encode_image_bytes_to_data_url
from utils.logging import logger
from utils.helpers import cleanup_file, user_sessions, strip_markdown, download_telegram_file
from utils.access_control import require_authorized
from config import MAX_TELEGRAM_IMAGE_BYTES


@bot.message_handler(content_types=['photo'])
@require_authorized
async def handle_photo_message(message: types.Message):
    """Handle photo messages."""
    user_id = message.from_user.id
    caption = message.caption or ""
    logger.info("Photo message | user_id=%s, has_caption=%s", user_id, bool(caption.strip()))
    await bot.send_chat_action(message.chat.id, 'typing')
    try:
        photo = message.photo[-1]
        # Downloaded via the bot's own token-scoped call so the token never
        # has to be embedded in a URL that gets passed to another service.
        file_bytes, file_path = await download_telegram_file(bot, photo.file_id, operation="photo_download")

        if len(file_bytes) > MAX_TELEGRAM_IMAGE_BYTES:
            logger.warning(
                "Photo rejected: too large | user_id=%s, size_bytes=%s, limit_bytes=%s",
                user_id, len(file_bytes), MAX_TELEGRAM_IMAGE_BYTES
            )
            await bot.send_message(
                message.chat.id,
                f"❌ Изображение слишком большое ({len(file_bytes) / 1024 / 1024:.1f} MB).\n"
                f"Максимальный размер: {MAX_TELEGRAM_IMAGE_BYTES / 1024 / 1024:.0f} MB."
            )
            return

        image_data_url = encode_image_bytes_to_data_url(file_bytes, filename_hint=file_path)

        if not caption or not caption.strip():
            user_sessions.set_pending_image(user_id, image_data_url)
            logger.info("Photo without caption: asking user for question | user_id=%s", user_id)
            await bot.send_message(
                message.chat.id,
                "📸 Изображение получено. Что именно нужно извлечь или проанализировать?\n\n"
                "Напиши следующим сообщением вопрос или задание, например:\n"
                "• «Опиши, что на изображении»\n"
                "• «Найди ошибку в этом коде»\n"
                "• «Переведи текст с картинки»"
            )
            return

        await bot.send_message(
            message.chat.id,
            f"📸 Анализирую изображение с вопросом: {caption.strip()}"
        )
        logger.debug("Photo with caption: calling Vision | user_id=%s, caption_len=%s", user_id, len(caption))
        response = await route_image_request(
            user_id=user_id,
            image_url=image_data_url,
            caption=caption.strip()
        )
        logger.info("Photo analyzed | user_id=%s, response_len=%s", user_id, len(response.get("text", "")))
        await bot.send_message(
            message.chat.id,
            f"🔍 Анализ изображения:\n\n{strip_markdown(response['text'])}"
        )
    except Exception as e:
        logger.error("Photo handler failed | user_id=%s, error=%s", user_id, e, exc_info=True)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при анализе изображения.\n"
            "Попробуйте отправить другое изображение."
        )


