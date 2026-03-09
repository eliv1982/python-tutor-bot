"""
Image Handler.
Handles image analysis with GPT-4 Vision using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from services.router import route_image_request
from utils.logging import logger
from utils.helpers import cleanup_file, user_sessions, strip_markdown


@bot.message_handler(content_types=['photo'])
async def handle_photo_message(message: types.Message):
    """Handle photo messages."""
    user_id = message.from_user.id
    caption = message.caption or ""
    logger.info("Photo message | user_id=%s, has_caption=%s", user_id, bool(caption.strip()))
    await bot.send_chat_action(message.chat.id, 'typing')
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"

        if not caption or not caption.strip():
            user_sessions.set_pending_image(user_id, file_url)
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
            image_url=file_url,
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


