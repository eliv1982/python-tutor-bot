"""
Text Message Handler.
Handles regular text messages from users using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from services.router import route_text_request, route_image_request
from utils.logging import logger
from utils.helpers import user_sessions, strip_markdown
from utils.access_control import require_authorized
from config import BotMode


def _get_mode_keyboard():
    """Клавиатура выбора режима (по нажатию — переключение)."""
    return types.InlineKeyboardMarkup(row_width=2).row(
        types.InlineKeyboardButton("📝 Text", callback_data="mode_text"),
        types.InlineKeyboardButton("🎤 Voice", callback_data="mode_voice"),
    ).row(
        types.InlineKeyboardButton("📸 Vision", callback_data="mode_vision"),
        types.InlineKeyboardButton("📚 RAG", callback_data="mode_rag"),
    )


@bot.callback_query_handler(func=lambda c: c.data and c.data.startswith("mode_"))
@require_authorized
async def callback_mode(callback: types.CallbackQuery):
    """Обработка нажатия кнопки режима — автоматическое переключение."""
    user_id = callback.from_user.id
    mode = callback.data.replace("mode_", "")
    valid = [BotMode.TEXT, BotMode.VOICE, BotMode.VISION, BotMode.RAG]
    logger.info("Callback mode | user_id=%s, callback_data=%s, mode=%s", user_id, callback.data, mode)
    if mode not in valid:
        logger.warning("Callback mode invalid | user_id=%s, mode=%s", user_id, mode)
        await bot.answer_callback_query(callback.id, "Неизвестный режим")
        return
    user_sessions.set_mode(user_id, mode)
    logger.info("Mode switched (button) | user_id=%s, new_mode=%s", user_id, mode)
    descriptions = {
        BotMode.TEXT: "📝 Текстовый режим — диалог по Python с GPT-4o",
        BotMode.VOICE: "🎤 Голосовой режим — ответы голосом и текстом",
        BotMode.VISION: "📸 Режим Vision — анализ изображений (код, ошибки)",
        BotMode.RAG: "📚 Режим RAG — ответы по базе знаний (документы)",
    }
    await bot.answer_callback_query(callback.id)
    await bot.send_message(
        callback.message.chat.id,
        f"✅ Режим изменён!\n\n{descriptions[mode]}",
    )


@bot.message_handler(commands=['mode'])
@require_authorized
async def cmd_mode(message: types.Message):
    """Handle /mode command — показать текущий режим и кнопки выбора."""
    user_id = message.from_user.id
    args = message.text.split(maxsplit=1)
    
    if len(args) >= 2:
        # /mode text и т.д. — переключение из текста
        new_mode = args[1].lower()
        valid_modes = [BotMode.TEXT, BotMode.VOICE, BotMode.VISION, BotMode.RAG]
        if new_mode not in valid_modes:
            await bot.send_message(
                message.chat.id,
                f"❌ Неизвестный режим: {new_mode}\n\nДоступные: text, voice, vision, rag",
            )
            return
        user_sessions.set_mode(user_id, new_mode)
        logger.info("Mode switched (command) | user_id=%s, new_mode=%s", user_id, new_mode)
        mode_descriptions = {
            BotMode.TEXT: "📝 Текстовый режим — диалог по Python с GPT-4o",
            BotMode.VOICE: "🎤 Голосовой режим — ответы голосом и текстом",
            BotMode.VISION: "📸 Режим Vision — анализ изображений (код, ошибки)",
            BotMode.RAG: "📚 Режим RAG — ответы по базе знаний (документы)",
        }
        await bot.send_message(
            message.chat.id,
            f"✅ Режим изменён!\n\n{mode_descriptions[new_mode]}",
        )
        return
    
    current_mode = user_sessions.get_mode(user_id)
    logger.debug("Command /mode (no arg) | user_id=%s, current_mode=%s, showing keyboard", user_id, current_mode)
    mode_info = (
        f"🔧 Текущий режим: {current_mode}\n\n"
        "Выберите режим кнопкой (переключение сразу):"
    )
    await bot.send_message(
        message.chat.id,
        mode_info,
        reply_markup=_get_mode_keyboard(),
    )


@bot.message_handler(commands=['image'])
@require_authorized
async def cmd_image(message: types.Message):
    """Handle /image command - generate image with specific parameters."""
    user_id = message.from_user.id
    
    # Parse command arguments
    args = message.text.split(maxsplit=1)
    
    if len(args) < 2:
        help_text = """🎨 Генерация изображений

Автоматическая генерация: напишите "Нарисуй...", "Создай изображение..." — ИИ создаст картинку.

Примеры:
• Нарисуй кота в космосе
• Создай изображение футуристического города

Прямая команда: /image <описание>
Бот использует DALL-E 3."""
        
        await bot.send_message(message.chat.id, help_text)
        return
    
    prompt = args[1]
    logger.info("Command /image | user_id=%s, prompt_len=%s", user_id, len(prompt))
    
    # Show typing indicator
    await bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Generate image directly
        from services.router import route_image_generation_request
        from utils.helpers import cleanup_file
        
        response = await route_image_generation_request(
            user_id=user_id,
            prompt=prompt,
            original_text=prompt
        )
        
        # Send text response
        await bot.send_message(message.chat.id, strip_markdown(response["text"]))
        
        # Send image if generated successfully
        if response.get('has_image') and response.get('image_path'):
            await bot.send_chat_action(message.chat.id, 'upload_photo')
            image_path = response['image_path']
            try:
                with open(image_path, 'rb') as photo:
                    await bot.send_photo(message.chat.id, photo)
            finally:
                cleanup_file(image_path)
    
    except Exception as e:
        logger.error("Command /image failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при генерации изображения.\n"
            "Попробуйте еще раз или перефразируйте запрос."
        )


@bot.message_handler(func=lambda message: message.content_type == 'text' and not message.text.startswith('/'))
@require_authorized
async def handle_text_message(message: types.Message):
    """Handle regular text messages. Если есть ожидающее изображение — это вопрос к нему."""
    user_id = message.from_user.id
    text = message.text.strip()
    logger.info("Text message | user_id=%s, text_len=%s", user_id, len(text))

    pending_image_data_url = user_sessions.get_pending_image(user_id)
    if pending_image_data_url:
        logger.info("Text as image follow-up | user_id=%s, caption_len=%s", user_id, len(text))
        user_sessions.clear_pending_image(user_id)
        await bot.send_chat_action(message.chat.id, 'typing')
        try:
            response = await route_image_request(
                user_id=user_id,
                image_url=pending_image_data_url,
                caption=text
            )
            await bot.send_message(
                message.chat.id,
                f"🔍 Анализ изображения:\n\n{strip_markdown(response['text'])}"
            )
        except Exception as e:
            # Wraps Telegram send calls (token-bearing request URL on HTTP
            # failure) alongside router/OpenAI calls — never log raw text.
            logger.error("Image follow-up failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
            await bot.send_message(
                message.chat.id,
                "❌ Ошибка при анализе изображения. Попробуйте отправить изображение с подписью."
            )
        return

    await bot.send_chat_action(message.chat.id, 'typing')
    try:
        response = await route_text_request(user_id, text)
        
        if response.get('has_image') and response.get('image_path'):
            logger.debug("Text response includes generated image | user_id=%s", user_id)
            # Send text response first
            await bot.send_message(message.chat.id, strip_markdown(response["text"]))
            
            # Then send the generated image
            from utils.helpers import cleanup_file
            image_path = response['image_path']
            
            try:
                await bot.send_chat_action(message.chat.id, 'upload_photo')
                with open(image_path, 'rb') as photo:
                    await bot.send_photo(message.chat.id, photo)
                logger.debug("Image sent | user_id=%s", user_id)
                
            finally:
                # Cleanup generated image file
                cleanup_file(image_path)
            
            return
        
        mode = user_sessions.get_mode(user_id)
        logger.debug("Text response mode | user_id=%s, mode=%s, response_len=%s", user_id, mode, len(response.get("text", "")))
        if mode == BotMode.VOICE:
            # Generate voice response
            from services.tts import generate_voice_response
            from utils.helpers import cleanup_file
            
            voice_path = await generate_voice_response(
                strip_markdown(response["text"]),
                voice=user_sessions.get_voice(user_id)
            )
            
            try:
                # Send text first
                await bot.send_message(message.chat.id, strip_markdown(response["text"]))
                
                # Then send voice
                with open(voice_path, 'rb') as audio:
                    await bot.send_voice(message.chat.id, audio)
                
            finally:
                # Cleanup
                cleanup_file(voice_path)
        else:
            # Send text response
            await bot.send_message(message.chat.id, strip_markdown(response["text"]))
    
    except Exception as e:
        # Wraps Telegram send calls (token-bearing request URL on HTTP
        # failure) alongside router/OpenAI/TTS calls — never log raw text.
        logger.error("Text message handler failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при обработке сообщения.\n"
            "Попробуйте еще раз или используйте /reset для сброса."
        )
