"""
Start and Help Command Handlers.
Handles /start and /help commands using pyTelegramBotAPI.
"""

import asyncio

from telebot import types
from bot import bot
from utils.logging import logger
from utils.helpers import user_sessions
from utils.access_control import require_authorized
from config import BotMode, DEFAULT_MODE


@bot.message_handler(commands=['start'])
@require_authorized
async def cmd_start(message: types.Message):
    """Handle /start command."""
    user_id = message.from_user.id
    # first_name is user-controlled personal data — used in the greeting
    # text sent back to this same user below, but never logged.
    user_name = message.from_user.first_name
    logger.info("Command /start | user_id=%s", user_id)
    
    # Initialize user session
    user_sessions.set_mode(user_id, DEFAULT_MODE)
    
    welcome_text = f"""👋 Привет, {user_name}!

Я — персональный тьютор по Python с мультимодальным функционалом:

🔤 Текст — объяснения, примеры кода, ответы на вопросы по Python
🎤 Голос — отправь голосовое сообщение, получи ответ голосом и текстом
📸 Изображения — анализ скриншотов кода, диаграмм, ошибок
📚 База знаний (RAG) — ответы по твоим материалам (учебники, конспекты)

Команды: /help · /mode · /voice · /reset · /stats

Режимы: /mode text · /mode voice · /mode rag · /mode vision

Начни с вопроса по Python или переключи режим — помогу с учёбой! 🐍"""
    
    await bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(commands=['help'])
@require_authorized
async def cmd_help(message: types.Message):
    """Handle /help command."""
    user_id = message.from_user.id
    logger.info("Command /help | user_id=%s", user_id)
    
    help_text = """📖 Personal Python Tutor — справка

Режимы (команда /mode):
• /mode text — текстовый диалог по Python
• /mode voice — голосовые ответы: отправь голос → Whisper → ответ голосом и текстом
• /mode rag — ответы по базе знаний (твои PDF/TXT/MD/DOCX)
• /mode vision — анализ изображений (скриншоты кода, ошибки, схемы)

Команды:
/mode <text|voice|rag|vision> — сменить режим
/voice <alloy|echo|nova|fable|onyx|shimmer> — голос для TTS
/voices — список голосов
/reset — очистить историю
/stats — статистика базы знаний

Примеры:
• «Объясни list comprehension»
• [Голос] «В чём разница между list и tuple?»
• [Фото ошибки] «Почему падает этот код?»
• В RAG: загрузи конспект → спроси по нему

Стек: LLM (Anthropic/OpenAI), Whisper, TTS, Vision, Qdrant (RAG)."""
    
    await bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=['reset'])
@require_authorized
async def cmd_reset(message: types.Message):
    """Handle /reset command - clear conversation history."""
    user_id = message.from_user.id
    user_sessions.clear_history(user_id)
    user_sessions.clear_pending_image(user_id)
    logger.info("Command /reset | user_id=%s, history_cleared=True, pending_image_cleared=True", user_id)
    
    await bot.send_message(
        message.chat.id,
        "✅ История диалога очищена!\n\n"
        "Начнем с чистого листа. Чем могу помочь?"
    )


@bot.message_handler(commands=['stats'])
@require_authorized
async def cmd_stats(message: types.Message):
    """Handle /stats command - show knowledge base statistics."""
    user_id = message.from_user.id
    logger.info("Command /stats | user_id=%s", user_id)
    try:
        from rag.query import get_knowledge_base_stats
        # get_knowledge_base_stats() reaches VectorIndex's shared lock,
        # which a concurrent upload/RAG query may hold for a while — call it
        # off the event loop so /stats can't stall on that contention.
        #
        # Stage 1E.1 cancellation review: same reasoning as rag/query.py's
        # similarity search — deliberately unshielded. It's a read-only
        # `collection.count()`, mutates nothing, and its result is simply
        # discarded if the caller is cancelled — no cleanup/ownership race.
        #
        # Stage 3A: scoped to this user — reference corpus + this user's
        # own private documents only, never a global count that would
        # reveal another user's private upload activity.
        stats = await asyncio.to_thread(get_knowledge_base_stats, user_id)
        logger.debug("Command /stats | stats=%s", stats)
        if "error" in stats:
            await bot.send_message(
                message.chat.id,
                f"⚠️ Ошибка получения статистики:\n{stats['error']}"
            )
            return
        
        total_docs = stats.get("total_documents", 0)

        stats_text = f"""📊 Статистика базы знаний

📄 Документов в индексе: {total_docs}

{"✅ База знаний готова к использованию!" if total_docs > 0 else "⚠️ База знаний пуста. Добавьте документы в data/documents/"}

Используйте /mode rag для работы с базой знаний."""
        
        await bot.send_message(message.chat.id, stats_text)
        
    except Exception as e:
        logger.error("Command /stats failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        await bot.send_message(
            message.chat.id,
            "⚠️ Ошибка получения статистики базы знаний."
        )
