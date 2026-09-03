"""
Start and Help Command Handlers.
Handles /start and /help commands using pyTelegramBotAPI.
"""

import asyncio

from telebot import types
from bot import bot
from utils.logging import logger
from app.session import user_sessions
from app.identity import resolve_user_uuid
from app.telegram_link import LINK_PAYLOAD_PREFIX, RedemptionOutcome, extract_link_secret, redeem_link
from utils.access_control import require_authorized
from config import BotMode, DEFAULT_MODE

# Stage 6C, Section I: every REJECTED_* outcome renders to this exact same
# text — deliberately generic, never revealing WHICH conflict occurred (a
# distinguishable response here would let a sender enumerate account state
# they otherwise have no way to observe).
_LINK_REJECTED_TEXT = (
    "⚠️ Не удалось привязать аккаунт. Попробуйте начать привязку заново на сайте."
)
_LINK_INVALID_OR_EXPIRED_TEXT = (
    "⚠️ Эта ссылка недействительна или уже истекла. Запросите новую ссылку на сайте."
)
_LINK_MERGED_TEXT = "✅ Готово! Этот Telegram-аккаунт теперь связан с вашим GitHub-аккаунтом на сайте."
_LINK_ALREADY_LINKED_TEXT = "✅ Этот Telegram-аккаунт уже связан с этим GitHub-аккаунтом."


def _link_outcome_text(outcome: RedemptionOutcome) -> str:
    if outcome == RedemptionOutcome.MERGED:
        return _LINK_MERGED_TEXT
    if outcome == RedemptionOutcome.ALREADY_LINKED:
        return _LINK_ALREADY_LINKED_TEXT
    if outcome == RedemptionOutcome.INVALID_OR_EXPIRED:
        return _LINK_INVALID_OR_EXPIRED_TEXT
    # Every REJECTED_* outcome falls through here (Section I) — see this
    # module's own docstring on _LINK_REJECTED_TEXT above.
    return _LINK_REJECTED_TEXT


@bot.message_handler(commands=['start'])
@require_authorized
async def cmd_start(message: types.Message):
    """Handle /start command."""
    telegram_user_id = message.from_user.id
    # first_name is user-controlled personal data — used in the greeting
    # text sent back to this same user below, but never logged.
    user_name = message.from_user.first_name
    logger.info("Command /start | telegram_user_id=%s", telegram_user_id)

    # Initialize user session — existing first canonical Telegram
    # resolution (Stage 6C, Section I: "existing first canonical Telegram
    # resolution" — preserved unchanged, and always run BEFORE any linking
    # logic below: db.telegram_link.redeem_attempt_sync() requires a
    # telegram_accounts row to already exist for this sender).
    user_uuid = await resolve_user_uuid(telegram_user_id)
    await user_sessions.set_mode(user_uuid, DEFAULT_MODE)

    # Stage 6C: a `/start link_<secret>` payload is redeemed here, narrowly
    # extending the existing handler — every other `/start` shape (no
    # payload, or a payload that isn't a link_ prefix at all) falls through
    # to the unchanged normal welcome flow below. The raw payload is never
    # logged (Section F/I.1) — only the typed outcome is.
    message_text = getattr(message, "text", None) or ""
    payload = message_text.split(maxsplit=1)[1].strip() if " " in message_text else ""
    if payload.startswith(LINK_PAYLOAD_PREFIX):
        raw_secret = extract_link_secret(payload)
        if raw_secret is None:
            await bot.send_message(message.chat.id, _LINK_INVALID_OR_EXPIRED_TEXT)
            return
        outcome = await redeem_link(telegram_user_id=telegram_user_id, raw_secret=raw_secret)
        logger.info(
            "Telegram link redemption | telegram_user_id=%s, outcome=%s", telegram_user_id, outcome.value
        )
        await bot.send_message(message.chat.id, _link_outcome_text(outcome))
        return

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
    telegram_user_id = message.from_user.id
    logger.info("Command /help | telegram_user_id=%s", telegram_user_id)
    
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
    telegram_user_id = message.from_user.id
    user_uuid = await resolve_user_uuid(telegram_user_id)
    user_sessions.clear_history(user_uuid)
    user_sessions.clear_pending_image(user_uuid)
    logger.info("Command /reset | telegram_user_id=%s, history_cleared=True, pending_image_cleared=True", telegram_user_id)

    await bot.send_message(
        message.chat.id,
        "✅ История диалога очищена!\n\n"
        "Начнем с чистого листа. Чем могу помочь?"
    )


@bot.message_handler(commands=['stats'])
@require_authorized
async def cmd_stats(message: types.Message):
    """Handle /stats command - show knowledge base statistics."""
    telegram_user_id = message.from_user.id
    logger.info("Command /stats | telegram_user_id=%s", telegram_user_id)
    try:
        from rag.query import get_knowledge_base_stats
        user_uuid = await resolve_user_uuid(telegram_user_id)
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
        stats = await asyncio.to_thread(get_knowledge_base_stats, str(user_uuid))
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
        logger.error("Command /stats failed | telegram_user_id=%s, error_type=%s", telegram_user_id, type(e).__name__)
        await bot.send_message(
            message.chat.id,
            "⚠️ Ошибка получения статистики базы знаний."
        )
