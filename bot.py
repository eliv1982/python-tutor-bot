"""
Main Bot Module.
Initializes and configures the Telegram bot using pyTelegramBotAPI.
"""

from telebot.async_telebot import AsyncTeleBot, ExceptionHandler

from telegram_config import TELEGRAM_BOT_TOKEN
from utils.logging import logger


class _SafeDispatcherExceptionHandler(ExceptionHandler):
    """
    Central backstop for exceptions pyTelegramBotAPI's own async dispatcher
    would otherwise catch and log itself (telebot.async_telebot.py's
    `_run_middlewares_and_handlers`): when no `exception_handler` is
    installed, an exception that escapes a handler is logged there via
    `logger.error(str(e))` + `logger.debug(traceback.format_exc())` on
    telebot's own 'TeleBot' logger — which has its own stderr handler
    installed at import time (telebot/__init__.py), independent of this
    app's own sanitized try/except blocks. A Telegram `ApiException` can
    embed the token-bearing `/bot<TOKEN>/...` request URL in that raw text,
    so this handler exists purely to intercept first and log only the
    exception's class name — every handler already has its own local
    try/except for expected failures, so reaching this point at all means
    an exception genuinely was not anticipated.

    Deliberately does not send any Telegram response itself (the handler
    that raised is already gone; guessing at a chat id here would risk
    misrouting) and never touches the exception's message/args or any
    update/message content — only `type(exception).__name__`.
    """

    async def handle(self, exception: Exception) -> bool:
        logger.error("Unhandled dispatcher exception | error_type=%s", type(exception).__name__)
        return True  # tells pyTelegramBotAPI this was handled: suppresses its own raw logging


# Create bot instance (без Markdown — ответы обычным текстом, без сбоев от _ и *)
bot = AsyncTeleBot(TELEGRAM_BOT_TOKEN, exception_handler=_SafeDispatcherExceptionHandler())

logger.info("Bot instance created")
