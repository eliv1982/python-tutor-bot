"""
Main Bot Module.
Initializes and configures the Telegram bot using pyTelegramBotAPI.
"""

from telebot.async_telebot import AsyncTeleBot

from config import TELEGRAM_BOT_TOKEN
from utils.logging import logger


# Create bot instance (без Markdown — ответы обычным текстом, без сбоев от _ и *)
bot = AsyncTeleBot(TELEGRAM_BOT_TOKEN)

logger.info("Bot instance created")
