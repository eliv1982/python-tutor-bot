"""
Main Entry Point.
Starts the Telegram bot using pyTelegramBotAPI.
"""

import asyncio
import sys

from bot import bot
from utils.logging import logger


async def setup_bot():
    """Setup bot with handlers and initialize RAG if needed."""
    logger.info("Setup: starting bot initialization")
    
    try:
        from handlers import start, text, voice, image, document_upload
        logger.info("Setup: handlers loaded (start, text, voice, image, document_upload)")
    except Exception as e:
        # An import failure's traceback/message can embed absolute source
        # paths, the OS username, site-packages/dependency locations, and
        # other environment-dependent detail — never guaranteed harmless,
        # so only the exception's class name is logged here.
        logger.error("Setup: handler import failed | error_type=%s", type(e).__name__)
        raise
    
    try:
        from rag.index import vector_index
        from rag.loader import SUPPORTED_EXTENSIONS
        from config import DOCUMENTS_DIR

        docs = list(DOCUMENTS_DIR.glob('*'))
        docs = [d for d in docs if d.is_file() and d.suffix.lower() in SUPPORTED_EXTENSIONS]
        # Count only: neither the filenames nor the absolute directory path
        # (which reveals the deployment's filesystem layout/username) are
        # needed for this diagnostic.
        logger.debug("Setup: RAG documents dir scan | file_count=%s", len(docs))

        if docs:
            logger.info("Setup: RAG indexing started, document_count=%s", len(docs))
            count = vector_index.index_documents_directory(force_reindex=False)
            logger.info("Setup: RAG indexing done, chunks_indexed=%s", count)
        else:
            logger.info("Setup: RAG skipped, no documents in data/documents/")
    except Exception as e:
        # index_documents_directory() calls OpenAIEmbeddings (network) —
        # never log raw exception text.
        logger.warning("Setup: RAG init failed (non-fatal) | error_type=%s", type(e).__name__)
    
    try:
        bot_info = await bot.get_me()
        logger.info("Setup: Telegram API OK, bot=@%s", bot_info.username)
    except Exception as e:
        # A direct Telegram API call: its HTTP exception can embed the
        # token-bearing request URL — never log raw exception text.
        logger.error("Setup: Telegram get_me failed | error_type=%s", type(e).__name__)


async def shutdown_bot():
    """Actions to perform on bot shutdown."""
    logger.info("Shutdown: closing bot session")
    try:
        await bot.close_session()
        logger.debug("Shutdown: session closed")
    except Exception as e:
        logger.debug("Shutdown: close_session exception (ignored) | error_type=%s", type(e).__name__)
    logger.info("Shutdown: complete")


async def main():
    """Main function to run the bot."""
    try:
        # Setup bot
        await setup_bot()
        
        logger.info("Main: entering infinity_polling (timeout=10, skip_pending=True)")
        await bot.infinity_polling(
            timeout=10,
            skip_pending=True
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    except Exception as e:
        # This top-level handler can catch anything bubbling out of
        # infinity_polling(), a live Telegram API call whose exceptions can
        # embed the token-bearing request URL — never log raw exception
        # text or a traceback.
        logger.error("Fatal error | error_type=%s", type(e).__name__)
        sys.exit(1)
    finally:
        await shutdown_bot()


if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("Personal Python Tutor Bot - Starting")
        logger.info("="*60)
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    except Exception as e:
        logger.error("Startup error | error_type=%s", type(e).__name__)
