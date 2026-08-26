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
        logger.error("Setup: failed to import handlers: %s", e, exc_info=True)
        raise
    
    try:
        from rag.index import vector_index
        from rag.loader import SUPPORTED_EXTENSIONS
        from config import DOCUMENTS_DIR

        docs = list(DOCUMENTS_DIR.glob('*'))
        docs = [d for d in docs if d.is_file() and d.suffix.lower() in SUPPORTED_EXTENSIONS]
        logger.debug("Setup: RAG documents dir scan: path=%s, files=%s", DOCUMENTS_DIR, [d.name for d in docs])
        
        if docs:
            logger.info("Setup: RAG indexing started, documents=%s", [d.name for d in docs])
            count = vector_index.index_documents_directory(force_reindex=False)
            logger.info("Setup: RAG indexing done, chunks_indexed=%s", count)
        else:
            logger.info("Setup: RAG skipped, no documents in data/documents/")
    except Exception as e:
        logger.warning("Setup: RAG init failed (non-fatal): %s", e, exc_info=True)
    
    try:
        bot_info = await bot.get_me()
        logger.info("Setup: Telegram API OK, bot=@%s", bot_info.username)
    except Exception as e:
        logger.error("Setup: Telegram get_me failed: %s", e, exc_info=True)


async def shutdown_bot():
    """Actions to perform on bot shutdown."""
    logger.info("Shutdown: closing bot session")
    try:
        await bot.close_session()
        logger.debug("Shutdown: session closed")
    except Exception as e:
        logger.debug("Shutdown: close_session exception (ignored): %s", e)
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
        logger.error(f"Fatal error: {e}", exc_info=True)
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
        logger.error(f"Startup error: {e}", exc_info=True)
