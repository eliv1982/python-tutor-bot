"""
Main Entry Point.
Starts the Telegram bot using pyTelegramBotAPI.
"""

import asyncio
import sys

from bot import bot
from utils.logging import logger, configure_logging


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
        from rag.index import get_vector_index

        # index_documents_directory() enumerates EXACTLY
        # config.BUILTIN_REFERENCE_FILES by default (Stage 2B-C Blocker 5)
        # and reconciles each one with zero embedding/Qdrant calls when
        # already current (Blocker 2) — always safe/cheap to call
        # unconditionally rather than pre-checking whether any files exist.
        # A missing manifest file fails loudly (caught below, non-fatal to
        # startup) rather than silently indexing fewer built-in documents.
        # get_vector_index() constructs the shared Qdrant-backed singleton
        # on this, its first real call (Stage 2B-D Blocker 4) — merely
        # importing rag.index earlier never did this.
        logger.info("Setup: RAG indexing started")
        count = get_vector_index().index_documents_directory(force_reindex=False)
        logger.info("Setup: RAG indexing done, chunks_indexed=%s", count)
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
    try:
        # Stage 2B-E Section N: deterministic release of the shared
        # VectorIndex singleton's local-persistent Qdrant client/storage-
        # path lock on every real shutdown, not merely relied on process
        # exit to release it. A lazy import (mirrors setup_bot()'s own
        # `from rag.index import get_vector_index` above) — close_vector_
        # index() is already a safe no-op if setup_bot()'s RAG indexing
        # never actually constructed the singleton (e.g. it failed and was
        # caught there), so this never constructs a VectorIndex merely to
        # close it.
        from rag.index import close_vector_index
        close_vector_index()
        logger.debug("Shutdown: vector index closed")
    except Exception as e:
        logger.debug("Shutdown: close_vector_index exception (ignored) | error_type=%s", type(e).__name__)
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
    # Real application startup/composition root (Stage 2B-D Section G) —
    # this is the one and only place configure_logging() is called. It
    # installs the real console+file (bot.log) handlers; nothing before
    # this point (including every module import above) creates bot.log.
    configure_logging()
    try:
        logger.info("="*60)
        logger.info("Personal Python Tutor Bot - Starting")
        logger.info("="*60)
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    except Exception as e:
        logger.error("Startup error | error_type=%s", type(e).__name__)
