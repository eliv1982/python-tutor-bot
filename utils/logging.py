"""
Logging configuration for the Personal Assistant Bot.
Provides colored console output and file logging (bot.log). Use LOG_LEVEL=DEBUG for detailed logs.

Stage 2B-D Section G: importing this module must never open bot.log (or
require any credential-validating config). `logger` below is created at
import time but carries no handlers until `configure_logging()` is
explicitly called — that call, not the import, is what reads
config.LOG_LEVEL/config.LOG_FILE and installs the real console+file
handlers. Call it exactly once, from the real application's startup/
composition root (main.py), never from a reusable library module. Logging
calls made before configure_logging() simply propagate to the root
logger's default behavior (Python's "handler of last resort" prints
WARNING+ to stderr) instead of raising or silently doing anything harmful.
"""

import logging
import sys
import threading
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


_configure_lock = threading.Lock()


def configure_logging(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Install (or reinstall) console + file handlers on the named logger
    (default: the shared 'bot' logger this module exposes as `logger`).

    Reads config.LOG_LEVEL/config.LOG_FILE lazily, at call time — this is
    what requires credential-validating config and creates bot.log, not
    merely importing this module. Call exactly once from the real
    application's startup/composition root; never from a reusable library
    module (Stage 2B-D Section G).

    Idempotent: a second call is a safe no-op against handler duplication
    — every existing handler on the target logger is explicitly closed
    (releasing its OS-level file handle) and removed before the new
    console+file handlers are installed.

    Args:
        name: Logger name (defaults to the shared 'bot' logger)
        level: Logging level (defaults to config.LOG_LEVEL)

    Returns:
        The configured logger instance.
    """
    from config import LOG_LEVEL, LOG_FILE  # lazy: only real startup needs this

    target_logger = logging.getLogger(name or 'bot')

    with _configure_lock:
        log_level = getattr(logging, level or LOG_LEVEL)
        target_logger.setLevel(log_level)

        for handler in list(target_logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            target_logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        target_logger.addHandler(console_handler)

        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        target_logger.addHandler(file_handler)

    return target_logger


# Shared application logger. Carries NO handlers until configure_logging()
# is explicitly called — merely importing this module never opens bot.log
# (Stage 2B-D Section G).
logger = logging.getLogger('bot')
