"""
Logging configuration for the Personal Assistant Bot.
Provides colored console output and file logging (bot.log). Use LOG_LEVEL=DEBUG for detailed logs.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config import LOG_LEVEL, LOG_FILE


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


def setup_logging(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration with both file and console handlers.
    
    Args:
        name: Logger name (defaults to root logger)
        level: Logging level (defaults to config.LOG_LEVEL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set logging level
    log_level = getattr(logging, level or LOG_LEVEL)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logging('bot')

