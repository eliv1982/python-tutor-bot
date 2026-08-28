"""
Configuration module for the Personal Assistant Telegram Bot.
Loads environment variables and provides configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env file")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

# The one and only OpenAI endpoint this application ever talks to. Passed
# explicitly to every OpenAI-backed client (chat/vision/STT/TTS, image
# generation, embeddings) rather than left unset, because the underlying
# SDKs each fall back to reading their own base-URL environment variable
# when no explicit value is given. This constant must never be made
# configurable via the environment again — that would reopen a way for
# requests to be silently redirected to a non-official endpoint.
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Text-LLM Provider Selection (Stage 2A)
class LLMProvider:
    ANTHROPIC = "anthropic"
    OPENAI = "openai"

_VALID_LLM_PROVIDERS = (LLMProvider.ANTHROPIC, LLMProvider.OPENAI)

# Selects which provider answers the three general-purpose text-tutoring
# paths (ordinary chat, RAG answer generation, RAG fallback — see
# services/text_llm.py). Vision/STT/TTS/image-generation/embeddings and the
# image-generation-intent classifier are NOT affected by this setting and
# remain OpenAI-specific regardless of its value. An unrecognized value
# fails closed at import time (below) rather than silently defaulting to
# either provider — same fail-fast posture as TELEGRAM_BOT_TOKEN/
# OPENAI_API_KEY above.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", LLMProvider.ANTHROPIC)
if LLM_PROVIDER not in _VALID_LLM_PROVIDERS:
    raise ValueError(
        f"LLM_PROVIDER must be one of {_VALID_LLM_PROVIDERS!r}, got {LLM_PROVIDER!r}"
    )

# Anthropic Configuration
# Required only when LLM_PROVIDER selects Anthropic: OpenAI's key (above)
# remains required unconditionally, since OpenAI is still needed for
# embeddings/vision/STT/TTS/image generation/image-intent classification no
# matter which provider answers the general text-tutoring paths.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if LLM_PROVIDER == LLMProvider.ANTHROPIC and not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY is not set in .env file (required when LLM_PROVIDER=anthropic)"
    )

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# The one and only Anthropic endpoint this application ever talks to —
# same rationale as OFFICIAL_OPENAI_BASE_URL above: the Anthropic SDK falls
# back to reading an ANTHROPIC_BASE_URL environment variable whenever
# base_url isn't passed explicitly (confirmed against the installed SDK: an
# AsyncAnthropic() constructed without an explicit base_url picks up a
# stray ANTHROPIC_BASE_URL immediately). This constant must never be made
# configurable via the environment — that would reopen the same
# silent-redirection risk this pattern already closes for OpenAI.
OFFICIAL_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# Bot Modes
class BotMode:
    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    RAG = "rag"

DEFAULT_MODE = os.getenv("BOT_MODE", BotMode.TEXT)

# Voice Configuration
class VoiceType:
    ALLOY = "alloy"      # Neutral
    ECHO = "echo"        # Male
    NOVA = "nova"        # Female
    FABLE = "fable"      # Male (British)
    ONYX = "onyx"        # Male (Deep)
    SHIMMER = "shimmer"  # Female (Warm)

DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", VoiceType.ALLOY)

# OpenAI Models
GPT_MODEL = "gpt-4o"
GPT_MINI_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "tts-1"
VISION_MODEL = "gpt-4o"
DALLE_MODEL = "dall-e-3"

# DALL-E Configuration
DALLE_DEFAULT_SIZE = "1024x1024"  # Options: 1024x1024, 1024x1792, 1792x1024
DALLE_DEFAULT_QUALITY = "standard"  # Options: standard, hd
DALLE_DEFAULT_STYLE = "vivid"  # Options: vivid, natural

# Maximum size (in bytes) of a Telegram-downloaded image accepted for
# vision analysis, enforced before base64 encoding / session storage.
MAX_TELEGRAM_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB

# Maximum size (in bytes) of a Telegram-downloaded document accepted for
# RAG ingestion, enforced against the actual downloaded bytes before the
# file is written to disk, parsed, or indexed.
MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Database Configuration
DB_PATH = BASE_DIR / os.getenv("DB_PATH", "data/embeddings.db")

# Data paths
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EMBEDDINGS_DB = DATA_DIR / "embeddings.db"

# Physical storage root for application-managed Telegram document uploads
# (opaque UUID-named files, see handlers/document_upload.py). Nested under
# DOCUMENTS_DIR but deliberately excluded from rag/loader.py's
# load_directory() startup/reference scan: an upload is already indexed
# into the persistent Chroma store at upload time with its original
# filename as source metadata, so blindly rescanning disk on startup would
# re-index it a second time under its opaque UUID filename. Interim design
# until the later PostgreSQL/Qdrant document registry stage — if the
# Chroma store is manually destroyed, uploads are not reconstructed from
# disk; that recovery path belongs to the later registry architecture.
#
# Deliberately NOT created here (unlike DATA_DIR/DOCUMENTS_DIR below):
# config.py is imported by every test module, and an eager mkdir would
# create this directory on the real filesystem the moment any test simply
# imports config, before a test monkeypatches this path to a tmp_path.
# handlers/document_upload.py creates it lazily, only when an upload
# actually needs to write to it.
MANAGED_UPLOADS_DIR = DOCUMENTS_DIR / "uploads"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "bot.log"

# RAG Configuration
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 3

# OpenAI Settings
TEMPERATURE = 0.7
MAX_TOKENS = 1500

# User session settings
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in history

