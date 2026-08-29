"""
Text-LLM Provider Facade.

The ONLY place provider selection lives for the three general-purpose
text-tutoring paths (ordinary chat in app/tutor.py; RAG answer
generation and RAG fallback in rag/query.py). Deliberately tiny: a single
explicit dispatch on config.LLM_PROVIDER, no retry, no fallback between
providers. A selected provider's failure propagates unchanged to the
caller's existing sanitized error handling — it must never be caught here
and silently retried through the other provider.

Vision, STT, TTS, image generation, embeddings, and the image-generation-
intent classifier are NOT routed through this facade and remain exclusively
on services/openai_client.py (see services/image_generation.py).
"""

from typing import Dict, List

import config
from services.anthropic_client import anthropic_client
from services.openai_client import openai_client
from utils.logging import logger
from config import LLMProvider, MAX_TOKENS


async def generate_text_response(
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Generate a text response using the configured provider.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens in response

    Returns:
        Generated text response

    Raises:
        Whatever the selected provider's own generate_text_response raises
        (e.g. an anthropic/openai SDK exception, or AnthropicResponseError)
        — never caught and retried through the other provider here.
    """
    if config.LLM_PROVIDER == LLMProvider.ANTHROPIC:
        logger.debug("text_llm dispatch | provider=%s", config.LLM_PROVIDER)
        return await anthropic_client.generate_text_response(messages, max_tokens=max_tokens)
    elif config.LLM_PROVIDER == LLMProvider.OPENAI:
        logger.debug("text_llm dispatch | provider=%s", config.LLM_PROVIDER)
        return await openai_client.generate_text_response(messages, max_tokens=max_tokens)
    else:
        # Unreachable in practice: config.py already validates LLM_PROVIDER
        # at import time and refuses to load with an invalid value. Kept as
        # an explicit fail-closed guard rather than an assert, in case a
        # future caller ever constructs this module's dependency differently.
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER!r}")
