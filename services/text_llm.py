"""
Text-LLM Provider Facade.

The ONLY place provider selection lives for the three general-purpose
text-tutoring paths (ordinary chat in app/tutor.py / app/text_chat.py; RAG
answer generation and RAG fallback in rag/query.py). Deliberately tiny: a
single explicit dispatch on config.LLM_PROVIDER, no retry, no fallback
between providers. A selected provider's failure propagates unchanged to
the caller's existing sanitized error handling — it must never be caught
here and silently retried through the other provider.

Vision, STT, TTS, image generation, embeddings, and the image-generation-
intent classifier are NOT routed through this facade and remain exclusively
on services/openai_client.py (see services/image_generation.py).

Generation timeout (Stage 7A-1): every dispatched provider call is wrapped
in asyncio.wait_for(..., timeout=config.TEXT_GENERATION_TIMEOUT_SECONDS) —
this is the ONE place the timeout is enforced, so it automatically covers
BOTH providers and BOTH callers (plain chat and RAG, which both funnel
through generate_text_response() here) without either caller needing its
own timeout logic. Deliberately never relies on the provider SDK's own
default timeout alone (services/anthropic_client.py's AsyncAnthropic and
services/openai_client.py's AsyncOpenAI are each built without an explicit
`timeout=`, so their SDK defaults — not bounded to this application's own
policy — would otherwise be the only bound in place). On timeout, the
underlying provider call is genuinely cancelled (asyncio.wait_for cancels
its wrapped task, propagating CancelledError into the awaited coroutine at
its current await point — the actual outbound HTTP request), and a fresh
TextGenerationTimeoutError is raised: a fixed, safe message only, never the
underlying provider exception/text (same privacy-safe-logging posture as
AnthropicResponseError elsewhere in this codebase).

SDK-native timeout normalization (Stage 7A-1 corrective pass): a provider
call can also time out through a path `asyncio.wait_for` itself never
observes as `asyncio.TimeoutError` — the installed SDK's OWN internal
timeout firing (e.g. an underlying httpx2 connect/read timeout the SDK
translates itself), which surfaces as the SDK's own official timeout
exception class rather than asyncio's: `openai.APITimeoutError` and
`anthropic.APITimeoutError` (both confirmed, by reading the installed
`openai._exceptions`/`anthropic._exceptions` modules, to be real classes
in the pinned SDK versions — `class APITimeoutError(APIConnectionError):
def __init__(self, request: httpx2.Request) -> None: ...` in both — never
guessed/assumed). `asyncio.wait_for` does not suppress an exception the
wrapped coroutine itself raises; it only enforces its own timeout, so
either of these SDK exceptions propagates out of `wait_for` exactly like
any other provider failure would, and is caught here alongside
`asyncio.TimeoutError` — all three sources converge on the exact same
`TextGenerationTimeoutError`, with the SAME "raise after the except block"
chain-secrecy fix described below, so a caller (app/text_chat.py) never
needs to distinguish which of the three actually fired. Both SDK-specific
classes are caught explicitly by name (never a broad `except Exception`
that could swallow and silently retry an unrelated provider failure).

Exception-chain secrecy (Stage 7A-1 corrective pass — verified, not merely
assumed): `raise TextGenerationTimeoutError(...) from None` inside an
`except asyncio.TimeoutError:` block sets `__cause__` to None, but Python's
IMPLICIT exception chaining independently sets `__context__` to whatever
exception is currently being handled at the point of the `raise` —
`from None` only sets `__suppress_context__` (which hides that chain from
the STANDARD `traceback` module's default formatting) and does nothing to
the `__context__` attribute itself, which remains a live reference to the
original `asyncio.TimeoutError` (harmless on its own, but the same pattern
matters for TextChatGenerationError in app/text_chat.py, which DOES wrap
exceptions that can carry raw provider text). Confirmed empirically for
this repo's exact Python version — see
tests/test_stage7a1_exception_chain_secrecy.py. Fixed here by raising
`TextGenerationTimeoutError` AFTER the `except` block has already exited
(a boolean flag, never the original exception object, crosses that
boundary) — at that point no exception is being handled, so the
interpreter never populates `__context__` at all, giving BOTH `__cause__`
and `__context__` as None with no special `from` clause needed.
"""

import asyncio
from typing import Dict, List

import anthropic
import openai

import config
from services.anthropic_client import anthropic_client
from services.openai_client import openai_client
from utils.logging import logger
from config import LLMProvider, MAX_TOKENS

__all__ = ["TextGenerationTimeoutError", "generate_text_response"]


class TextGenerationTimeoutError(RuntimeError):
    """Raised when a provider text-generation call does not complete
    within config.TEXT_GENERATION_TIMEOUT_SECONDS (Stage 7A-1). Deliberately
    a fixed, safe message — never the underlying provider exception or any
    prompt/response content. Existing callers (app/tutor.py, app/text_chat.py,
    rag/query.py) already wrap generate_text_response() in a broad
    try/except that logs only type(e).__name__ and returns a sanitized
    user-facing message — this exception is designed to pass through that
    path unchanged, exactly like AnthropicResponseError."""


async def generate_text_response(
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Generate a text response using the configured provider, bounded by
    config.TEXT_GENERATION_TIMEOUT_SECONDS (Stage 7A-1).

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens in response

    Returns:
        Generated text response

    Raises:
        TextGenerationTimeoutError if the call does not complete within the
        configured timeout. Otherwise, whatever the selected provider's own
        generate_text_response raises (e.g. an anthropic/openai SDK
        exception, or AnthropicResponseError) — never caught and retried
        through the other provider here.
    """
    if config.LLM_PROVIDER == LLMProvider.ANTHROPIC:
        provider_call = anthropic_client.generate_text_response(messages, max_tokens=max_tokens)
    elif config.LLM_PROVIDER == LLMProvider.OPENAI:
        provider_call = openai_client.generate_text_response(messages, max_tokens=max_tokens)
    else:
        # Unreachable in practice: config.py already validates LLM_PROVIDER
        # at import time and refuses to load with an invalid value. Kept as
        # an explicit fail-closed guard rather than an assert, in case a
        # future caller ever constructs this module's dependency differently.
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER!r}")

    logger.debug("text_llm dispatch | provider=%s", config.LLM_PROVIDER)
    timed_out = False
    try:
        return await asyncio.wait_for(provider_call, timeout=config.TEXT_GENERATION_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(
            "text_llm generation timed out (asyncio wait_for) | provider=%s, timeout_seconds=%s",
            config.LLM_PROVIDER, config.TEXT_GENERATION_TIMEOUT_SECONDS,
        )
        timed_out = True
    except (openai.APITimeoutError, anthropic.APITimeoutError) as e:
        # The installed SDK's OWN native timeout — see this module's own
        # docstring ("SDK-native timeout normalization"). Only the safe
        # class name is logged, never the SDK exception's own text/request.
        logger.error(
            "text_llm generation timed out (SDK-native) | provider=%s, timeout_seconds=%s, sdk_error_type=%s",
            config.LLM_PROVIDER, config.TEXT_GENERATION_TIMEOUT_SECONDS, type(e).__name__,
        )
        timed_out = True

    # Raised OUTSIDE the except block on purpose — see this module's own
    # docstring ("Exception-chain secrecy") for why raising from inside
    # `except asyncio.TimeoutError:` (even with `from None`) leaves
    # __context__ pointing at the original exception. Only a boolean flag
    # crosses the except-block boundary, never the exception object itself.
    if timed_out:
        raise TextGenerationTimeoutError("Text generation timed out")
