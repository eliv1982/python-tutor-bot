"""
Anthropic Client for the Personal Assistant Bot.
Provides the general-purpose text-generation path when LLM_PROVIDER=anthropic
(see config.py and services/text_llm.py). Text generation ONLY — vision,
STT, TTS, image generation, embeddings, and the image-generation-intent
classifier remain exclusively on services/openai_client.py in this stage.
"""

from typing import Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    OFFICIAL_ANTHROPIC_BASE_URL,
    MAX_TOKENS,
)
from utils.logging import logger


class AnthropicResponseError(RuntimeError):
    """
    Raised when an Anthropic response contains no usable text content
    (empty content list, thinking-only content, or a refusal with no
    accompanying text).

    Deliberately carries only a fixed, safe message — never the raw
    response object or provider text, which could echo prompt/context
    content. Existing router/RAG callers already wrap generate_text_response
    in a broad try/except that logs only type(e).__name__ and returns a
    sanitized user-facing message (Stage 1D privacy guarantee) — this
    exception is designed to pass through that path unchanged.
    """


def _split_system_message(
    messages: List[Dict[str, str]]
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """
    Split a leading {"role": "system", ...} entry out of `messages` for
    Anthropic's request shape.

    Anthropic's Messages API takes system content via a top-level `system=`
    parameter, never as a "system"-role entry inside the messages array
    (confirmed against the installed SDK: MessageParam's role type permits
    "system" structurally, but that spelling is a distinct, model-gated
    "mid-conversation operator message" feature this application does not
    use — Sonnet 5 does not support it at all). Only a LEADING system
    message is recognized, matching this application's own message-
    construction convention (services/router.py and rag/query.py always
    place it first when present, never elsewhere).

    Fails clearly (ValueError) rather than silently reinterpreting anything:
    - a leading system message whose "content" key is missing, None, or any
      non-string value (previously this was silently dropped as if no
      system message had been sent at all — a genuinely absent system
      message remains valid, but a malformed one must never be
      reinterpreted as one);
    - any role other than "system" (leading only), "user", or "assistant"
      in the remaining messages;
    - any non-string message content (this adapter does not support
      multipart content — none of its three call sites ever send any).
    """
    if messages and messages[0].get("role") == "system":
        if "content" not in messages[0] or not isinstance(messages[0]["content"], str):
            raise ValueError(
                "Anthropic adapter requires a leading system message to have "
                "plain string content"
            )
        system_content = messages[0]["content"]
        remaining = messages[1:]
    else:
        system_content = None
        remaining = messages

    anthropic_messages: List[Dict[str, str]] = []
    for message in remaining:
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"Unexpected message role for Anthropic request: {role!r}")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Anthropic adapter only supports plain string message content")
        anthropic_messages.append({"role": role, "content": content})

    return system_content, anthropic_messages


def _extract_text(response) -> str:
    """
    Extract plain text from an Anthropic Message response.

    response.content is a LIST of content blocks (TextBlock, ThinkingBlock,
    ...) — unlike OpenAI's response.choices[0].message.content, it is never
    a single string. Text blocks are concatenated in order; non-text blocks
    (thinking, disabled in this stage but defensively handled anyway) are
    ignored rather than erroring.

    stop_reason == "refusal" is not special-cased: if the refused response
    still carries a text block, it's returned like any other text response;
    if not, this falls through to the same "no usable text" error as any
    other empty/non-text response — no new Telegram-facing refusal UX in
    this stage.
    """
    text_parts = [
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ]
    text = "".join(text_parts)
    if not text:
        raise AnthropicResponseError("Anthropic response contained no usable text content")
    return text


class AnthropicClient:
    """Async client for Anthropic text generation."""

    def __init__(self):
        """Initialize the Anthropic client."""
        # base_url is pinned explicitly (not left to SDK defaults): the
        # anthropic SDK falls back to reading an ANTHROPIC_BASE_URL
        # environment variable when base_url isn't passed, which would
        # otherwise let a stray value in a developer's .env silently
        # re-route requests — same rationale as OFFICIAL_OPENAI_BASE_URL in
        # config.py / services/openai_client.py.
        #
        # http_client is likewise pinned explicitly. trust_env=False alone is
        # NOT sufficient here, unlike a plain httpx2.AsyncClient: confirmed
        # against the installed SDK (anthropic/_base_client.py,
        # _DefaultAsyncHttpxClient.__init__, both on 1.0.0 and 1.2.0), the
        # Anthropic SDK runs its OWN proxy discovery
        # (anthropic._utils._httpx.get_environment_proxies(), which wraps
        # urllib.request.getproxies() — so it sees both conventional
        # HTTP_PROXY/HTTPS_PROXY/ALL_PROXY env vars AND OS-level discovery,
        # e.g. Windows Registry / macOS system config) UNCONDITIONALLY,
        # before trust_env is ever consulted, and builds "http://"/"https://"/
        # "all://" transport mounts from whatever it finds. trust_env=False
        # only affects the transport *objects* those mounts would use; it
        # does not stop the mounts from being created and installed.
        #
        # The SDK does prioritize caller-supplied `mounts=` over its own
        # discovered ones (`proxy_mounts.update(kwargs.get("mounts", {}))`
        # in _base_client.py, confirmed on both 1.0.0 and 1.2.0), so passing
        # explicit None mounts for exactly the three keys the SDK's own
        # discovery ever populates with a live proxy overrides them
        # unconditionally, regardless of what get_environment_proxies()
        # returns. Verified empirically against both anthropic==1.0.0 and
        # anthropic==1.2.0 in disposable environments: with a hostile
        # HTTP_PROXY/HTTPS_PROXY/ALL_PROXY set, or with
        # anthropic._utils._httpx.getproxies() patched to emulate OS-level
        # discovery, trust_env=False alone still produces live proxy
        # transport mounts; adding the explicit mounts below produces mounts
        # that resolve to None (direct connection) in both cases. See
        # tests/test_stage2a_text_llm_provider.py's proxy regression tests.
        #
        # DefaultAsyncHttpxClient is the SDK's own public factory for its
        # recommended httpx2 client defaults (timeout/limits/redirects);
        # trust_env and mounts are both forwarded to httpx2.AsyncClient
        # unchanged — no direct httpx2 import is needed here.
        self.client = AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url=OFFICIAL_ANTHROPIC_BASE_URL,
            http_client=DefaultAsyncHttpxClient(
                trust_env=False,
                mounts={"http://": None, "https://": None, "all://": None},
            ),
        )
        logger.info("Anthropic client initialized with official Anthropic API")

    async def generate_text_response(
        self,
        messages: List[Dict[str, str]],
        model: str = ANTHROPIC_MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """
        Generate a text response using Claude.

        Stage 2A default request profile: thinking explicitly disabled, no
        `temperature`/`top_p`/`top_k` sent at all — Claude Sonnet 5 rejects
        non-default sampling parameters, and the installed SDK's
        messages.create() doesn't even expose them as named parameters, so
        this adapter deliberately has no sampling-parameter argument to
        translate OpenAI's TEMPERATURE into. Good interactive tutoring
        quality without hidden-thinking cost on every ordinary message is
        the explicit goal for this stage, not a tunable reasoning mode.

        Args:
            messages: List of message dictionaries with 'role' and
                'content' — the same shape services/openai_client.py's
                generate_text_response() accepts. An optional leading
                {"role": "system", ...} entry is extracted to Anthropic's
                top-level `system` parameter; every other entry must have
                role "user" or "assistant".
            model: Model to use
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        system_content, anthropic_messages = _split_system_message(messages)
        try:
            logger.debug(
                "Anthropic messages.create | model=%s, messages=%s, has_system=%s",
                model, len(anthropic_messages), system_content is not None,
            )
            create_kwargs = {}
            if system_content is not None:
                create_kwargs["system"] = system_content

            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "disabled"},
                messages=anthropic_messages,
                **create_kwargs,
            )

            result = _extract_text(response)
            usage = getattr(response, "usage", None)
            usage_str = f", usage={usage}" if usage else ""
            logger.info(
                "Anthropic text response | model=%s, len=%s, stop_reason=%s%s",
                model, len(result), getattr(response, "stop_reason", None), usage_str,
            )
            return result
        except AnthropicResponseError:
            # Already a safe, fixed message — re-raise as-is (no provider
            # exception text involved).
            raise
        except Exception as e:
            # Anthropic SDK exceptions (HTTP/auth/rate-limit) are external
            # and must never be logged raw — only their class name is safe.
            logger.error("Anthropic text response failed | model=%s, error_type=%s", model, type(e).__name__)
            raise


# Global client instance
anthropic_client = AnthropicClient()
