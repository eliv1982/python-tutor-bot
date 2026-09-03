"""
Tutoring Orchestration (Stage 5B application boundary).

Moved from services/router.py: this module already took only
primitive/domain values (user_id, text, Path) and never depended on
telebot message/update objects, a Telegram bot instance, Telegram send/
edit APIs, Telegram keyboards, or Telegram-specific access control — so
this move is a rename/relocation into the explicit application layer,
not a behavior change. A Telegram (or future FastAPI) adapter translates
its own transport-specific input into these primitive values and
translates the returned dict back into its own response format.

Stage 7A-1: route_text_request()'s plain-chat/RAG generation is now
delegated to app.text_chat — a stateless, adapter-independent core shared
with any future adapter. THIS module keeps everything genuinely
Telegram-specific: image-intent detection/routing, and Telegram's own
ephemeral conversation state (app.session.user_sessions) — read here as
an explicit snapshot and passed into the core, never read or written by
the core itself. See app/text_chat.py's own docstring for the full split
rationale.

Stage 7A-1 corrective pass — Telegram linearizability: route_text_request()
no longer calls app.text_chat.run_text_chat() as a single opaque call.
Instead it:
  1. validates the scalar request (app.text_chat.validate_scalar_request())
     BEFORE ever requesting an admission permit — a malformed request must
     never contend for a generation slot;
  2. acquires ONE admission permit for the whole transaction
     (app.text_chat.generation_admission_controller.acquire());
  3. under that SAME permit: snapshots Telegram's own history, validates/
     normalizes it, runs generation (app.text_chat.
     execute_admitted_text_chat()), and — only on success — atomically
     commits the completed user/assistant pair
     (app.session.UserSession.add_exchange()).
This closes a race the earlier delegation-only design left open: a
second concurrent same-user request could previously have been admitted
in the gap between the first request's generation finishing (which
released its own, narrower permit) and its history commit running,
observing a stale snapshot that was missing the first request's
still-uncommitted turn. Holding ONE permit across the whole snapshot ->
generate -> commit span means a second concurrent same-user request is
now rejected with GenerationBusyError immediately — before it ever reads
history or calls the provider — for as long as the first request's
transaction (including its commit) is still in flight. `text_chat.<name>`
is always accessed through the module object (never a separately bound
alias) so a test that swaps `text_chat.generation_admission_controller`
or `text_chat.execute_admitted_text_chat` is seen identically by both
this module and app/text_chat.py's own internals — see
tests/test_stage7a1_telegram_regression.py.

Stage 7A-1 SECOND corrective pass — full validation/admission ordering:
an independent re-audit found that, despite the restructuring above,
route_text_request() still performed Telegram history reads, raw-mode
logging, and the image-intent classifier's own provider call BEFORE
scalar validation and BEFORE admission — a malformed request (bad UUID/
mode/whitespace-only message) or a same-user busy rejection could
therefore still trigger a history read and a classifier call before ever
failing. The ordering is now, unconditionally:
  1. resolve `mode` from Telegram's own session state if not given
     explicitly (the only thing that must happen before validation, since
     validation needs an actual value to check);
  2. text_chat.validate_scalar_request() — raises BEFORE any history read,
     any logging of `mode`, the image-intent classifier, or admission;
  3. ONE admission-permit acquisition
     (text_chat.generation_admission_controller.acquire()) — a same-user
     busy rejection raises GenerationBusyError HERE, before the line below
     ever runs;
  4. ONLY THEN: a single Telegram history snapshot (user_sessions.
     get_history()) — reused for BOTH the image-intent classifier below
     AND (if not an image request) generation/normalization, so there is
     no separate, earlier history read for intent classification the way
     the first corrective pass still had;
  5. the image-intent classifier (detect_image_generation_intent()) — now
     itself gated behind the SAME permit, per requirement: it is
     "provider-backed routing decision" work, not merely a Telegram-local
     computation, so it must never run for a request that scalar
     validation or admission would have rejected.
If the classifier decides this is an image-generation request, the
permit is released (by exiting the `async with` block normally) BEFORE
route_image_generation_request() is ever called — that path has its own,
separate history bookkeeping and is intentionally outside admission
control's scope (unchanged from before), so it must never hold a
text-generation permit while it runs its own DALL-E call. Only the
`mode`/message length are ever logged before validation passes; the raw
`mode` value itself is only ever logged AFTER validate_scalar_request()
has confirmed it is a canonical value — see
tests/test_stage7a1_telegram_regression.py's precondition-ordering
proofs.

route_rag_request() (a pre-Stage-7A-1 direct rag.query.query_knowledge_base()
call with its own retrieval/history bookkeeping) has been removed: it had
no production caller anywhere in this codebase and bypassed admission
control, the timeout/error taxonomy, and this module's own atomic history
commit entirely — exactly the kind of legacy bypass Stage 7A-1 exists to
close. RAG requests reach app.text_chat's validated/admitted core the
same way plain-chat ones do: via route_text_request(mode=BotMode.RAG).
"""

import uuid
from typing import Dict, Any, Optional
from pathlib import Path

import app.text_chat as text_chat
from services.stt import transcribe_voice_message
from services.tts import generate_voice_response
from services.vision import analyze_image
from services.image_generation import detect_image_generation_intent, generate_image
from utils.logging import logger
from utils.helpers import strip_markdown
from app.session import user_sessions


async def route_text_request(
    user_id: uuid.UUID,
    text: str,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route text request to appropriate handler.

    Stage 7A-1 corrective pass: image-intent routing stays here (a
    separate, Telegram-only path with its own history bookkeeping,
    entirely outside admission control's scope); actual plain-chat/RAG
    generation and Telegram's own history snapshot/commit now happen
    together under ONE admission permit — see this module's own docstring
    for the full linearizability rationale. Telegram's history is
    committed ONLY after a successful generation, as a single atomic
    operation — never before, and never as two independently observable
    appends — so a timeout, cancellation, or provider exception never
    leaves a partially-recorded turn (a user message with no matching
    assistant reply) that would otherwise silently pollute the context of
    the next successful turn.

    Args:
        user_id: Canonical internal user UUID (Stage 5C)
        text: User's text message
        mode: Bot mode (text, rag, etc.)

    Returns:
        Response dictionary with 'text' and optional 'voice_path' and 'image_path'
    """
    try:
        if mode is None:
            mode = await user_sessions.get_mode(user_id)

        # Scalar validation BEFORE admission, BEFORE any history read,
        # BEFORE any raw-mode logging, and BEFORE the image-intent
        # classifier (Stage 7A-1 second corrective pass): a malformed
        # request must never trigger any of those side effects, let alone
        # occupy a generation permit. `mode` is not logged here — only
        # AFTER this call confirms it is a canonical value (see the
        # logger.debug call below, inside the admitted block).
        text_chat.validate_scalar_request(user_id=user_id, message=text, mode=mode)

        pending_image_prompt: Optional[str] = None
        pending_image_confidence = None
        result = None

        # ONE admission permit covers the ENTIRE remaining transaction —
        # the history snapshot, the image-intent classifier (itself a
        # provider-backed routing decision), generation, and the atomic
        # history commit (Stage 7A-1 second corrective pass). A same-user
        # busy rejection therefore happens HERE, before any of that runs:
        # acquire() raises GenerationBusyError immediately, before this
        # block's body ever executes.
        async with text_chat.generation_admission_controller.acquire(user_id) as permit:
            # A single history snapshot, taken only now, reused for BOTH
            # the classifier below and (if this is not an image request)
            # generation/normalization — no separate, earlier read for
            # intent classification.
            history = user_sessions.get_history(user_id)
            logger.debug("route_text_request | user_id=%s, mode=%s, history_len=%s, text_len=%s", user_id, mode, len(history), len(text))

            image_intent = await detect_image_generation_intent(text, history)
            if image_intent.get('needs_generation') and image_intent.get('confidence', 0) > 0.5:
                pending_image_prompt = image_intent.get('prompt', text)
                pending_image_confidence = image_intent.get('confidence')
            else:
                history_snapshot = text_chat.validate_and_normalize_history(history)
                result = await text_chat.execute_admitted_text_chat(
                    user_id=user_id,
                    message=text,
                    history_snapshot=history_snapshot,
                    mode=mode,
                    permit=permit,
                )

                # One atomic, adapter-owned commit — never two
                # independently observable appends — still under the SAME
                # permit that covered generation (Stage 7A-1 corrective
                # pass).
                user_sessions.add_exchange(user_id, text, result.text)

        if pending_image_prompt is not None:
            # The permit above has already been released (the `async
            # with` block exited normally) — image generation has its own,
            # separate history bookkeeping and must never hold a
            # text-generation permit while its own DALL-E call runs.
            logger.info("route_text_request: image generation | user_id=%s, confidence=%s", user_id, pending_image_confidence)
            return await route_image_generation_request(
                user_id=user_id,
                prompt=pending_image_prompt,
                original_text=text
            )

        logger.info("route_text_request done | user_id=%s, mode=%s, response_len=%s", user_id, mode, len(result.text))
        return {
            "text": result.text,
            "mode": result.mode
        }

    except Exception as e:
        # Wraps app.text_chat (validation, admission control, timeout, and
        # RAG/provider calls) and detect_image_generation_intent() — never
        # log or return raw exception text.
        logger.error("route_text_request failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        return {
            "text": "Извините, произошла ошибка при обработке запроса.",
            "error": type(e).__name__
        }


async def route_voice_request(
    user_id: uuid.UUID,
    voice_path: Path
) -> Dict[str, Any]:
    """
    Route voice request: transcribe, process, and generate voice response.

    Args:
        user_id: Canonical internal user UUID (Stage 5C)
        voice_path: Path to voice message file

    Returns:
        Response dictionary with 'text', 'transcription', 'voice_path', and optional 'image_path'
    """
    try:
        logger.info("route_voice_request start | user_id=%s", user_id)
        transcription = await transcribe_voice_message(voice_path)
        logger.info("route_voice_request: transcribed | user_id=%s, len=%s", user_id, len(transcription))

        # Process text request (may include image generation)
        text_response = await route_text_request(user_id, transcription)

        # Check if response contains an image
        if text_response.get('has_image'):
            logger.info("route_voice_request: image response (no TTS) | user_id=%s", user_id)
            return {
                "text": text_response["text"],
                "transcription": transcription,
                "has_image": True,
                "image_path": text_response.get("image_path"),
                "revised_prompt": text_response.get("revised_prompt"),
                "voice_path": None  # No voice response when image is generated
            }

        user_voice = await user_sessions.get_voice(user_id)
        plain_text = strip_markdown(text_response["text"])
        logger.debug("route_voice_request: generating TTS | user_id=%s, voice=%s", user_id, user_voice)
        voice_response_path = await generate_voice_response(plain_text, voice=user_voice)
        logger.info("route_voice_request done | user_id=%s", user_id)
        return {
            "text": plain_text,
            "transcription": transcription,
            "voice_path": voice_response_path,
            "has_image": False
        }

    except Exception as e:
        logger.error("route_voice_request failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        return {
            "text": "Извините, произошла ошибка при обработке голосового сообщения.",
            "error": type(e).__name__
        }


async def route_image_request(
    user_id: uuid.UUID,
    image_path: Optional[Path] = None,
    image_url: Optional[str] = None,
    caption: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route image request: analyze image with Vision API.

    Args:
        user_id: Canonical internal user UUID (Stage 5C)
        image_path: Local path to image
        image_url: URL to image
        caption: Optional caption/question about image

    Returns:
        Response dictionary with 'text' (analysis result)
    """
    try:
        # Prepare custom prompt if caption provided
        custom_prompt = None
        if caption:
            custom_prompt = f"{caption}\n\nПроанализируй изображение с учетом этого вопроса."

        logger.info("route_image_request | user_id=%s, has_caption=%s", user_id, bool(caption))
        analysis = await analyze_image(
            image_path=image_path,
            image_url=image_url,
            custom_prompt=custom_prompt
        )

        # Add to conversation history — ONE atomic exchange (Stage 7A-1
        # second corrective pass), never two independently observable
        # add_message() calls a concurrent thread observer could see a
        # half of.
        context = f"[Пользователь отправил изображение]"
        if caption:
            context += f" с подписью: {caption}"
        user_sessions.add_exchange(user_id, context, analysis)

        logger.info("route_image_request done | user_id=%s, analysis_len=%s", user_id, len(analysis))
        return {"text": analysis}
    except Exception as e:
        logger.error("route_image_request failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        return {
            "text": "Извините, произошла ошибка при анализе изображения.",
            "error": type(e).__name__
        }


async def route_image_generation_request(
    user_id: uuid.UUID,
    prompt: str,
    original_text: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid"
) -> Dict[str, Any]:
    """
    Route image generation request: generate image with DALL-E.

    Args:
        user_id: Canonical internal user UUID (Stage 5C)
        prompt: Processed prompt for image generation
        original_text: Original user text
        size: Image size (1024x1024, 1024x1792, 1792x1024)
        quality: Image quality (standard, hd)
        style: Image style (vivid, natural)

    Returns:
        Response dictionary with 'text', 'image_path', and generation details
    """
    try:
        logger.info("route_image_generation | user_id=%s, prompt_len=%s", user_id, len(prompt))

        # Generate image
        result = await generate_image(
            prompt=prompt,
            size=size,
            quality=quality,
            style=style
        )

        # Prepare response text
        response_text = "🎨 Изображение создано!\n\n"
        if result['revised_prompt'] != result['original_prompt']:
            response_text += f"Улучшенный промпт:\n{result['revised_prompt']}\n\n"
        response_text += "Вот что получилось:"

        # Add to conversation history — ONE atomic exchange (Stage 7A-1
        # second corrective pass), never two independently observable
        # add_message() calls.
        user_sessions.add_exchange(
            user_id,
            f"[Запрос на генерацию изображения: {original_text}]",
            f"[Изображение создано: {result['revised_prompt'][:100]}...]"
        )

        logger.info("route_image_generation done | user_id=%s, has_image=True", user_id)
        return {
            "text": response_text,
            "image_path": result['image_path'],
            "revised_prompt": result['revised_prompt'],
            "original_prompt": result['original_prompt'],
            "has_image": True
        }

    except Exception as e:
        # Wraps a direct DALL-E HTTPS call carrying the Authorization
        # header — never log raw exception text or a traceback. The
        # str(e).lower() checks below only classify which pre-written
        # generic message to show; they never surface e's content itself.
        logger.error("route_image_generation failed | user_id=%s, error_type=%s", user_id, type(e).__name__)

        error_message = "❌ Извините, произошла ошибка при генерации изображения. "

        if "billing" in str(e).lower() or "quota" in str(e).lower():
            error_message += "Возможно, исчерпан лимит API. Проверьте баланс OpenAI."
        elif "content_policy" in str(e).lower():
            error_message += "Запрос нарушает политику контента OpenAI."
        else:
            error_message += f"Попробуйте еще раз или перефразируйте запрос."

        # ONE atomic exchange (Stage 7A-1 second corrective pass) — never
        # an orphaned user-only turn observable between two separate calls.
        user_sessions.add_exchange(
            user_id, f"[Запрос на генерацию изображения: {original_text}]", error_message
        )

        return {
            "text": error_message,
            "error": type(e).__name__,
            "has_image": False
        }
