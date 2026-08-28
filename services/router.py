"""
Request Router.
Routes different types of requests to appropriate handlers.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from services import text_llm
from services.stt import transcribe_voice_message
from services.tts import generate_voice_response
from services.vision import analyze_image
from services.image_generation import detect_image_generation_intent, generate_image
from utils.logging import logger
from utils.helpers import user_sessions, strip_markdown
from config import BotMode


async def route_text_request(
    user_id: int,
    text: str,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route text request to appropriate handler.
    
    Args:
        user_id: User ID
        text: User's text message
        mode: Bot mode (text, rag, etc.)
    
    Returns:
        Response dictionary with 'text' and optional 'voice_path' and 'image_path'
    """
    try:
        if mode is None:
            mode = user_sessions.get_mode(user_id)
        history = user_sessions.get_history(user_id)
        logger.debug("route_text_request | user_id=%s, mode=%s, history_len=%s, text_len=%s", user_id, mode, len(history), len(text))
        image_intent = await detect_image_generation_intent(text, history)
        if image_intent.get('needs_generation') and image_intent.get('confidence', 0) > 0.5:
            logger.info("route_text_request: image generation | user_id=%s, confidence=%s", user_id, image_intent.get('confidence'))
            return await route_image_generation_request(
                user_id=user_id,
                prompt=image_intent.get('prompt', text),
                original_text=text
            )
        
        # Add user message to history
        user_sessions.add_message(user_id, "user", text)
        
        if mode == BotMode.RAG:
            logger.info("route_text_request: RAG | user_id=%s, query_len=%s", user_id, len(text))
            from rag.query import query_knowledge_base
            response_text = await query_knowledge_base(text, history)
        else:
            logger.debug("route_text_request: GPT | user_id=%s, model=default", user_id)
            system_prompt = (
                "Ты — персональный тьютор по Python. Отвечай на русском, кратко и по делу. "
                "Не используй разметку markdown — только обычный текст. Примеры кода пиши с отступом, без ** и без обратных кавычек. Объясняй концепции и лучшие практики."
            )
            messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": text}]
            response_text = await text_llm.generate_text_response(messages)
        
        # Add assistant response to history
        user_sessions.add_message(user_id, "assistant", response_text)
        
        logger.info("route_text_request done | user_id=%s, mode=%s, response_len=%s", user_id, mode, len(response_text))
        return {
            "text": response_text,
            "mode": mode
        }
        
    except Exception as e:
        # Wraps OpenAI/RAG calls — never log or return raw exception text.
        logger.error("route_text_request failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        return {
            "text": "Извините, произошла ошибка при обработке запроса.",
            "error": type(e).__name__
        }


async def route_voice_request(
    user_id: int,
    voice_path: Path
) -> Dict[str, Any]:
    """
    Route voice request: transcribe, process, and generate voice response.
    
    Args:
        user_id: User ID
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
        
        user_voice = user_sessions.get_voice(user_id)
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
    user_id: int,
    image_path: Optional[Path] = None,
    image_url: Optional[str] = None,
    caption: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route image request: analyze image with Vision API.
    
    Args:
        user_id: User ID
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
        
        # Add to conversation history
        context = f"[Пользователь отправил изображение]"
        if caption:
            context += f" с подписью: {caption}"
        user_sessions.add_message(user_id, "user", context)
        user_sessions.add_message(user_id, "assistant", analysis)
        
        logger.info("route_image_request done | user_id=%s, analysis_len=%s", user_id, len(analysis))
        return {"text": analysis}
    except Exception as e:
        logger.error("route_image_request failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        return {
            "text": "Извините, произошла ошибка при анализе изображения.",
            "error": type(e).__name__
        }


async def route_rag_request(
    user_id: int,
    query: str
) -> Dict[str, Any]:
    """
    Route RAG request: query knowledge base.
    
    Args:
        user_id: User ID
        query: User's query
    
    Returns:
        Response dictionary with 'text' and 'sources'
    """
    try:
        from rag.query import query_knowledge_base
        
        # Get conversation history
        history = user_sessions.get_history(user_id)
        
        # Query knowledge base
        logger.debug(f"Querying knowledge base for user {user_id}")
        response = await query_knowledge_base(query, history)
        
        # Add to history
        user_sessions.add_message(user_id, "user", query)
        user_sessions.add_message(user_id, "assistant", response)
        
        logger.info(f"RAG request processed for user {user_id}")
        return {
            "text": response,
            "mode": "rag"
        }
        
    except Exception as e:
        logger.error("route_rag_request failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
        # Fallback to regular text response
        return await route_text_request(user_id, query, mode=BotMode.TEXT)


async def route_image_generation_request(
    user_id: int,
    prompt: str,
    original_text: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid"
) -> Dict[str, Any]:
    """
    Route image generation request: generate image with DALL-E.
    
    Args:
        user_id: User ID
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
        
        # Add to conversation history
        user_sessions.add_message(user_id, "user", f"[Запрос на генерацию изображения: {original_text}]")
        user_sessions.add_message(
            user_id, 
            "assistant", 
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

        # Add error to history
        user_sessions.add_message(user_id, "user", f"[Запрос на генерацию изображения: {original_text}]")
        
        error_message = "❌ Извините, произошла ошибка при генерации изображения. "
        
        if "billing" in str(e).lower() or "quota" in str(e).lower():
            error_message += "Возможно, исчерпан лимит API. Проверьте баланс OpenAI."
        elif "content_policy" in str(e).lower():
            error_message += "Запрос нарушает политику контента OpenAI."
        else:
            error_message += f"Попробуйте еще раз или перефразируйте запрос."
        
        user_sessions.add_message(user_id, "assistant", error_message)
        
        return {
            "text": error_message,
            "error": type(e).__name__,
            "has_image": False
        }

