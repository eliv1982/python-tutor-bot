"""
Text-to-Speech Service.
Handles text to voice conversion.
"""

from pathlib import Path
from typing import Optional

from services.openai_client import openai_client
from utils.logging import logger
from config import VoiceType, DEFAULT_VOICE


async def generate_voice_response(
    text: str,
    voice: str = DEFAULT_VOICE
) -> Path:
    """
    Generate voice response from text.
    
    Args:
        text: Text to convert to speech
        voice: Voice type to use
    
    Returns:
        Path to generated audio file
    """
    try:
        # Validate voice type
        if voice not in VoiceType.ALL:
            logger.warning("TTS: invalid voice=%s, using default", voice)
            voice = DEFAULT_VOICE
        logger.debug("TTS generate_voice_response | voice=%s, text_len=%s", voice, len(text))
        audio_path = await openai_client.generate_speech(text, voice=voice)
        logger.info("TTS done | name=%s", audio_path.name)
        return audio_path
    except Exception as e:
        # Wraps an OpenAI TTS API call — never log raw exception text.
        logger.error("TTS failed | voice=%s, error_type=%s", voice, type(e).__name__)
        raise


def get_voice_info(voice: str) -> dict:
    """
    Get information about a voice type.
    
    Args:
        voice: Voice identifier
    
    Returns:
        Dictionary with voice information
    """
    voices = {
        VoiceType.ALLOY: {
            "name": "Alloy",
            "type": "Нейтральный",
            "description": "Сбалансированный голос"
        },
        VoiceType.ECHO: {
            "name": "Echo",
            "type": "Мужской",
            "description": "Четкий мужской голос"
        },
        VoiceType.NOVA: {
            "name": "Nova",
            "type": "Женский",
            "description": "Энергичный женский голос"
        },
        VoiceType.FABLE: {
            "name": "Fable",
            "type": "Мужской (британский)",
            "description": "Британский акцент"
        },
        VoiceType.ONYX: {
            "name": "Onyx",
            "type": "Мужской (глубокий)",
            "description": "Глубокий мужской голос"
        },
        VoiceType.SHIMMER: {
            "name": "Shimmer",
            "type": "Женский (теплый)",
            "description": "Теплый женский голос"
        }
    }
    
    return voices.get(voice, voices[VoiceType.ALLOY])


def get_available_voices() -> str:
    """
    Get formatted list of available voices.
    
    Returns:
        Formatted string with voice information
    """
    result = "📢 Доступные голоса:\n\n"
    for voice in VoiceType.ALL:
        info = get_voice_info(voice)
        result += f"• {info['name']} ({voice})\n"
        result += f"  Тип: {info['type']}\n"
        result += f"  {info['description']}\n\n"
    
    return result

