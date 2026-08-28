"""
Speech-to-Text Service.
Handles voice message transcription.
"""

from pathlib import Path
from typing import Union

from services.openai_client import openai_client
from utils.logging import logger
from utils.helpers import convert_ogg_to_wav, cleanup_file


async def transcribe_voice_message(audio_path: Union[str, Path]) -> str:
    """
    Transcribe a voice message to text.
    
    Args:
        audio_path: Path to audio file (OGG or WAV)
    
    Returns:
        Transcribed text
    """
    audio_path = Path(audio_path)
    wav_path = None
    
    try:
        # Convert OGG to WAV if needed
        if audio_path.suffix.lower() == '.ogg':
            logger.debug("STT: converting OGG to WAV | name=%s", audio_path.name)
            wav_path = convert_ogg_to_wav(audio_path)
            transcription_path = wav_path
        else:
            transcription_path = audio_path
        
        # Transcribe using OpenAI Whisper
        text = await openai_client.transcribe_audio(transcription_path)
        
        logger.info("STT transcription done | len=%s", len(text))
        return text
    except Exception as e:
        # Wraps an OpenAI Whisper API call — never log raw exception text.
        logger.error("STT transcription failed | error_type=%s", type(e).__name__)
        raise
        
    finally:
        # Cleanup converted file if created
        if wav_path and wav_path != audio_path:
            cleanup_file(wav_path)

