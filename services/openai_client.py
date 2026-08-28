"""
OpenAI Client for the Personal Assistant Bot.
Provides methods for text generation, vision, STT, and TTS.
"""

from typing import List, Dict, Optional
from openai import AsyncOpenAI
from pathlib import Path

from config import (
    OPENAI_API_KEY,
    OFFICIAL_OPENAI_BASE_URL,
    GPT_MODEL,
    WHISPER_MODEL,
    TTS_MODEL,
    VISION_MODEL,
    TEMPERATURE,
    MAX_TOKENS
)
from utils.logging import logger

# The only image representation this application ever sends to OpenAI's
# vision endpoint. Rejecting anything else (in particular http(s):// URLs)
# guarantees a Telegram file URL/bot token can never reach OpenAI.
SAFE_IMAGE_URL_PREFIX = "data:image/"


class OpenAIClient:
    """Async client for OpenAI API operations."""

    def __init__(self):
        """Initialize the OpenAI client."""
        # Pinned explicitly (not left to SDK defaults) because the openai
        # SDK itself falls back to an OPENAI_BASE_URL *environment
        # variable* when base_url isn't passed — a stray leftover in a
        # developer's .env would otherwise silently re-route requests.
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OFFICIAL_OPENAI_BASE_URL)
        logger.info("OpenAI client initialized with official OpenAI API")
    
    async def generate_text_response(
        self,
        messages: List[Dict[str, str]],
        model: str = GPT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """
        Generate text response using GPT model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use
            temperature: Response randomness (0-2)
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated text response
        """
        try:
            logger.debug("OpenAI chat.completions.create | model=%s, messages=%s", model, len(messages))
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            tokens = f", usage={usage}" if usage else ""
            logger.info("OpenAI text response | model=%s, len=%s%s", model, len(result), tokens)
            return result
        except Exception as e:
            # OpenAI SDK exceptions (HTTP/auth/rate-limit) are external and
            # must never be logged raw — only their class name is safe.
            logger.error("OpenAI text response failed | model=%s, error_type=%s", model, type(e).__name__)
            raise
    
    async def analyze_image(
        self,
        image_url: str,
        prompt: str = "Опиши это изображение подробно. Что ты видишь?",
        model: str = VISION_MODEL
    ) -> str:
        """
        Analyze an image using GPT-4 Vision.

        Args:
            image_url: Base64 data URL (data:image/...;base64,...). This is
                the provider boundary: any other value (an http(s):// URL,
                including a Telegram file URL) is rejected so a Telegram
                bot token can never be transmitted to OpenAI.
            prompt: Analysis prompt
            model: Vision model to use

        Returns:
            Image analysis result
        """
        if not image_url.startswith(SAFE_IMAGE_URL_PREFIX):
            logger.error("OpenAI vision rejected unsafe image reference | expected_prefix=%s", SAFE_IMAGE_URL_PREFIX)
            raise ValueError(
                "analyze_image only accepts base64 data URLs (data:image/...;base64,...); "
                "remote URLs are not supported"
            )
        try:
            logger.debug("OpenAI vision | model=%s, prompt_len=%s", model, len(prompt))
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS
            )
            
            result = response.choices[0].message.content
            logger.info("OpenAI vision done | model=%s, result_len=%s", model, len(result))
            return result
        except Exception as e:
            logger.error("OpenAI vision failed | model=%s, error_type=%s", model, type(e).__name__)
            raise
    
    async def transcribe_audio(
        self,
        audio_file_path: Path,
        model: str = WHISPER_MODEL
    ) -> str:
        """
        Transcribe audio file to text using Whisper.
        
        Args:
            audio_file_path: Path to audio file
            model: Whisper model to use
        
        Returns:
            Transcribed text
        """
        try:
            logger.info("OpenAI Whisper transcribe | model=%s", model)
            with open(audio_file_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="text"
                )
            
            logger.info("OpenAI Whisper done | result_len=%s", len(response))
            return response
        except Exception as e:
            logger.error("OpenAI Whisper failed | error_type=%s", type(e).__name__)
            raise
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = TTS_MODEL,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate speech from text using TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, nova, fable, onyx, shimmer)
            model: TTS model to use
            output_path: Path to save audio file
        
        Returns:
            Path to generated audio file
        """
        try:
            logger.info("OpenAI TTS | voice=%s, text_len=%s", voice, len(text))
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            # Default output path
            if output_path is None:
                from config import DATA_DIR
                import uuid
                output_path = DATA_DIR / f"tts_{uuid.uuid4()}.mp3"
            
            # Save audio to file
            response.stream_to_file(str(output_path))
            
            logger.info("OpenAI TTS done | name=%s", output_path.name)
            return output_path
        except Exception as e:
            logger.error("OpenAI TTS failed | voice=%s, error_type=%s", voice, type(e).__name__)
            raise


# Global client instance
openai_client = OpenAIClient()

