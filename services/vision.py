"""
Vision Service.
Handles image analysis using GPT-4 Vision.
"""

from typing import Optional
import base64
from pathlib import Path

from services.openai_client import openai_client
from utils.logging import logger


async def analyze_image(
    image_path: Optional[Path] = None,
    image_url: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> str:
    """
    Analyze an image using GPT-4 Vision.
    
    Args:
        image_path: Local path to image file
        image_url: URL to image (Telegram file URL)
        custom_prompt: Custom analysis prompt
    
    Returns:
        Analysis result
    """
    try:
        logger.debug("Vision analyze_image | has_url=%s, has_path=%s, custom_prompt=%s", bool(image_url), bool(image_path), bool(custom_prompt))
        if image_url:
            final_url = image_url
        elif image_path:
            # Convert local image to base64
            final_url = encode_image_to_base64(image_path)
        else:
            raise ValueError("Either image_path or image_url must be provided")
        
        # Default prompt
        if custom_prompt is None:
            custom_prompt = """Проанализируй это изображение подробно:
            
            1. Что изображено на фото?
            2. Опиши основные объекты и детали
            3. Если это документ - извлеки текст и данные
            4. Если это объект - опиши его назначение
            5. Дай полезную информацию или рекомендации
            
            Ответь на русском языке."""
        
        # Analyze image
        logger.debug("Analyzing image with Vision API")
        result = await openai_client.analyze_image(
            image_url=final_url,
            prompt=custom_prompt
        )
        
        logger.info("Vision analyze_image done | result_len=%s", len(result))
        return result
    except Exception as e:
        logger.error("Vision analyze_image failed | error=%s", e, exc_info=True)
        raise


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode image file to base64 data URL.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Base64 encoded data URL
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine MIME type
        extension = image_path.suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(extension, 'image/jpeg')
        
        return f"data:{mime_type};base64,{base64_image}"
        
    except Exception as e:
        logger.error("Vision encode_image failed | path=%s, error=%s", image_path, e)
        raise


async def analyze_document_image(image_path: Path) -> str:
    """
    Specialized analysis for document images.
    
    Args:
        image_path: Path to document image
    
    Returns:
        Extracted text and analysis
    """
    prompt = """Это изображение документа. Пожалуйста:
    
    1. Извлеки весь текст из документа
    2. Определи тип документа (паспорт, договор, чек и т.д.)
    3. Извлеки ключевую информацию (даты, имена, суммы)
    4. Представь данные структурированно
    
    Ответь на русском языке."""
    
    return await analyze_image(image_path=image_path, custom_prompt=prompt)


async def analyze_object_image(image_path: Path) -> str:
    """
    Specialized analysis for object images.
    
    Args:
        image_path: Path to object image
    
    Returns:
        Object description and information
    """
    prompt = """Проанализируй объект на этом изображении:
    
    1. Что это за объект?
    2. Для чего он используется?
    3. Основные характеристики
    4. Интересные факты или рекомендации
    
    Ответь на русском языке."""
    
    return await analyze_image(image_path=image_path, custom_prompt=prompt)

