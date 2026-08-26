"""
Voice Message Handler.
Handles voice messages with STT and TTS using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from services.router import route_voice_request
from services.tts import get_available_voices, get_voice_info
from utils.logging import logger
from utils.helpers import user_sessions, save_file_async, cleanup_files, strip_markdown, download_telegram_file
from config import VoiceType


@bot.message_handler(commands=['voice'])
async def cmd_voice(message: types.Message):
    """Handle /voice command - change TTS voice."""
    user_id = message.from_user.id
    
    # Parse command arguments
    args = message.text.split(maxsplit=1)
    
    if len(args) < 2:
        # Show current voice and available voices
        current_voice = user_sessions.get_voice(user_id)
        current_info = get_voice_info(current_voice)
        
        voice_list = get_available_voices()
        
        await bot.send_message(
            message.chat.id,
            f"🔊 Текущий голос: {current_info['name']} ({current_voice})\n"
            f"Тип: {current_info['type']}\n\n"
            f"{voice_list}\n"
            f"Использование: /voice <название>\nПример: /voice nova"
        )
        return
    
    # Set new voice
    new_voice = args[1].lower()
    valid_voices = [
        VoiceType.ALLOY,
        VoiceType.ECHO,
        VoiceType.NOVA,
        VoiceType.FABLE,
        VoiceType.ONYX,
        VoiceType.SHIMMER
    ]
    
    if new_voice not in valid_voices:
        await bot.send_message(
            message.chat.id,
            f"❌ Неизвестный голос: {new_voice}\n\nИспользуйте /voice для списка доступных голосов."
        )
        return
    
    user_sessions.set_voice(user_id, new_voice)
    logger.info(f"User {user_id} switched to voice: {new_voice}")
    
    voice_info = get_voice_info(new_voice)
    
    await bot.send_message(
        message.chat.id,
        f"✅ Голос изменен!\n\n"
        f"🔊 {voice_info['name']} ({new_voice})\n"
        f"Тип: {voice_info['type']}\n"
        f"{voice_info['description']}"
    )


@bot.message_handler(commands=['voices'])
async def cmd_voices(message: types.Message):
    """Handle /voices command - list all available voices."""
    voice_list = get_available_voices()
    await bot.send_message(message.chat.id, voice_list)


@bot.message_handler(content_types=['voice'])
async def handle_voice_message(message: types.Message):
    """Handle voice messages."""
    user_id = message.from_user.id
    
    logger.info("Voice message | user_id=%s, file_id=%s", user_id, message.voice.file_id if message.voice else None)
    
    await bot.send_chat_action(message.chat.id, 'typing')
    
    voice_file_path = None
    audio_response_path = None
    image_path = None
    
    try:
        # Download voice message
        voice_bytes, _ = await download_telegram_file(bot, message.voice.file_id, operation="voice_download")

        # Save to temporary file
        voice_file_path = await save_file_async(voice_bytes, "ogg")
        
        logger.debug("Voice file saved | user_id=%s, path=%s", user_id, voice_file_path)
        
        # Process voice request
        response = await route_voice_request(user_id, voice_file_path)
        
        # Send transcription
        await bot.send_message(
            message.chat.id,
            f"🎤 Распознано: {response['transcription']}\n"
        )
        
        # Check if response contains an image
        if response.get('has_image') and response.get('image_path'):
            logger.info("Voice response: image generated | user_id=%s", user_id)
            await bot.send_message(message.chat.id, strip_markdown(response["text"]))
            
            # Then send the generated image
            image_path = response['image_path']
            
            try:
                await bot.send_chat_action(message.chat.id, 'upload_photo')
                with open(image_path, 'rb') as photo:
                    await bot.send_photo(message.chat.id, photo)
                logger.debug("Voice: image sent | user_id=%s", user_id)
            except Exception as img_error:
                logger.error("Voice: failed to send image | user_id=%s, error=%s", user_id, img_error)
            
            return
        
        # Show voice action
        await bot.send_chat_action(message.chat.id, 'record_voice')
        
        await bot.send_message(message.chat.id, strip_markdown(response["text"]))
        # Send voice response
        audio_response_path = response.get("voice_path")
        if audio_response_path:
            with open(audio_response_path, 'rb') as audio:
                await bot.send_voice(message.chat.id, audio)
    
    except Exception as e:
        logger.error("Voice message failed | user_id=%s, error=%s", user_id, e, exc_info=True)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при обработке голосового сообщения.\n"
            "Попробуйте еще раз."
        )
    
    finally:
        # Cleanup temporary files
        cleanup_files(voice_file_path, audio_response_path, image_path)


@bot.message_handler(content_types=['audio'])
async def handle_audio_message(message: types.Message):
    """Handle audio files (similar to voice)."""
    await bot.send_message(
        message.chat.id,
        "ℹ️ Для обработки аудио используйте голосовые сообщения.\n"
        "Аудиофайлы в виде документов не поддерживаются."
    )
