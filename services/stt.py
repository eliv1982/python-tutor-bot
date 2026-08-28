"""
Speech-to-Text Service.
Handles voice message transcription.
"""

import asyncio
from pathlib import Path
from typing import Union

from services.openai_client import openai_client
from utils.logging import logger
from utils.helpers import convert_ogg_to_wav, cleanup_file, submit_worker, await_worker


def _resolve_cancelled_conversion(conversion_future: "asyncio.Future", expected_wav_path: Path) -> None:
    """
    Called from `transcribe_voice_message`'s `except asyncio.CancelledError:`
    handler around the OGG->WAV conversion step, after
    `await_worker(conversion_future)` has already blocked until
    `conversion_future` reached a terminal state — genuine terminal state of
    the executor thread itself (see `utils.helpers.submit_worker()`), not
    merely of an asyncio Task wrapping it — so ffmpeg/pydub is guaranteed to
    no longer be reading/writing any file at this point, and it is safe for
    handlers/voice.py's own `finally` cleanup of the OGG input to run right
    after this returns.

    Uses `expected_wav_path` (deterministically derived from the OGG path
    via `Path.with_suffix('.wav')`, same as `convert_ogg_to_wav()` itself)
    rather than relying on `conversion_future.result()` having been assigned
    to a local variable — on the cancelled path, the `wav_path = ...`
    assignment in `transcribe_voice_message` never ran. `cleanup_file()` is
    a no-op for a path that doesn't exist, so calling it unconditionally
    below is safe whether or not the worker actually produced a WAV file.
    """
    if conversion_future.cancelled():
        cleanup_file(expected_wav_path)
        return
    exc = conversion_future.exception()
    if exc is not None:
        # convert_ogg_to_wav() already sanitizes its own failure log
        # (Stage 1D) — this only records that cancellation raced it, never
        # the raw exception text.
        logger.warning(
            "STT: cancelled during OGG->WAV conversion, worker failed | error_type=%s",
            type(exc).__name__
        )
        cleanup_file(expected_wav_path)
        return
    # Conversion succeeded despite cancellation: the WAV is on disk but no
    # longer needed by anything — clean it up before letting cancellation
    # propagate further.
    logger.warning("STT: cancelled during OGG->WAV conversion, worker succeeded, cleaning up WAV")
    cleanup_file(conversion_future.result())


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
            # pydub shells out to ffmpeg — blocking subprocess/library work,
            # run via submit_worker()/await_worker() (see
            # _resolve_cancelled_conversion()): if this coroutine is
            # cancelled — even repeatedly, or as part of a broad/shutdown-
            # style sweep — while ffmpeg is still running, the worker is
            # never abandoned, and the caller-side OGG cleanup in
            # handlers/voice.py cannot race a still-running conversion.
            expected_wav_path = audio_path.with_suffix('.wav')
            conversion_future = submit_worker(convert_ogg_to_wav, audio_path)
            try:
                wav_path = await await_worker(conversion_future)
            except asyncio.CancelledError:
                _resolve_cancelled_conversion(conversion_future, expected_wav_path)
                raise
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
        # Cleanup converted file if created (cancellation path already
        # cleaned it up above and never reaches this with wav_path set).
        if wav_path and wav_path != audio_path:
            cleanup_file(wav_path)

