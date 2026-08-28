"""
Helper functions for the Personal Assistant Bot.
Provides utility functions for file operations, audio conversion, etc.
"""

import asyncio
import functools
import os
import re
import uuid
import aiofiles
from pathlib import Path
from typing import Optional, Tuple, Union

from config import BASE_DIR
from utils.logging import logger


def submit_worker(func, *args, **kwargs) -> "asyncio.Future":
    """
    Submit `func(*args, **kwargs)` to the running event loop's DEFAULT
    executor and return the resulting `asyncio.Future` directly.

    Deliberately NOT `asyncio.create_task(asyncio.to_thread(...))`.
    `asyncio.to_thread()` is itself just `await
    loop.run_in_executor(None, func)` — but wrapping that coroutine in a
    `Task` means the Task owns the suspension point, and cancelling a Task
    cancels whatever Future it is currently suspended on (its internal
    `_fut_waiter`). Doing that here would cancel the run_in_executor Future
    even while the callable is already RUNNING in the executor thread:
    `concurrent.futures.Future.cancel()` on already-running work fails
    silently, but the asyncio-level wrapper still flips to CANCELLED
    immediately regardless — so a cancelled Task's terminal state is NOT
    proof the underlying thread has actually stopped. Worse, that Task
    would be independently reachable (and cancellable) by anything that
    walks `asyncio.all_tasks()` (e.g. a shutdown sweep), bypassing any
    shielding a caller does around it.

    The Future returned here has none of that: it is never enumerated by
    `asyncio.all_tasks()` (it isn't a Task), and nothing in this module ever
    calls `.cancel()` on it — so it can only ever reach a terminal state
    because the executor thread itself actually finished (with a result or
    an exception). `await_worker()` below relies on exactly this guarantee.
    """
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


async def await_worker(future: "asyncio.Future"):
    """
    Await `future` (from `submit_worker()`) without ever abandoning the
    executor thread it represents, even under repeated — or shutdown-style,
    broad — cancellation of the calling Task.

    Normal path (never cancelled): behaves exactly like `await future` — a
    worker result is returned, a worker exception propagates normally.

    Cancelled path: the FIRST `asyncio.CancelledError` delivered to this
    coroutine is recorded and NOT allowed to escape immediately. Instead we
    loop and re-issue `asyncio.shield(future)`. Since shielding never cancels
    `future` itself (see `submit_worker()`), the worker thread keeps running
    completely undisturbed no matter how many further cancellations arrive
    at this same await point — each one is caught the same way, and only the
    ORIGINAL CancelledError is ever preserved. This is what lets a caller
    survive being cancelled twice (or N times) without abandoning the
    worker, and without an internal Task that a shutdown sweep over
    `asyncio.all_tasks()` could reach independently of this loop.

    Every `CancelledError` caught here has to be one of two different
    things, distinguished by `future.cancelled()`:

    - `future.cancelled()` is False: `future` itself is still alive (or
      finished normally/with an exception) — this CancelledError came from
      cancelling the outer `asyncio.shield(future)` wrapper, not `future`.
      This is the ordinary "caller was cancelled while the worker keeps
      running" case: record it (if it's the first) and loop again.
    - `future.cancelled()` is True: `future` itself is already terminal —
      cancelled, not merely "cancellation requested". There is no running
      worker left to wait for, so looping again would spin forever: every
      future re-`shield()` of an already-cancelled Future returns
      immediately and raises this same CancelledError again with nothing
      ever changing (see `submit_worker()` — nothing here calls `.cancel()`
      on `future`, but a caller could still pass in one that is already
      cancelled, e.g. never actually submitted). This case terminates the
      loop immediately: if an outer cancellation was already recorded, that
      ORIGINAL one stays the caller-visible outcome (the inner Future's own
      cancellation is discarded, never allowed to override it); otherwise
      this cancellation — genuinely `future`'s own — is re-raised as-is.

    Once `future` genuinely reaches a terminal state:
    - if a cancellation was recorded, the ORIGINAL `CancelledError` is
      re-raised — never a later one, and never a worker exception in its
      place. Callers are expected to inspect `future` themselves (now
      guaranteed done: `.cancelled()` / `.exception()` / `.result()`) in
      their own `except asyncio.CancelledError:` handler to decide what the
      worker's real outcome means for the resources it owns. A worker
      exception observed this way is retrieved via `future.exception()`
      before this function returns, so asyncio never reports it as
      "exception was never retrieved".
    - if no cancellation was ever recorded, a worker exception simply
      propagates from here like `await future` would.
    """
    first_cancellation: Optional[asyncio.CancelledError] = None
    while True:
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            if future.cancelled():
                # `future` itself is terminal — no worker left to wait for.
                # Looping again here would never suspend (every re-shield
                # of an already-done Future returns synchronously), which
                # is exactly the infinite-busy-loop this branch exists to
                # avoid. Surface whichever cancellation is caller-visible.
                if first_cancellation is not None:
                    raise first_cancellation
                raise
            if first_cancellation is None:
                first_cancellation = exc
            continue
        except Exception:
            if first_cancellation is not None:
                # The worker failed while we were already reconciling a
                # cancellation. That outcome belongs to the caller's own
                # resolver (via future.exception()) — it must not replace
                # the cancellation the caller is entitled to see.
                break
            raise
        else:
            break
    if first_cancellation is not None:
        raise first_cancellation
    return result


class TelegramDownloadError(RuntimeError):
    """
    Raised when downloading a file from Telegram fails.

    Deliberately carries only a fixed, safe message. pyTelegramBotAPI's
    underlying request errors can render request details that include the
    token-bearing `/file/bot<TOKEN>/...` URL, so the original exception
    must never be logged, chained (traceback), or surfaced to the user.
    """


async def download_telegram_file(bot, file_id: str, operation: str) -> Tuple[bytes, str]:
    """
    Download a Telegram file's bytes via the bot's own token-scoped calls.

    Args:
        bot: AsyncTeleBot instance
        file_id: Telegram file_id to resolve and download
        operation: Short label for logging (e.g. "photo_download")

    Returns:
        (file_bytes, file_path) — file_path is Telegram's internal path,
        useful only as a MIME-type hint; it carries no token.

    Raises:
        TelegramDownloadError: on any failure. Only the operation name and
        exception type are logged — never the raw exception or a traceback,
        since either can contain the token-bearing file URL.
    """
    try:
        file_info = await bot.get_file(file_id)
        file_bytes = await bot.download_file(file_info.file_path)
        return file_bytes, file_info.file_path
    except Exception as e:
        logger.error(
            "Telegram download failed | operation=%s, error_type=%s",
            operation, type(e).__name__
        )
        raise TelegramDownloadError(f"Telegram download failed during {operation}") from None


def strip_markdown(text: str) -> str:
    """
    Убирает разметку Markdown из текста для отображения в Telegram обычным текстом.
    Удаляет **, *, `, ###, ##, #, ```, ---, ссылки [text](url) и т.п.
    """
    if not text or not text.strip():
        return text
    t = text
    # Блоки кода: ```lang?\n...\n``` -> оставить только содержимое
    t = re.sub(r"```[\w]*\n([\s\S]*?)```", r"\1", t)
    # Оставшиеся ``` убрать
    t = t.replace("```", "")
    # Инлайн-код `...` -> ...
    t = re.sub(r"`([^`]+)`", r"\1", t)
    # **жирный** -> жирный
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    # *курсив* -> курсив (одиночные * не в паре оставляем)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    # Заголовки # ## ### -> убрать решётки
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    # Горизонтальная черта
    t = re.sub(r"^---+$", "", t, flags=re.MULTILINE)
    # Ссылки [текст](url) -> текст
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    # Убрать лишние пустые строки (более 2 подряд -> 2)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


async def save_file_async(file_content: bytes, extension: str = "tmp") -> Path:
    """
    Save file content asynchronously to a temporary file.
    
    Args:
        file_content: Binary content of the file
        extension: File extension (without dot)
    
    Returns:
        Path to the saved file
    """
    filename = f"{uuid.uuid4()}.{extension}"
    filepath = BASE_DIR / "data" / filename
    
    try:
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(file_content)
        logger.debug("File saved | name=%s", filepath.name)
        return filepath
    except Exception as e:
        # OSError messages commonly embed the full path (and therefore the
        # deployment's absolute directory structure) — log only the type.
        logger.error("Error saving file | error_type=%s", type(e).__name__)
        raise


def convert_ogg_to_wav(ogg_path: Union[str, Path]) -> Path:
    """
    Convert OGG audio file to WAV format using pydub.
    
    Args:
        ogg_path: Path to the OGG file
    
    Returns:
        Path to the converted WAV file
    """
    try:
        # Lazy import to avoid audioop error on startup
        from pydub import AudioSegment
    except ImportError as e:
        logger.error("pydub not available | error_type=%s", type(e).__name__)
        raise

    ogg_path = Path(ogg_path)
    wav_path = ogg_path.with_suffix('.wav')

    try:
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format='wav')
        logger.debug("Audio conversion done | name=%s", wav_path.name)
        return wav_path
    except Exception as e:
        # AudioSegment.from_ogg()/export() shell out to ffmpeg; its stderr
        # can embed absolute input paths, codec/container metadata, or
        # user-controlled media metadata — never log raw exception text.
        logger.error("Audio conversion failed | error_type=%s", type(e).__name__)
        raise


def cleanup_file(filepath: Union[str, Path, None]) -> None:
    """
    Delete a file safely. Ignores None.
    
    Args:
        filepath: Path to the file to delete (or None to skip)
    """
    if filepath is None:
        return
    try:
        filepath = Path(filepath)
        if filepath.exists():
            filepath.unlink()
            logger.debug("Cleaned up file | name=%s", filepath.name)
    except Exception as e:
        logger.warning("Error cleaning up file | error_type=%s", type(e).__name__)


def cleanup_files(*filepaths: Union[str, Path]) -> None:
    """
    Delete multiple files safely.
    
    Args:
        *filepaths: Paths to files to delete
    """
    for filepath in filepaths:
        cleanup_file(filepath)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


class UserSession:
    """Simple user session manager to store conversation history."""
    
    def __init__(self):
        self.sessions = {}
    
    def get_history(self, user_id: int) -> list:
        """Get conversation history for a user."""
        return self.sessions.get(user_id, [])
    
    def add_message(self, user_id: int, role: str, content: str):
        """Add a message to user's conversation history."""
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        
        self.sessions[user_id].append({
            "role": role,
            "content": content
        })
        
        # Limit history length
        from config import MAX_HISTORY_LENGTH
        if len(self.sessions[user_id]) > MAX_HISTORY_LENGTH * 2:
            self.sessions[user_id] = self.sessions[user_id][-MAX_HISTORY_LENGTH * 2:]
    
    def clear_history(self, user_id: int):
        """Clear conversation history for a user."""
        if user_id in self.sessions:
            del self.sessions[user_id]
    
    def get_mode(self, user_id: int) -> str:
        """Get current mode for a user."""
        return self.sessions.get(f"{user_id}_mode", "text")
    
    def set_mode(self, user_id: int, mode: str):
        """Set mode for a user."""
        self.sessions[f"{user_id}_mode"] = mode
    
    def get_voice(self, user_id: int) -> str:
        """Get current voice setting for a user."""
        from config import DEFAULT_VOICE
        return self.sessions.get(f"{user_id}_voice", DEFAULT_VOICE)
    
    def set_voice(self, user_id: int, voice: str):
        """Set voice for a user."""
        self.sessions[f"{user_id}_voice"] = voice

    def set_pending_image(self, user_id: int, image_data_url: str):
        """Сохранить base64 data URL изображения в ожидании вопроса от пользователя."""
        self.sessions[f"{user_id}_pending_image"] = image_data_url

    def get_pending_image(self, user_id: int) -> Optional[str]:
        """Получить base64 data URL изображения, ожидающего вопрос (или None)."""
        return self.sessions.get(f"{user_id}_pending_image")

    def clear_pending_image(self, user_id: int):
        """Сбросить ожидающее изображение."""
        if f"{user_id}_pending_image" in self.sessions:
            del self.sessions[f"{user_id}_pending_image"]


# Global session manager instance
user_sessions = UserSession()

