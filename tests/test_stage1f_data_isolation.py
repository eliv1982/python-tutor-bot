"""
Stage 1F-B remediation regression tests (Codex Blocker 2).

An independent audit demonstrated that tests/test_stage1d_privacy_logging.py
::test_voice_stt_failure_leaks_nothing_and_returns_generic_message — which
calls the REAL handlers.voice.handle_voice_message(), mocking only the
Telegram bot and the transcription call — ended up writing a real file
under the repository's real `data/` directory. Normal cleanup removed it
(handle_voice_message's own `finally: cleanup_files(...)`), but a read-only
checkout would make the test fail outright, and an interrupted run could
leave residue.

Root cause: utils/helpers.py's save_file_async() (the only thing
handlers/voice.py uses to persist a downloaded voice message before STT)
built its path as `BASE_DIR / "data" / filename` — BASE_DIR is
`Path(__file__).parent` in config.py, never redirected by anything —
instead of reading `config.DATA_DIR`, which tests/conftest.py's
pytest_configure DOES redirect to a session-temp directory. So the
isolation that already protected rag/index.py's Chroma store and
utils/logging.py's log file never covered this path at all.

Fix: save_file_async() now reads config.DATA_DIR (identical production
behavior: config.DATA_DIR == BASE_DIR / "data" outside pytest), and
conftest.py's redirect of config.DATA_DIR is now held for the entire test
session (previously it was redirected only for the duration of two
proactive imports, then restored to the real path — which is exactly the
window in which a later, ordinarily-imported module like utils.helpers
would have bound to the real path instead).
"""

from pathlib import Path

import pytest

import config as app_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = (PROJECT_ROOT / "data").resolve()


def test_config_data_dir_is_redirected_away_from_the_real_repo_data_dir():
    """
    The property every other test/proof in this file (and the real voice
    handler, transitively) depends on: for the whole session,
    config.DATA_DIR must never resolve into the real repo's data/
    directory. tests/conftest.py's pytest_configure sets this once, before
    any project module is imported, and never restores it.
    """
    resolved = app_config.DATA_DIR.resolve()
    assert resolved != REAL_DATA_DIR
    assert REAL_DATA_DIR not in resolved.parents


@pytest.mark.asyncio
async def test_save_file_async_writes_only_into_the_redirected_temp_dir():
    """
    Calls the REAL save_file_async() (no mocking) — the same function
    handlers/voice.py:100 calls for every real voice message — and proves
    the resulting file lives under the redirected (temp) DATA_DIR and
    nowhere under the real repo's data/ directory.
    """
    from utils.helpers import cleanup_file, save_file_async

    path = await save_file_async(b"fake ogg bytes - isolation proof only", "ogg")
    try:
        assert path.exists()
        resolved = path.resolve()
        assert resolved.parent == app_config.DATA_DIR.resolve()
        assert resolved != REAL_DATA_DIR
        assert REAL_DATA_DIR not in resolved.parents
    finally:
        cleanup_file(path)


@pytest.mark.asyncio
async def test_real_voice_handler_path_does_not_touch_real_repo_data_dir(monkeypatch):
    """
    End-to-end proof at the actual vulnerable call site: invokes the real
    handlers.voice.handle_voice_message() (only the Telegram bot calls and
    the STT/router step are mocked — download_telegram_file and
    save_file_async both run for real, exactly as in the Codex-flagged
    test), and asserts nothing appears under the real repo's data/
    directory before or after, by snapshotting its contents.

    Deliberately does not touch the real filesystem's data/ permissions or
    mutate it in any way — this is a pure before/after listing comparison.
    """
    import handlers.voice as voice_handler
    import app.tutor as router_module
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    before = sorted(p.name for p in REAL_DATA_DIR.iterdir()) if REAL_DATA_DIR.exists() else None

    monkeypatch.setattr(
        voice_handler.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="voice/file_1.oga")),
    )
    monkeypatch.setattr(voice_handler.bot, "download_file", AsyncMock(return_value=b"fake ogg bytes"))
    monkeypatch.setattr(voice_handler.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(voice_handler.bot, "send_message", AsyncMock())
    monkeypatch.setattr(
        router_module, "transcribe_voice_message",
        AsyncMock(side_effect=Exception("boom")),
    )

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=424242),
        chat=SimpleNamespace(id=424242),
        voice=SimpleNamespace(file_id="fake-voice-id"),
    )

    await voice_handler.handle_voice_message(message)

    after = sorted(p.name for p in REAL_DATA_DIR.iterdir()) if REAL_DATA_DIR.exists() else None
    assert after == before, (
        f"real repo data/ directory changed during a test run: before={before}, after={after}"
    )
