"""
Stage 4 closure regression tests: provider-neutral Telegram wording.

Scope: handlers/text.py's text-mode descriptions (callback_mode and
cmd_mode) and handlers/start.py's cmd_help stack line no longer claim
GPT-4o is the (universal) text-generation model — the actual text
provider is selectable and defaults to Anthropic Claude (see
config.py/services/text_llm.py, exercised separately and thoroughly by
tests/test_stage2a_text_llm_provider.py). Wording is STATIC
provider-neutral per the approved decision — these tests assert on the
actual rendered/sent text, not on config.LLM_PROVIDER, and never monkeypatch
LLM_PROVIDER themselves.

No provider/client/config logic is touched or exercised here; this module
only proves user-visible wording, using this repository's existing
handler-test convention (bare telebot.types objects built via __new__,
bot.send_message/answer_callback_query mocked on the single shared `bot`
instance — see tests/test_stage1c_access_control.py).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telebot import types

import handlers.start as start
import handlers.text as text
from bot import bot as shared_bot
from utils.helpers import user_sessions


@pytest.fixture(autouse=True)
def _clean_sessions():
    user_sessions.sessions.clear()
    yield
    user_sessions.sessions.clear()


@pytest.fixture(autouse=True)
def _mock_bot_send(monkeypatch):
    monkeypatch.setattr(shared_bot, "send_message", AsyncMock())
    monkeypatch.setattr(shared_bot, "answer_callback_query", AsyncMock())


def _new_message() -> types.Message:
    return types.Message.__new__(types.Message)


def _new_callback() -> types.CallbackQuery:
    return types.CallbackQuery.__new__(types.CallbackQuery)


def _make_text_message(user_id: int, text_: str):
    message = _new_message()
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.text = text_
    message.content_type = "text"
    return message


def _make_callback(user_id: int, data: str):
    callback = _new_callback()
    callback.id = "cb-1"
    callback.data = data
    callback.from_user = SimpleNamespace(id=user_id, first_name="Test")
    callback.message = SimpleNamespace(chat=SimpleNamespace(id=user_id))
    return callback


USER_ID = 555555555


@pytest.mark.asyncio
async def test_mode_callback_text_wording_is_provider_neutral():
    """The inline-keyboard 'switched to text mode' message no longer claims
    GPT-4o, and still names the mode clearly."""
    callback = _make_callback(USER_ID, data="mode_text")
    await text.callback_mode(callback)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert "GPT-4o" not in sent_text
    assert "Текстовый режим — диалог по Python" in sent_text


@pytest.mark.asyncio
async def test_mode_command_text_wording_is_provider_neutral():
    """/mode text's confirmation message no longer claims GPT-4o."""
    message = _make_text_message(USER_ID, "/mode text")
    await text.cmd_mode(message)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert "GPT-4o" not in sent_text
    assert "Текстовый режим — диалог по Python" in sent_text


@pytest.mark.asyncio
async def test_help_mode_list_wording_is_provider_neutral():
    """/help's '/mode text — ...' line no longer claims GPT-4o."""
    message = _make_text_message(USER_ID, "/help")
    await start.cmd_help(message)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert "/mode text — текстовый диалог по Python" in sent_text
    assert "/mode text — текстовый диалог по Python (GPT-4o)" not in sent_text


@pytest.mark.asyncio
async def test_help_stack_wording_no_longer_claims_gpt4o_as_universal_llm():
    """/help's stack summary no longer names GPT-4o as the (universal) text
    LLM, and instead names both selectable text providers — accurate under
    either configured LLM_PROVIDER value, without reading config at all."""
    message = _make_text_message(USER_ID, "/help")
    await start.cmd_help(message)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert "GPT-4o" not in sent_text
    assert "Стек: LLM (Anthropic/OpenAI), Whisper, TTS, Vision, Qdrant (RAG)." in sent_text
    # Non-text OpenAI-backed capabilities remain named exactly as before —
    # only the text-LLM claim changed.
    assert "Whisper" in sent_text
    assert "TTS" in sent_text
    assert "Vision" in sent_text
    assert "Qdrant (RAG)" in sent_text
