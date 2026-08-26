"""
Stage 1C regression tests: temporary Telegram access gate.

Covers:
- TELEGRAM_ALLOWED_USER_IDS parsing policy (utils.access_control.parse_allowed_user_ids)
- fail-closed behavior: missing/empty/fully-malformed allowlist denies everyone
- authorized users reach existing handler behavior unchanged
- unauthorized users are rejected BEFORE any protected work: no OpenAI/router
  calls, no Telegram file downloads, no filesystem writes, no RAG
  query/ingestion, and no UserSession creation/mutation
- the denial response reveals nothing about the allowlist/config

All Telegram/OpenAI/RAG boundaries are mocked. No network access is
performed by this test module. `access_control.TELEGRAM_ALLOWED_USER_IDS`
is monkeypatched per test (never the real environment), matching this
codebase's existing convention for import-time-bound singletons (see
tests/conftest.py).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from telebot import types

import handlers.document_upload as document_upload
import handlers.image as image
import handlers.start as start
import handlers.text as text
import handlers.voice as voice
import utils.access_control as access_control
from bot import bot as shared_bot
from utils.helpers import user_sessions

AUTHORIZED_ID = 111111111
UNAUTHORIZED_ID = 222222222


@pytest.fixture(autouse=True)
def _default_test_access_allowed():
    """
    Shadows conftest.py's same-named autouse fixture, which defaults every
    other test module to "authorized" so unrelated tests don't need to know
    about the Stage 1C gate. This module tests the gate itself, so it must
    NOT install that override — leave utils.access_control.is_authorized as
    the real function; each test below controls access explicitly via
    monkeypatch on TELEGRAM_ALLOWED_USER_IDS.
    """
    yield


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Reset the in-memory session store between tests (Stage 1A convention)."""
    user_sessions.sessions.clear()
    yield
    user_sessions.sessions.clear()


@pytest.fixture(autouse=True)
def _mock_bot_send(monkeypatch):
    """
    Every handler eventually calls bot.send_message / bot.answer_callback_query.
    Mocking them here (on the single shared `bot` instance every handler
    module imports) keeps every test offline by default; individual tests
    still assert on call counts/args via these same mocks.
    """
    monkeypatch.setattr(shared_bot, "send_message", AsyncMock())
    monkeypatch.setattr(shared_bot, "answer_callback_query", AsyncMock())
    monkeypatch.setattr(shared_bot, "send_chat_action", AsyncMock())


def _new_message() -> types.Message:
    """
    Bare `telebot.types.Message` instance, built via `__new__` to skip
    `__init__` (which demands many positional args no test fixture wants to
    fabricate) while still being a real `types.Message` for `isinstance`.

    Stage 1C.2: `access_control._deny()` routes the denial response by
    `isinstance(update, types.Message)` / `isinstance(update, types.CallbackQuery)`,
    not by attribute sniffing — real `Message` objects have their own
    top-level `.id` (an alias for `message_id`), so a plain `SimpleNamespace`
    standing in for a message could previously be routed like a callback by
    mistake. Fixtures must therefore be real instances of the type they
    represent, exactly as production updates from pyTelegramBotAPI are.
    """
    return types.Message.__new__(types.Message)


def _new_callback() -> types.CallbackQuery:
    """Bare `telebot.types.CallbackQuery` instance — see `_new_message`."""
    return types.CallbackQuery.__new__(types.CallbackQuery)


def _make_text_message(user_id: int, text_: str):
    message = _new_message()
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.text = text_
    message.content_type = "text"
    return message


def _make_photo_message(user_id: int, caption: str = "", file_id: str = "fake-photo-id"):
    message = _new_message()
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.caption = caption
    message.photo = [SimpleNamespace(file_id=file_id)]
    return message


def _make_voice_message(user_id: int, file_id: str = "fake-voice-id"):
    message = _new_message()
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.voice = SimpleNamespace(file_id=file_id)
    return message


def _make_document_message(user_id: int, file_name: str = "notes.txt", file_id: str = "fake-doc-id"):
    document = SimpleNamespace(file_name=file_name, mime_type="application/pdf", file_id=file_id, file_size=100)
    message = _new_message()
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.document = document
    return message, document


def _make_callback(user_id: int, data: str = "mode_voice", callback_id: str = "cb-1"):
    callback = _new_callback()
    callback.id = callback_id
    callback.data = data
    callback.from_user = SimpleNamespace(id=user_id, first_name="Test")
    callback.message = SimpleNamespace(chat=SimpleNamespace(id=user_id))
    return callback


# ---------------------------------------------------------------------------
# A. Fail closed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fail_closed_with_empty_allowlist_denies_any_user(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())

    message = _make_text_message(UNAUTHORIZED_ID, "/start")
    await start.cmd_start(message)

    shared_bot.send_message.assert_awaited_once()
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE

    # No protected work happened: no session was created for this user.
    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions


@pytest.mark.asyncio
async def test_fail_closed_denies_even_when_caller_happens_to_be_authorized_elsewhere(monkeypatch):
    """A non-empty but unrelated allowlist still denies a user not on it —
    fail closed is per-user, not merely "allowlist is non-empty"."""
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))

    message = _make_text_message(UNAUTHORIZED_ID, "/start")
    await start.cmd_start(message)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE
    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions


# ---------------------------------------------------------------------------
# B. Authorized user preserves existing behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authorized_user_reaches_handler_behavior(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))

    message = _make_text_message(AUTHORIZED_ID, "/start")
    await start.cmd_start(message)

    # Existing /start behavior: session mode initialized, welcome text sent.
    from config import DEFAULT_MODE
    assert user_sessions.get_mode(AUTHORIZED_ID) == DEFAULT_MODE
    shared_bot.send_message.assert_awaited_once()
    sent_text = shared_bot.send_message.await_args.args[1]
    assert "Привет" in sent_text
    assert sent_text != access_control.ACCESS_DENIED_MESSAGE


# ---------------------------------------------------------------------------
# C. Unauthorized text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unauthorized_text_message_does_not_call_router_or_mutate_session(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    route_mock = AsyncMock()
    monkeypatch.setattr(text, "route_text_request", route_mock)

    message = _make_text_message(UNAUTHORIZED_ID, "Explain list comprehensions")
    await text.handle_text_message(message)

    route_mock.assert_not_called()
    assert user_sessions.get_history(UNAUTHORIZED_ID) == []
    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE


# ---------------------------------------------------------------------------
# D. Unauthorized photo
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unauthorized_photo_does_not_download_or_call_vision(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    download_mock = AsyncMock()
    route_mock = AsyncMock()
    monkeypatch.setattr(image, "download_telegram_file", download_mock)
    monkeypatch.setattr(image, "route_image_request", route_mock)

    message = _make_photo_message(UNAUTHORIZED_ID, caption="What is wrong with this code?")
    await image.handle_photo_message(message)

    download_mock.assert_not_called()
    route_mock.assert_not_called()
    assert user_sessions.get_pending_image(UNAUTHORIZED_ID) is None
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE


@pytest.mark.asyncio
async def test_unauthorized_photo_without_caption_does_not_store_pending_image(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    monkeypatch.setattr(image, "download_telegram_file", AsyncMock())

    message = _make_photo_message(UNAUTHORIZED_ID, caption="")
    await image.handle_photo_message(message)

    assert user_sessions.get_pending_image(UNAUTHORIZED_ID) is None


# ---------------------------------------------------------------------------
# E. Unauthorized voice
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unauthorized_voice_does_not_download_or_call_stt(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    download_mock = AsyncMock()
    route_mock = AsyncMock()
    monkeypatch.setattr(voice, "download_telegram_file", download_mock)
    monkeypatch.setattr(voice, "route_voice_request", route_mock)

    message = _make_voice_message(UNAUTHORIZED_ID)
    await voice.handle_voice_message(message)

    download_mock.assert_not_called()
    route_mock.assert_not_called()
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE


# ---------------------------------------------------------------------------
# F. Unauthorized document
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unauthorized_document_does_not_download_store_or_index(monkeypatch, tmp_path):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)
    process_mock = AsyncMock()
    monkeypatch.setattr(document_upload, "process_document_upload", process_mock)
    monkeypatch.setattr(document_upload.bot, "get_file", AsyncMock())

    message, _document = _make_document_message(UNAUTHORIZED_ID)
    await document_upload.handle_document_message(message)

    process_mock.assert_not_called()
    document_upload.bot.get_file.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE


# ---------------------------------------------------------------------------
# G. Unauthorized state-changing command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unauthorized_mode_command_does_not_mutate_session(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())

    message = _make_text_message(UNAUTHORIZED_ID, "/mode voice")
    await text.cmd_mode(message)

    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE
    assert "Режим изменён" not in sent_text


@pytest.mark.asyncio
async def test_unauthorized_reset_does_not_clear_or_touch_session(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    # Pre-existing state must survive an unauthorized /reset untouched.
    user_sessions.add_message(UNAUTHORIZED_ID, "user", "hello")
    user_sessions.set_pending_image(UNAUTHORIZED_ID, "data:image/png;base64,x")

    message = _make_text_message(UNAUTHORIZED_ID, "/reset")
    await start.cmd_reset(message)

    assert user_sessions.get_history(UNAUTHORIZED_ID) == [{"role": "user", "content": "hello"}]
    assert user_sessions.get_pending_image(UNAUTHORIZED_ID) == "data:image/png;base64,x"
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE


@pytest.mark.asyncio
async def test_unauthorized_mode_callback_does_not_mutate_session(monkeypatch):
    """Bonus coverage: the inline-keyboard callback entry point is gated too."""
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())

    callback = _make_callback(UNAUTHORIZED_ID, data="mode_voice")
    await text.callback_mode(callback)

    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions
    shared_bot.answer_callback_query.assert_awaited_once()
    assert shared_bot.answer_callback_query.await_args.args[1] == access_control.ACCESS_DENIED_MESSAGE
    shared_bot.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# H. Configuration parsing (utils.access_control.parse_allowed_user_ids)
# ---------------------------------------------------------------------------

def test_parse_single_valid_id():
    assert access_control.parse_allowed_user_ids("123456789") == frozenset({123456789})


def test_parse_multiple_valid_ids():
    assert access_control.parse_allowed_user_ids("111,222,333") == frozenset({111, 222, 333})


def test_parse_tolerates_surrounding_whitespace():
    assert access_control.parse_allowed_user_ids("  111 , 222 ,333  ") == frozenset({111, 222, 333})


def test_parse_collapses_duplicates():
    assert access_control.parse_allowed_user_ids("111,111,222,111") == frozenset({111, 222})


@pytest.mark.parametrize("raw", [None, "", "   ", ",, ,"])
def test_parse_empty_or_whitespace_only_denies_all(raw):
    assert access_control.parse_allowed_user_ids(raw) == frozenset()


def test_parse_policy_b_keeps_valid_entries_and_drops_malformed_ones():
    """
    Deliberate policy choice (Option B, documented in
    utils/access_control.py): a mix of valid and malformed entries keeps
    the valid ones rather than invalidating the whole configuration, so one
    typo can't lock out every correctly-configured operator id.
    """
    assert access_control.parse_allowed_user_ids("111,abc,222,") == frozenset({111, 222})
    assert access_control.parse_allowed_user_ids("not-a-number") == frozenset()


def test_parse_fully_malformed_input_denies_all():
    assert access_control.parse_allowed_user_ids("abc,def,xyz") == frozenset()


def test_is_authorized_uses_only_numeric_id(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))
    assert access_control.is_authorized(AUTHORIZED_ID) is True
    assert access_control.is_authorized(UNAUTHORIZED_ID) is False
    assert access_control.is_authorized(str(AUTHORIZED_ID)) is False  # no type coercion


# ---------------------------------------------------------------------------
# I. Malformed Telegram identity fails closed (Stage 1C.1 corrective pass)
# ---------------------------------------------------------------------------
#
# Even with a non-empty allowlist, a malformed/unusual update must never
# reach the protected handler, must never raise out of the decorator, and
# must never trigger any protected side effect.

@pytest.mark.asyncio
async def test_message_with_from_user_none_denies_and_does_not_call_handler(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))

    message = _new_message()
    message.from_user = None
    message.chat = SimpleNamespace(id=999)
    message.text = "/start"
    message.content_type = "text"
    await start.cmd_start(message)

    shared_bot.send_message.assert_awaited_once()
    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE
    assert "999_mode" not in user_sessions.sessions


@pytest.mark.asyncio
async def test_callback_with_from_user_none_denies_and_does_not_call_handler(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))

    callback = _new_callback()
    callback.id = "cb-malformed"
    callback.data = "mode_voice"
    callback.from_user = None
    callback.message = SimpleNamespace(chat=SimpleNamespace(id=999))
    await text.callback_mode(callback)

    shared_bot.answer_callback_query.assert_awaited_once()
    assert shared_bot.answer_callback_query.await_args.args[1] == access_control.ACCESS_DENIED_MESSAGE
    assert "999_mode" not in user_sessions.sessions


@pytest.mark.asyncio
async def test_from_user_missing_id_denies_and_does_not_call_handler(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))

    message = _new_message()
    message.from_user = SimpleNamespace(first_name="Ghost")  # no `id`
    message.chat = SimpleNamespace(id=999)
    message.text = "/start"
    message.content_type = "text"
    await start.cmd_start(message)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE
    assert "999_mode" not in user_sessions.sessions


@pytest.mark.asyncio
async def test_non_integer_id_denies_and_does_not_call_handler(monkeypatch):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))

    message = _make_text_message(AUTHORIZED_ID, "/start")
    message.from_user.id = "111111111"  # string, not int — no coercion allowed
    await start.cmd_start(message)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE
    assert f"{AUTHORIZED_ID}_mode" not in user_sessions.sessions


@pytest.mark.asyncio
async def test_boolean_id_denies_and_does_not_call_handler(monkeypatch):
    """
    bool is an int subclass in Python; True happening to equal 1 must never
    let it masquerade as a real numeric Telegram user id.
    """
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({1}))

    message = _make_text_message(True, "/start")
    await start.cmd_start(message)

    sent_text = shared_bot.send_message.await_args.args[1]
    assert sent_text == access_control.ACCESS_DENIED_MESSAGE
    # Rejection happens at identity-extraction time, before is_authorized()
    # is ever consulted with the bool.
    assert access_control._extract_user_id(message) is None


def test_extract_user_id_rejects_malformed_shapes():
    assert access_control._extract_user_id(SimpleNamespace(from_user=None)) is None
    assert access_control._extract_user_id(SimpleNamespace(from_user=SimpleNamespace())) is None
    assert access_control._extract_user_id(SimpleNamespace(from_user=SimpleNamespace(id="123"))) is None
    assert access_control._extract_user_id(SimpleNamespace(from_user=SimpleNamespace(id=True))) is None
    assert access_control._extract_user_id(SimpleNamespace()) is None
    assert access_control._extract_user_id(SimpleNamespace(from_user=SimpleNamespace(id=123))) == 123


@pytest.mark.asyncio
async def test_deny_does_not_raise_with_no_usable_response_target(monkeypatch):
    """
    A completely malformed update — not even a recognized Message/
    CallbackQuery shape — must fail closed silently rather than raising
    into the caller.
    """
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    await access_control._deny(SimpleNamespace(from_user=None))
    shared_bot.send_message.assert_not_called()
    shared_bot.answer_callback_query.assert_not_called()


# ---------------------------------------------------------------------------
# J. Denial-response delivery failures are token-safe (Stage 1C.1)
# ---------------------------------------------------------------------------
#
# If bot.send_message / bot.answer_callback_query themselves raise (e.g. a
# real pyTelegramBotAPI HTTP exception, which can embed the bot token in
# its request URL), that exception must never propagate into the caller and
# must never appear — token, URL, or raw text — in anything logged.

_FAKE_TOKEN = "123456789:AAFakeTokenForTestingOnlyDoNotUse"
_FAKE_TELEGRAM_EXCEPTION_TEXT = (
    f"A request to the Telegram API was unsuccessful. "
    f"Error code: 401. Description: Unauthorized "
    f"[https://api.telegram.org/bot{_FAKE_TOKEN}/sendMessage]"
)


@pytest.mark.asyncio
async def test_unauthorized_message_denial_send_failure_does_not_leak_token(monkeypatch, caplog):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    shared_bot.send_message.side_effect = RuntimeError(_FAKE_TELEGRAM_EXCEPTION_TEXT)
    route_mock = AsyncMock()
    monkeypatch.setattr(text, "route_text_request", route_mock)

    message = _make_text_message(UNAUTHORIZED_ID, "hello")
    with caplog.at_level("WARNING"):
        await text.handle_text_message(message)  # must not raise

    route_mock.assert_not_called()
    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions

    log_text = caplog.text
    assert _FAKE_TOKEN not in log_text
    assert "api.telegram.org" not in log_text
    assert _FAKE_TELEGRAM_EXCEPTION_TEXT not in log_text
    assert "Traceback" not in log_text


@pytest.mark.asyncio
async def test_unauthorized_callback_denial_answer_failure_does_not_leak_token(monkeypatch, caplog):
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())
    shared_bot.answer_callback_query.side_effect = RuntimeError(_FAKE_TELEGRAM_EXCEPTION_TEXT)

    callback = _make_callback(UNAUTHORIZED_ID, data="mode_voice")
    with caplog.at_level("WARNING"):
        await text.callback_mode(callback)  # must not raise

    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions

    log_text = caplog.text
    assert _FAKE_TOKEN not in log_text
    assert "api.telegram.org" not in log_text
    assert _FAKE_TELEGRAM_EXCEPTION_TEXT not in log_text
    assert "Traceback" not in log_text


# ---------------------------------------------------------------------------
# K. Registration-level regression test (Stage 1C.1)
# ---------------------------------------------------------------------------
#
# Guards the core decorator-order invariant: every handler actually
# registered with the bot dispatcher must be the require_authorized
# wrapper, never the original unprotected function. This would catch, e.g.,
# a future edit that puts @require_authorized above @bot.message_handler(...)
# instead of below, or drops it entirely — both of which silently disable
# the gate while looking correct at a glance.

def test_handler_registration_is_fully_authorization_wrapped():
    EXPECTED_MESSAGE_HANDLER_COUNT = 13
    EXPECTED_CALLBACK_HANDLER_COUNT = 1

    assert len(shared_bot.message_handlers) == EXPECTED_MESSAGE_HANDLER_COUNT
    assert len(shared_bot.callback_query_handlers) == EXPECTED_CALLBACK_HANDLER_COUNT

    async def _reference_original(update):
        return None

    wrapper_code = access_control.require_authorized(_reference_original).__code__

    registered_functions = [h["function"] for h in shared_bot.message_handlers] + [
        h["function"] for h in shared_bot.callback_query_handlers
    ]
    for func in registered_functions:
        assert func.__code__ is wrapper_code, (
            f"{getattr(func, '__qualname__', func)!r} is registered as the raw "
            "handler, not the require_authorized wrapper"
        )
        assert hasattr(func, "__wrapped__"), (
            f"{getattr(func, '__qualname__', func)!r} is missing __wrapped__ "
            "(functools.wraps not applied by require_authorized)"
        )


# ---------------------------------------------------------------------------
# L. Denial routed by concrete update type, not by attribute (Stage 1C.2)
# ---------------------------------------------------------------------------
#
# telebot's real types.Message carries its own top-level `.id` (an alias
# for message_id — see telebot/types.py, Message.__init__). That means
# "does this update have a top-level `.id`?" cannot distinguish a Message
# from a CallbackQuery. access_control._deny() must therefore route by
# isinstance(update, types.Message) / isinstance(update, types.CallbackQuery)
# and read only the field that belongs to the matched type — never guess
# from whichever attributes happen to be present.

@pytest.mark.asyncio
async def test_malformed_message_with_misleading_top_level_id_never_routed_as_callback(monkeypatch):
    """
    A real types.Message with no usable chat.id but a top-level `.id`
    (the message_id alias every real Message carries) must never have that
    `.id` handed to answer_callback_query — it must only ever be considered
    for send_message, and only via `chat.id`.
    """
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({AUTHORIZED_ID}))

    message = _new_message()
    message.id = 555  # message_id alias — must NEVER be read as a callback id
    message.from_user = SimpleNamespace(id=UNAUTHORIZED_ID, first_name="Test")
    message.chat = None  # no usable chat target
    message.text = "/start"
    message.content_type = "text"

    await start.cmd_start(message)  # must not raise

    shared_bot.answer_callback_query.assert_not_called()
    shared_bot.send_message.assert_not_called()
    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions


@pytest.mark.asyncio
async def test_malformed_callback_with_misleading_chat_never_routed_as_message(monkeypatch):
    """
    A real types.CallbackQuery carrying an unexpected top-level `.chat`
    attribute must still be denied purely as a callback: only
    answer_callback_query (using `.id`) may be called, `.chat` must never
    be read as a message target, and send_message must never be called.
    """
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())

    callback = _new_callback()
    callback.id = "cb-1"
    callback.data = "mode_voice"
    callback.from_user = SimpleNamespace(id=UNAUTHORIZED_ID, first_name="Test")
    callback.chat = SimpleNamespace(id=999)  # unexpected — must be ignored
    callback.message = SimpleNamespace(chat=SimpleNamespace(id=999))

    await text.callback_mode(callback)  # must not raise

    shared_bot.answer_callback_query.assert_awaited_once()
    assert shared_bot.answer_callback_query.await_args.args[0] == "cb-1"
    assert shared_bot.answer_callback_query.await_args.args[1] == access_control.ACCESS_DENIED_MESSAGE
    shared_bot.send_message.assert_not_called()
    assert f"{UNAUTHORIZED_ID}_mode" not in user_sessions.sessions


@pytest.mark.asyncio
async def test_deny_ignores_unknown_object_type_even_with_misleading_id_and_chat(monkeypatch):
    """
    An update that is neither a types.Message nor a types.CallbackQuery
    must never be attribute-sniffed into either denial path, even when it
    happens to carry both a top-level `.id` and a `.chat`-shaped attribute
    that could otherwise look plausible for either type.
    """
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())

    fake_update = SimpleNamespace(
        from_user=SimpleNamespace(id=UNAUTHORIZED_ID, first_name="Test"),
        id=42,
        chat=SimpleNamespace(id=999),
    )
    await access_control._deny(fake_update)  # must not raise

    shared_bot.send_message.assert_not_called()
    shared_bot.answer_callback_query.assert_not_called()
