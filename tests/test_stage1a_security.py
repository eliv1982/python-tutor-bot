"""
Stage 1A / 1A.1 regression tests:
- ProxyAPI removal and OPENAI_BASE_URL removal (official OpenAI endpoint only)
- vision credential-disclosure fix (no Telegram token/URL reaches OpenAI)
- provider-boundary enforcement (only base64 data URLs accepted)
- safe Telegram-download exception logging (no token leakage into logs)
- pending-image size containment and /reset cleanup

All external calls (Telegram, OpenAI) are mocked. No network access is
performed by this test module.
"""

import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import config
from app.session import user_sessions

TEST_TOKEN = config.TELEGRAM_BOT_TOKEN  # dummy token from tests/conftest.py
FAKE_TELEGRAM_TOKEN = "987654321:FAKE-TOKEN-FOR-LOG-LEAK-TEST"


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Reset the in-memory session store between tests."""
    user_sessions.sessions.clear()
    yield
    user_sessions.sessions.clear()


def _make_photo_message(user_id: int, caption: str = "", file_id: str = "fake-file-id"):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        caption=caption,
        photo=[SimpleNamespace(file_id=file_id)],
    )


def _make_text_message(user_id: int, text: str):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id),
        text=text,
        content_type="text",
    )


def _fake_sdk_response(content: str):
    """Mimic the shape of an OpenAI SDK chat.completions.create() response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


# ---------------------------------------------------------------------------
# ProxyAPI / OPENAI_BASE_URL removal
# ---------------------------------------------------------------------------

def test_no_proxyapi_references_in_active_config_sources():
    """Repository config/runtime files must not mention ProxyAPI at all."""
    project_root = config.BASE_DIR
    files_to_check = [
        project_root / "config.py",
        project_root / ".env.example",
        project_root / "README.md",
        project_root / "services" / "openai_client.py",
        project_root / "services" / "image_generation.py",
    ]
    for path in files_to_check:
        content = path.read_text(encoding="utf-8").lower()
        assert "proxyapi" not in content, f"ProxyAPI reference found in {path}"


def test_no_openai_base_url_references_in_config_and_docs():
    """The literal OPENAI_BASE_URL token (as a name, not as a substring of
    something else) must not exist in config/env/docs at all.

    OpenAI now always means the official OpenAI endpoint; a generic
    base-URL escape hatch would let ProxyAPI (or anything else) back in.

    Uses a word-boundary regex rather than a plain substring check because
    config.py legitimately defines OFFICIAL_OPENAI_BASE_URL, which
    contains "OPENAI_BASE_URL" as a substring without being it.
    """
    standalone_token = re.compile(r"(?<![A-Za-z0-9_])OPENAI_BASE_URL(?![A-Za-z0-9_])")
    project_root = config.BASE_DIR
    files_to_check = [
        project_root / "config.py",
        project_root / ".env.example",
        project_root / "README.md",
        project_root / "services" / "image_generation.py",
    ]
    for path in files_to_check:
        content = path.read_text(encoding="utf-8")
        assert not standalone_token.search(content), f"OPENAI_BASE_URL reference found in {path}"


def test_no_openai_api_base_references_in_config_and_docs():
    """OPENAI_API_BASE (the langchain-openai equivalent escape hatch) must
    likewise not exist as active config anywhere outside rag/index.py's
    own explanatory comment (which documents the defense, not a read)."""
    standalone_token = re.compile(r"(?<![A-Za-z0-9_])OPENAI_API_BASE(?![A-Za-z0-9_])")
    project_root = config.BASE_DIR
    files_to_check = [
        project_root / "config.py",
        project_root / ".env.example",
        project_root / "README.md",
        project_root / "services" / "openai_client.py",
        project_root / "services" / "image_generation.py",
    ]
    for path in files_to_check:
        content = path.read_text(encoding="utf-8")
        assert not standalone_token.search(content), f"OPENAI_API_BASE reference found in {path}"


def test_openai_client_does_not_read_openai_base_url_env_var():
    """services/openai_client.py may only *mention* OPENAI_BASE_URL in a
    comment explaining the defense below; it must never read it as config
    and must always pass an explicit, hardcoded base_url to AsyncOpenAI.

    This matters because the official openai SDK itself falls back to an
    OPENAI_BASE_URL *environment variable* whenever base_url is left
    unset (verified against the installed SDK) — so merely deleting our
    own config.OPENAI_BASE_URL is not sufficient; a stray OPENAI_BASE_URL
    left in a developer's .env would otherwise silently re-route every
    request through it again.
    """
    project_root = config.BASE_DIR
    content = (project_root / "services" / "openai_client.py").read_text(encoding="utf-8")

    assert 'os.getenv("OPENAI_BASE_URL")' not in content
    assert "os.environ.get('OPENAI_BASE_URL')" not in content
    assert '"OPENAI_BASE_URL",' not in content  # not imported from config
    assert "base_url=OFFICIAL_OPENAI_BASE_URL" in content


def test_openai_client_pins_base_url_against_sdk_env_fallback():
    """Regression guard for the exact escape hatch above: even with a
    ProxyAPI-pointing OPENAI_BASE_URL sitting in the environment, the
    constructed client must still target the official OpenAI endpoint."""
    import os
    from openai import AsyncOpenAI

    previous = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = "https://api.proxyapi.ru/openai/v1"
    try:
        from services.openai_client import OFFICIAL_OPENAI_BASE_URL
        client = AsyncOpenAI(api_key="sk-test", base_url=OFFICIAL_OPENAI_BASE_URL)
        assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"
    finally:
        if previous is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = previous


def test_config_has_no_proxyapi_or_base_url_attrs():
    assert not hasattr(config, "USE_PROXYAPI")
    assert not hasattr(config, "PROXYAPI_BASE_URL_DEFAULT")
    assert not hasattr(config, "OPENAI_BASE_URL")


def test_openai_client_uses_official_endpoint_only():
    from services.openai_client import openai_client

    assert not hasattr(openai_client, "use_proxyapi")
    assert str(openai_client.client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_image_generation_uses_official_openai_endpoint():
    from services import image_generation

    assert image_generation.OPENAI_API_BASE_URL == "https://api.openai.com/v1"
    assert "proxyapi" not in image_generation.OPENAI_API_BASE_URL.lower()


@pytest.mark.asyncio
async def test_image_generation_uses_configured_model_and_official_url(monkeypatch, tmp_path):
    """Goal 5 (Stage 1A): the configured DALLE_MODEL must be wired into the
    request. Goal (Stage 1A.1 #1): the URL is the fixed official endpoint,
    never constructed from an environment base URL."""
    from services import image_generation

    # Redirect output away from the real application data directory.
    monkeypatch.setattr(image_generation, "GENERATED_IMAGES_DIR", tmp_path)

    captured = {}

    class FakeResponse:
        status = 200

        async def json(self):
            return {"data": [{"b64_json": "AAAA", "revised_prompt": "a cat"}]}

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class FakeSession:
        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(image_generation.aiohttp, "ClientSession", lambda: FakeSession())

    result = await image_generation.generate_image(prompt="a cat in space")

    assert captured["payload"]["model"] == config.DALLE_MODEL
    assert captured["url"] == "https://api.openai.com/v1/images/generations"
    assert result["image_path"].parent == tmp_path
    assert result["image_path"].exists()


# ---------------------------------------------------------------------------
# Provider boundary: only base64 data URLs are accepted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_boundary_rejects_external_http_url(monkeypatch):
    from services.openai_client import openai_client

    create_mock = AsyncMock()
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    with pytest.raises(ValueError):
        await openai_client.analyze_image(image_url="http://example.com/cat.jpg", prompt="describe")

    create_mock.assert_not_called()


@pytest.mark.asyncio
async def test_provider_boundary_rejects_telegram_token_url(monkeypatch):
    from services.openai_client import openai_client

    create_mock = AsyncMock()
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    token_url = f"https://api.telegram.org/file/bot{TEST_TOKEN}/photos/file_1.jpg"
    with pytest.raises(ValueError):
        await openai_client.analyze_image(image_url=token_url, prompt="describe")

    create_mock.assert_not_called()


@pytest.mark.asyncio
async def test_provider_boundary_accepts_data_url(monkeypatch):
    from services.openai_client import openai_client

    create_mock = AsyncMock(return_value=_fake_sdk_response("a cat"))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    result = await openai_client.analyze_image(
        image_url="data:image/png;base64,QUJD", prompt="describe"
    )

    assert result == "a cat"
    create_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# End-to-end: actual outbound OpenAI SDK payload, not a mocked analyze_image
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_caption_flow_outbound_sdk_payload_has_no_secret(monkeypatch):
    """Full handler -> router -> vision -> openai_client -> SDK path, with
    only client.chat.completions.create mocked (the real provider boundary)."""
    import handlers.image as image_handler
    from services.openai_client import openai_client

    user_id = 111

    monkeypatch.setattr(
        image_handler.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="photos/file_1.jpg")),
    )
    fake_bytes = b"\xff\xd8\xff\xe0fakejpegbytes"
    monkeypatch.setattr(image_handler.bot, "download_file", AsyncMock(return_value=fake_bytes))
    monkeypatch.setattr(image_handler.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(image_handler.bot, "send_message", AsyncMock())

    create_mock = AsyncMock(return_value=_fake_sdk_response("It's a cat."))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    message = _make_photo_message(user_id, caption="What is this?")
    await image_handler.handle_photo_message(message)

    create_mock.assert_awaited_once()
    outbound_messages = create_mock.await_args.kwargs["messages"]
    image_url = outbound_messages[0]["content"][1]["image_url"]["url"]

    assert image_url.startswith("data:image/"), image_url
    assert TEST_TOKEN not in image_url
    assert "api.telegram.org/file/bot" not in image_url
    # Sanity: the actual downloaded bytes really did make it into the payload.
    import base64
    assert base64.b64encode(fake_bytes).decode("utf-8") in image_url


@pytest.mark.asyncio
async def test_pending_image_flow_outbound_sdk_payload_has_no_secret(monkeypatch):
    """Pending-image + later-text path, asserted against the real SDK call."""
    import handlers.image as image_handler
    import handlers.text as text_handler
    from services.openai_client import openai_client

    user_id = 222

    monkeypatch.setattr(
        image_handler.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="photos/file_2.png")),
    )
    fake_bytes = b"\x89PNGfakepngbytes"
    monkeypatch.setattr(image_handler.bot, "download_file", AsyncMock(return_value=fake_bytes))
    monkeypatch.setattr(image_handler.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(image_handler.bot, "send_message", AsyncMock())
    monkeypatch.setattr(text_handler.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(text_handler.bot, "send_message", AsyncMock())

    # Step 1: photo with no caption -> stored as pending.
    photo_message = _make_photo_message(user_id, caption="")
    await image_handler.handle_photo_message(photo_message)

    pending = user_sessions.get_pending_image(user_id)
    assert pending is not None
    assert pending.startswith("data:image/")
    assert TEST_TOKEN not in pending
    assert "api.telegram.org/file/bot" not in pending

    # Step 2: a later text message answers the "what do you want to know" prompt.
    create_mock = AsyncMock(return_value=_fake_sdk_response("This is a screenshot of Python code."))
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create_mock)

    text_message = _make_text_message(user_id, "Find the bug in this code")
    await text_handler.handle_text_message(text_message)

    create_mock.assert_awaited_once()
    outbound_messages = create_mock.await_args.kwargs["messages"]
    image_url = outbound_messages[0]["content"][1]["image_url"]["url"]
    prompt_text = outbound_messages[0]["content"][0]["text"]

    assert image_url == pending
    assert TEST_TOKEN not in image_url
    assert "api.telegram.org/file/bot" not in image_url
    assert "Find the bug in this code" in prompt_text

    # Pending image must be cleared after being consumed.
    assert user_sessions.get_pending_image(user_id) is None


# ---------------------------------------------------------------------------
# Safe Telegram-download exception logging
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_photo_download_exception_does_not_leak_token_in_logs(monkeypatch, caplog):
    import handlers.image as image_handler

    leaking_message = (
        f"Failed to fetch https://api.telegram.org/file/bot{FAKE_TELEGRAM_TOKEN}"
        "/photos/file_1.jpg: 404 Not Found"
    )

    async def raise_leaking_error(file_id):
        raise Exception(leaking_message)

    monkeypatch.setattr(image_handler.bot, "get_file", raise_leaking_error)
    monkeypatch.setattr(image_handler.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(image_handler.bot, "send_message", AsyncMock())

    message = _make_photo_message(user_id=333, caption="what is this")

    with caplog.at_level(logging.DEBUG):
        await image_handler.handle_photo_message(message)

    log_text = caplog.text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert leaking_message not in log_text
    assert "api.telegram.org" not in log_text
    # The safe, structured replacement must still be observable.
    assert "photo_download" in log_text


@pytest.mark.asyncio
async def test_voice_download_exception_does_not_leak_token_in_logs(monkeypatch, caplog):
    import handlers.voice as voice_handler

    leaking_message = (
        f"Failed to fetch https://api.telegram.org/file/bot{FAKE_TELEGRAM_TOKEN}"
        "/voice/file_1.oga: 404 Not Found"
    )

    async def raise_leaking_error(file_id):
        raise Exception(leaking_message)

    monkeypatch.setattr(voice_handler.bot, "get_file", raise_leaking_error)
    monkeypatch.setattr(voice_handler.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(voice_handler.bot, "send_message", AsyncMock())

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=444),
        chat=SimpleNamespace(id=444),
        voice=SimpleNamespace(file_id="fake-voice-id"),
    )

    with caplog.at_level(logging.DEBUG):
        await voice_handler.handle_voice_message(message)

    log_text = caplog.text
    assert FAKE_TELEGRAM_TOKEN not in log_text
    assert leaking_message not in log_text
    assert "api.telegram.org" not in log_text
    assert "voice_download" in log_text


# ---------------------------------------------------------------------------
# Pending-image size containment and /reset cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oversized_photo_is_rejected_before_storage(monkeypatch):
    import handlers.image as image_handler

    user_id = 555
    oversized_bytes = b"x" * (config.MAX_TELEGRAM_IMAGE_BYTES + 1)

    monkeypatch.setattr(
        image_handler.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="photos/big.jpg")),
    )
    monkeypatch.setattr(image_handler.bot, "download_file", AsyncMock(return_value=oversized_bytes))
    monkeypatch.setattr(image_handler.bot, "send_chat_action", AsyncMock())
    send_message_mock = AsyncMock()
    monkeypatch.setattr(image_handler.bot, "send_message", send_message_mock)

    message = _make_photo_message(user_id, caption="")
    await image_handler.handle_photo_message(message)

    # Nothing must have been stored for a follow-up question.
    assert user_sessions.get_pending_image(user_id) is None
    send_message_mock.assert_awaited_once()
    assert "слишком" in send_message_mock.await_args.args[1].lower()


@pytest.mark.asyncio
async def test_reset_clears_pending_image(monkeypatch):
    import handlers.start as start_handler

    user_id = 666
    user_sessions.set_pending_image(user_id, "data:image/jpeg;base64,QUJD")
    assert user_sessions.get_pending_image(user_id) is not None

    monkeypatch.setattr(start_handler.bot, "send_message", AsyncMock())

    message = SimpleNamespace(from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=user_id))
    await start_handler.cmd_reset(message)

    assert user_sessions.get_pending_image(user_id) is None
