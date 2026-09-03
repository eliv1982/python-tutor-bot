"""
Stage 6C regression tests: the raw Telegram link bearer secret must never
appear in logs, exception text, reprs, or database fields (Section F) —
end to end through the real application layer, HTTP endpoint, and Telegram
dispatch. Real disposable PostgreSQL via tests/conftest.py's postgres_db.

Every "must not leak" check below goes through
tests/secrecy_helpers.assert_no_secret_leak() rather than a bare
`assert raw_secret not in some_text` (Stage 6C corrective pass,
independent-audit MINOR 2): a bare assert's own failure-report
introspection would print BOTH operands — the secret itself included — on
a failure, which is exactly the outcome these tests exist to prove never
happens. See that helper's own docstring, and Section A at the bottom of
this file for a controlled self-test proving the helper itself fails with
a fixed, redacted message and never emits the sentinel secret it was
given.
"""

import logging
import os
import random
import secrets
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient
from telebot import types

import app.auth_session as auth_session
import app.telegram_link as telegram_link
import db.auth_sessions as db_auth_sessions
import db.github_identity as db_github_identity
import db.identity as db_identity
import db.telegram_link as db_telegram_link
import handlers.start  # noqa: F401 -- import-time side effect: registers cmd_start on the shared bot
import telegram_link_config
import web_config
from bot import bot as shared_bot
from secrecy_helpers import assert_no_secret_leak
from web.app import create_app
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


@pytest.fixture(autouse=True)
def _insecure_posture_for_testing(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    monkeypatch.setattr(telegram_link_config, "TELEGRAM_BOT_USERNAME", "my_tutor_bot")
    yield


def _github_only_user() -> uuid.UUID:
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    return db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)


# ---------------------------------------------------------------------------
# A. The secrecy-check helper itself — controlled self-test (Stage 6C
# corrective pass, independent-audit MINOR 2, required negative proof)
# ---------------------------------------------------------------------------


def test_assert_no_secret_leak_self_test_fails_with_fixed_redacted_message():
    """Deliberately feeds assert_no_secret_leak() a sentinel it WILL find,
    via pytest.raises(pytest.fail.Exception) (never a bare
    `with pytest.raises(...)` around a plain `assert`) — proving the
    helper's own failure is a FIXED, generic message that never echoes the
    sentinel, regardless of how the sentinel appeared in the checked
    text."""
    sentinel = "definitely-a-sentinel-secret-9f8e7d6c5b4a1230"

    with pytest.raises(pytest.fail.Exception) as excinfo:
        assert_no_secret_leak(sentinel, f"some captured text containing {sentinel}")

    message = str(excinfo.value)
    assert message == "secrecy check failed: a tracked secret leaked into a captured surface"
    assert sentinel not in message


def test_assert_no_secret_leak_clears_caplog_and_containers_before_failing(caplog):
    sentinel = "another-sentinel-3a2b1c0d9e8f"
    caplog.set_level(logging.DEBUG)
    logging.getLogger(__name__).info("leaking %s here", sentinel)
    mutable_container = [f"also leaked: {sentinel}"]

    with pytest.raises(pytest.fail.Exception):
        assert_no_secret_leak(sentinel, caplog=caplog, clear_containers=[mutable_container])

    assert sentinel not in caplog.text
    assert mutable_container == []


def test_assert_no_secret_leak_is_silent_when_nothing_leaked():
    result = assert_no_secret_leak("a-secret-that-does-not-appear", "completely unrelated text")
    assert result is None


def test_assert_no_secret_leak_survives_real_pytest_reporting(tmp_path):
    """The three self-tests above only inspect the CAUGHT
    pytest.fail.Exception's own message string, in-process — none of them
    prove that pytest's OWN reporting layer (the thing that actually
    renders a failure to a real terminal/CI log, including its assertion-
    rewriting introspection for an ordinary bare `assert`) leaves the
    sentinel out too. This closes that gap (Stage 6C corrective pass,
    independent-audit MINOR 2, required strengthened self-test): it writes
    a deliberately-failing test into `tmp_path` — pytest's own disposable
    per-test directory, NEVER the repository itself, and cleaned up
    automatically — that calls assert_no_secret_leak() with a sentinel it
    WILL find, runs it through a genuinely SEPARATE, nested `python -m
    pytest` process, and inspects the COMPLETE combined stdout+stderr of
    that nested run (never only a caught exception), confirming: the
    nested run failed (non-zero exit); the fixed, redacted failure text is
    present; and the sentinel is ABSENT from the entire rendered output.
    The sentinel is a synthetic, randomly-generated value that exists only
    for this one test run — never a real link secret — and appears only
    inside the generated file's own source, never in this test's own name,
    parameters, or assertions. The outer "sentinel absent" check itself
    goes through assert_no_secret_leak() (never a bare `assert sentinel
    not in combined_output`), for the identical reason as everywhere else
    in this file; every other check here computes a boolean first and
    fails with a fixed message that never dumps `combined_output` itself
    (which, if the helper genuinely were broken, is exactly where the
    sentinel could have leaked)."""
    sentinel = f"nested-pytest-sentinel-{secrets.token_hex(16)}"
    nested_test_file = tmp_path / "test_nested_secrecy_selftest.py"
    nested_test_file.write_text(
        "from secrecy_helpers import assert_no_secret_leak\n"
        "\n"
        "\n"
        "def test_deliberately_leaks():\n"
        f"    sentinel = {sentinel!r}\n"
        "    assert_no_secret_leak(sentinel, f'leaked text containing {sentinel}')\n",
        encoding="utf-8",
    )

    tests_dir = str(Path(__file__).resolve().parent)
    project_root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (tests_dir, project_root, env.get("PYTHONPATH", "")) if p)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(nested_test_file), "-v", "-p", "no:cacheprovider"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined_output = proc.stdout + proc.stderr

    nested_failed = proc.returncode != 0
    fixed_message_present = (
        "secrecy check failed: a tracked secret leaked into a captured surface" in combined_output
    )
    if not nested_failed or not fixed_message_present:
        pytest.fail(
            f"nested pytest self-test did not fail as expected "
            f"(returncode={proc.returncode}, fixed_message_present={fixed_message_present})",
            pytrace=False,
        )

    assert_no_secret_leak(sentinel, combined_output)


# ---------------------------------------------------------------------------
# B. Dataclass repr-safety
# ---------------------------------------------------------------------------


def test_link_start_result_repr_never_includes_the_deep_link():
    from datetime import datetime, timezone

    result = telegram_link.LinkStartResult(
        deep_link="https://t.me/my_tutor_bot?start=link_super-secret-value", expires_at=datetime.now(timezone.utc)
    )
    assert_no_secret_leak("super-secret-value", repr(result))
    assert_no_secret_leak("https://t.me", repr(result))


# ---------------------------------------------------------------------------
# C. app.telegram_link.start_link() end to end — logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_link_never_logs_the_raw_secret(caplog, postgres_db):
    user_id = _github_only_user()
    with caplog.at_level(logging.DEBUG):
        result = await telegram_link.start_link(user_id)

    assert result is not None
    raw_secret = result.deep_link.rsplit("link_", 1)[1]
    # Two independent needles against the same haystack (caplog.text) —
    # never one call with both the secret AND a string that legitimately
    # CONTAINS it, which would make the check trivially "fail" against
    # itself rather than against caplog.
    assert_no_secret_leak(raw_secret, caplog=caplog)
    assert_no_secret_leak(result.deep_link, caplog=caplog)


@pytest.mark.asyncio
async def test_redeem_link_never_logs_the_raw_secret(caplog, postgres_db):
    user_id = _github_only_user()
    result = await telegram_link.start_link(user_id)
    raw_secret = result.deep_link.rsplit("link_", 1)[1]

    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    with caplog.at_level(logging.DEBUG):
        outcome = await telegram_link.redeem_link(telegram_user_id=telegram_id, raw_secret=raw_secret)

    assert outcome == telegram_link.RedemptionOutcome.MERGED
    assert_no_secret_leak(raw_secret, caplog=caplog)


# ---------------------------------------------------------------------------
# D. HTTP endpoint — response body/headers and logs
# ---------------------------------------------------------------------------


async def _session_for(user_id):
    return await auth_session.create_session(user_id, issued_secure=False)


@pytest.mark.asyncio
async def test_start_endpoint_logs_never_include_the_raw_secret(caplog):
    user_id = _github_only_user()
    issued = await _session_for(user_id)
    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)

    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/api/link/telegram/start", headers={CSRF_HEADER_NAME: derive_csrf_token(issued.raw_token)}
        )

    assert response.status_code == 200
    raw_secret = response.json()["deep_link"].rsplit("link_", 1)[1]
    assert_no_secret_leak(raw_secret, caplog=caplog)


# ---------------------------------------------------------------------------
# E. Real Telegram dispatch — sent message text and logs
# ---------------------------------------------------------------------------


def _text_message(user_id: int, text_: str) -> types.Message:
    message = types.Message.__new__(types.Message)
    message.from_user = SimpleNamespace(id=user_id, first_name="Test")
    message.chat = SimpleNamespace(id=user_id)
    message.text = text_
    message.content_type = "text"
    return message


@pytest.mark.asyncio
async def test_dispatch_never_logs_or_echoes_the_raw_secret(caplog, monkeypatch):
    user_id = _github_only_user()
    result = await telegram_link.start_link(user_id)
    raw_secret = result.deep_link.rsplit("link_", 1)[1]

    send_message_mock = AsyncMock()
    monkeypatch.setattr(shared_bot, "send_message", send_message_mock)

    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    message = _text_message(telegram_id, f"/start link_{raw_secret}")

    with caplog.at_level(logging.DEBUG):
        await shared_bot.process_new_messages([message])

    send_message_mock.assert_awaited_once()
    sent_text = send_message_mock.await_args.args[1]
    assert_no_secret_leak(raw_secret, sent_text, caplog=caplog, clear_containers=[send_message_mock.call_args_list])


@pytest.mark.asyncio
async def test_dispatch_of_a_malformed_link_payload_never_logs_it(caplog, monkeypatch):
    send_message_mock = AsyncMock()
    monkeypatch.setattr(shared_bot, "send_message", send_message_mock)

    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    distinctive = "distinctive-malformed-payload-9d8e7f"
    message = _text_message(telegram_id, f"/start link_{distinctive}")

    with caplog.at_level(logging.DEBUG):
        await shared_bot.process_new_messages([message])

    assert_no_secret_leak(distinctive, caplog=caplog)


# ---------------------------------------------------------------------------
# F. Exception text never carries the raw secret
# ---------------------------------------------------------------------------


def test_transient_failure_exception_text_never_carries_the_raw_secret(postgres_db):
    user_id = _github_only_user()
    raw_secret = secrets.token_urlsafe(32)
    import hashlib
    from datetime import datetime, timedelta, timezone

    db_telegram_link.create_attempt_sync(
        web_user_id=user_id,
        link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    def _boom():
        # Raised via redeem_attempt_sync()'s own _test_hook_after_claim
        # seam (a stable, named checkpoint right after the atomic claim —
        # never brittle raw Session.execute() call-counting) — the
        # RuntimeError's own text deliberately embeds telegram_id (never
        # the raw secret) to prove this assertion is a genuine, specific
        # check and not vacuously true for an empty exception message.
        raise RuntimeError(f"simulated failure while processing digest for user {telegram_id}")

    with pytest.raises(RuntimeError) as excinfo:
        db_telegram_link.redeem_attempt_sync(
            link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(),
            telegram_user_id=telegram_id,
            _test_hook_after_claim=_boom,
        )

    assert_no_secret_leak(raw_secret, str(excinfo.value))
