"""
Stage 6C regression tests: telegram_link_config.py's bot-username
normalization/validation and TTL bounds (Section H), and
app/telegram_link.py's raw-secret canonical-shape validation and the
64-character Telegram /start payload limit (Section F/O). All offline —
no database, no network.
"""

import base64
import importlib
from pathlib import Path

import pytest

import telegram_link_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
from app.telegram_link import (
    LINK_PAYLOAD_PREFIX,
    MAX_START_PAYLOAD_LENGTH,
    _is_canonical_secret,
    extract_link_secret,
)


# ---------------------------------------------------------------------------
# A. Bot username normalization/validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("my_tutor_bot", "my_tutor_bot"),
        ("@my_tutor_bot", "my_tutor_bot"),
        ("  @my_tutor_bot  ", "my_tutor_bot"),
        ("MyTutorBot", "MyTutorBot"),
        ("ab12Bot", "ab12Bot"),  # 7 chars, minimum-ish, ends in Bot
    ],
)
def test_valid_usernames_normalize_correctly(raw, expected):
    assert telegram_link_config._normalize_bot_username(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "bot",  # too short (3 chars < 5)
        "ab" + "x" * 30 + "bot",  # too long (> 32 chars)
        "1_tutor_bot",  # doesn't start with a letter
        "tutor-bot-bot",  # hyphen not allowed
        "tutor bot",  # space not allowed
        "tutor_bott_",  # doesn't end in "bot"
        "tutorbo",  # doesn't end in "bot"
        "tütor_bot",  # non-ASCII letter
    ],
)
def test_malformed_usernames_are_rejected(raw):
    assert telegram_link_config._normalize_bot_username(raw) is None


def test_username_ending_check_is_case_insensitive():
    for suffix in ("bot", "Bot", "BOT", "bOt"):
        assert telegram_link_config._normalize_bot_username(f"tutor_{suffix}") is not None


# ---------------------------------------------------------------------------
# A.1 Exact 5-32 character boundary (Stage 6C corrective pass, MINOR 1): the
# previous `{3,30}` middle-quantifier silently raised the real minimum to 7
# characters, rejecting genuinely valid 5-6 character usernames the
# `5 <= len(candidate) <= 32` length check would otherwise accept. `{1,28}`
# restores the documented 5..32 total range exactly.
# ---------------------------------------------------------------------------


def _username_of_total_length(total: int) -> str:
    """Builds a syntactically-valid candidate of an EXACT total length:
    one leading letter + a middle of underscores + the fixed 3-char "bot"
    suffix. `total` must be >= 4 (1 letter + 0 middle + 3 suffix)."""
    middle_length = total - 1 - 3
    assert middle_length >= 0, "total too short to build a valid-shape candidate at all"
    return "a" + ("_" * middle_length) + "bot"


@pytest.mark.parametrize("total_length", [5, 6, 32])
def test_valid_boundary_lengths_are_accepted(total_length):
    candidate = _username_of_total_length(total_length)
    assert len(candidate) == total_length
    assert telegram_link_config._normalize_bot_username(candidate) == candidate


@pytest.mark.parametrize("total_length", [4, 33])
def test_invalid_boundary_lengths_are_rejected(total_length):
    candidate = _username_of_total_length(total_length)
    assert len(candidate) == total_length
    assert telegram_link_config._normalize_bot_username(candidate) is None


def test_missing_or_malformed_username_does_not_prevent_module_import(monkeypatch):
    """Section H: TELEGRAM_BOT_USERNAME must never fail closed at import
    time, unlike every other credential check in this codebase."""
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "not valid!!")
    reloaded = importlib.reload(telegram_link_config)
    try:
        assert reloaded.TELEGRAM_BOT_USERNAME is None
    finally:
        monkeypatch.delenv("TELEGRAM_BOT_USERNAME", raising=False)
        importlib.reload(telegram_link_config)


def test_telegram_deep_link_raises_when_username_unavailable(monkeypatch):
    monkeypatch.setattr(telegram_link_config, "TELEGRAM_BOT_USERNAME", None)
    with pytest.raises(RuntimeError):
        telegram_link_config.telegram_deep_link("secret")
    with pytest.raises(RuntimeError):
        telegram_link_config.telegram_bot_path()


def test_telegram_deep_link_shape(monkeypatch):
    monkeypatch.setattr(telegram_link_config, "TELEGRAM_BOT_USERNAME", "my_tutor_bot")
    link = telegram_link_config.telegram_deep_link("abc123")
    assert telegram_link_config.telegram_bot_path() == "/my_tutor_bot"
    assert link == "https://t.me/my_tutor_bot?start=link_abc123"


def test_module_never_imports_telegram_bot_token():
    """Section H: the web adapter must never import/require
    TELEGRAM_BOT_TOKEN. Checked structurally — a fresh subprocess imports
    ONLY telegram_link_config and asserts telegram_config.py never lands in
    sys.modules as a side effect — never a docstring/source-text substring
    check, since this module's own docstring legitimately discusses
    telegram_config.py in prose."""
    import subprocess
    import sys

    script = (
        "import sys, telegram_link_config; "
        "assert 'telegram_config' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, cwd=str(_PROJECT_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert not hasattr(telegram_link_config, "TELEGRAM_BOT_TOKEN")


def test_link_attempt_ttl_within_bounds():
    assert 0 < telegram_link_config.LINK_ATTEMPT_TTL_SECONDS <= 900


def test_link_attempt_ttl_rejects_excessive_override(monkeypatch):
    monkeypatch.setenv("TELEGRAM_LINK_ATTEMPT_TTL_SECONDS", "999999")
    with pytest.raises(ValueError):
        importlib.reload(telegram_link_config)
    monkeypatch.delenv("TELEGRAM_LINK_ATTEMPT_TTL_SECONDS", raising=False)
    importlib.reload(telegram_link_config)


# ---------------------------------------------------------------------------
# B. Secret canonical shape + 64-char Telegram payload limit
# ---------------------------------------------------------------------------


def test_a_real_generated_secret_round_trips_through_extract_link_secret():
    import secrets as _secrets

    raw = _secrets.token_urlsafe(32)
    payload = f"{LINK_PAYLOAD_PREFIX}{raw}"
    assert extract_link_secret(payload) == raw


def test_link_payload_never_exceeds_telegrams_64_character_start_limit():
    import secrets as _secrets

    raw = _secrets.token_urlsafe(32)
    payload = f"{LINK_PAYLOAD_PREFIX}{raw}"
    assert len(payload) <= MAX_START_PAYLOAD_LENGTH
    # Exact, not just "under the bound" — pins the real, current size so a
    # future accidental change to either constant is caught immediately.
    assert len(payload) == 48


def test_extract_link_secret_rejects_wrong_length():
    assert extract_link_secret(LINK_PAYLOAD_PREFIX + "tooshort") is None
    assert extract_link_secret(LINK_PAYLOAD_PREFIX + "x" * 100) is None


def test_extract_link_secret_rejects_invalid_alphabet():
    bogus = "!" * 43
    assert extract_link_secret(LINK_PAYLOAD_PREFIX + bogus) is None


def test_extract_link_secret_rejects_non_canonical_encoding_of_the_same_bytes():
    """Mirrors app/auth_session.py's own corrective-pass regression
    (tests/test_stage6a_corrective2_secret_and_token_validation.py's
    test_impossible_final_quantum_is_rejected): a 43-char, alphabet-valid
    string that decodes to the SAME 32 bytes as a real secret, but is not
    itself the CANONICAL encoding of those bytes (nonzero unused low bits
    in the final base64 character), must be rejected — not silently
    accepted as if it were the genuine secret it happens to decode to."""
    import secrets as _secrets

    real_secret = _secrets.token_urlsafe(32)
    padding = "=" * (-len(real_secret) % 4)
    decoded = base64.urlsafe_b64decode(real_secret + padding)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

    found = None
    for cand in alphabet:
        if cand == real_secret[-1]:
            continue
        candidate = real_secret[:-1] + cand
        try:
            if base64.urlsafe_b64decode(candidate + padding) == decoded:
                found = candidate
                break
        except Exception:
            continue

    assert found is not None, "test setup failed to construct a non-canonical same-bytes candidate"
    assert not _is_canonical_secret(found)
    assert extract_link_secret(LINK_PAYLOAD_PREFIX + found) is None
    # The canonical original must still be accepted.
    assert _is_canonical_secret(real_secret)
    assert extract_link_secret(LINK_PAYLOAD_PREFIX + real_secret) == real_secret


def test_extract_link_secret_returns_none_without_the_link_prefix():
    assert extract_link_secret("not_a_link_payload") is None
    assert extract_link_secret("") is None


def test_is_canonical_secret_rejects_empty_and_none_like_values():
    assert _is_canonical_secret("") is False
    assert _is_canonical_secret("x") is False
