"""
Stage 6A independent-audit corrective pass #2 — Major 2 (SESSION_SECRET_KEY
whitespace-padding validation) and the MINOR canonical-token-shape
hardening.

## Major 2

Pre-fix rule (corrective pass #1): strip LEADING/TRAILING whitespace, then
require the STRIPPED value to encode to >= 32 UTF-8 bytes. The auditor's
exact reproduction: `"a" + " " * 40 + "b"` strips to 42 bytes (>= 32,
accepted) despite carrying only 2 real characters — internal whitespace
still counted as "material".

Fixed rule (see web_config.py's own comment block for the authoritative
statement): SESSION_SECRET_KEY must contain NO Unicode whitespace
character anywhere (leading, trailing, or internal — checked with `str.
isspace()` per character), AND the value AS CONFIGURED (nothing stripped)
must encode to >= 32 UTF-8 bytes.

Every scenario runs in an isolated subprocess (same technique
tests/test_stage6a_corrective1_web_config.py already established) — never
disturbs this session's own already-imported web_config module, and a
developer's real .env can never influence the result.

## Minor (token canonicalization)

app.auth_session._is_canonical_token() (renamed from pass #1's
_has_plausible_token_shape()) now requires a full base64url decode +
canonical re-encode round trip, not merely "right length, right alphabet"
— see that function's own docstring for why a 43-character, alphabet-valid
string could otherwise decode to the SAME 32 bytes as a different,
legitimately-issued 43-character string (an "impossible final quantum").
"""

import base64
import json
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_MODULE_IMPORT_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    project_root = sys.argv[1]
    module_name = sys.argv[2]
    sys.path.insert(0, project_root)

    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False

    try:
        module = __import__(module_name)
    except Exception as e:
        print("IMPORT_RESULT=" + json.dumps({
            "raised": True,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }))
        sys.exit(0)

    print("IMPORT_RESULT=" + json.dumps({"raised": False}))
    """
)


def _base_subprocess_env() -> dict:
    env = {}
    for name in ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "USERPROFILE"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _import_web_config(secret: str) -> dict:
    env = _base_subprocess_env()
    env["SESSION_SECRET_KEY"] = secret

    proc = subprocess.run(
        [sys.executable, "-c", _MODULE_IMPORT_SCRIPT, str(PROJECT_ROOT), "web_config"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    result_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("IMPORT_RESULT=")), None
    )
    assert result_line is not None, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return json.loads(result_line[len("IMPORT_RESULT="):])


def _assert_rejected(secret: str) -> None:
    result = _import_web_config(secret)
    assert result["raised"] is True, f"expected rejection for {secret!r}, got {result}"
    assert result["error_type"] == "ValueError"


def _assert_accepted(secret: str) -> None:
    result = _import_web_config(secret)
    assert result["raised"] is False, f"expected acceptance for {secret!r}, got {result}"


# --- Major 2: missing / empty / whitespace-only -----------------------------


def test_missing_secret_is_rejected():
    env = _base_subprocess_env()
    proc = subprocess.run(
        [sys.executable, "-c", _MODULE_IMPORT_SCRIPT, str(PROJECT_ROOT), "web_config"],
        capture_output=True, text=True, timeout=30, env=env,
    )
    result_line = next((l for l in proc.stdout.splitlines() if l.startswith("IMPORT_RESULT=")), None)
    result = json.loads(result_line[len("IMPORT_RESULT="):])
    assert result["raised"] is True
    assert result["error_type"] == "ValueError"


def test_empty_secret_is_rejected():
    _assert_rejected("")


def test_whitespace_only_secret_is_rejected():
    _assert_rejected(" " * 40)


# --- Major 2: the auditor's exact reproduction and its variants ------------


def test_internal_ascii_spaces_padding_is_rejected():
    """The auditor's exact reproduction: only 2 real characters, padded
    with internal spaces to 42 stripped bytes — must now be rejected
    outright, never silently accepted."""
    _assert_rejected("a" + (" " * 40) + "b")


def test_leading_whitespace_is_rejected_even_with_enough_real_material():
    _assert_rejected(" " + ("a" * 32))


def test_trailing_whitespace_is_rejected_even_with_enough_real_material():
    _assert_rejected(("a" * 32) + " ")


def test_internal_tab_is_rejected():
    _assert_rejected(("a" * 16) + "\t" + ("a" * 16))


def test_internal_newline_is_rejected():
    _assert_rejected(("a" * 16) + "\n" + ("a" * 16))


def test_internal_carriage_return_is_rejected():
    _assert_rejected(("a" * 16) + "\r" + ("a" * 16))


def test_internal_unicode_non_breaking_space_is_rejected():
    """Unicode-aware whitespace check (str.isspace()), not merely ASCII
    space — U+00A0 NO-BREAK SPACE is whitespace by Python's own
    definition and must not be treated as real secret material."""
    _assert_rejected(("a" * 16) + " " + ("a" * 16))


# --- Major 2: plain ASCII length boundary -----------------------------------


def test_31_ascii_bytes_is_rejected_as_too_short():
    _assert_rejected("a" * 31)


def test_exactly_32_ascii_bytes_is_accepted():
    _assert_accepted("a" * 32)


def test_more_than_32_ascii_bytes_is_accepted():
    _assert_accepted("a" * 40)


# --- Major 2: multi-byte Unicode, non-whitespace ----------------------------


def test_multibyte_unicode_below_32_utf8_bytes_is_rejected():
    """10 CJK characters, 3 UTF-8 bytes each = 30 bytes < 32 — must be
    rejected by BYTE length even though it's well over 32 CHARACTERS is
    not the point; this value happens to be exactly 10 characters, 30
    bytes."""
    secret = "あ" * 10  # 'あ' * 10 -> 30 UTF-8 bytes
    _assert_rejected(secret)


def test_multibyte_unicode_at_or_above_32_utf8_bytes_is_accepted():
    secret = "あ" * 11  # 'あ' * 11 -> 33 UTF-8 bytes
    result = _import_web_config(secret)
    assert result["raised"] is False, result


# --- Major 2: the value is used exactly as configured, never transformed ---


def test_accepted_secret_is_not_silently_transformed():
    """A valid, no-whitespace, >=32-byte secret must be assigned VERBATIM
    — never stripped/normalized (there is nothing to strip, but this
    guards against a future regression reintroducing silent
    transformation)."""
    secret = "Xx" + ("q" * 41) + "Zz"
    env = _base_subprocess_env()
    env["SESSION_SECRET_KEY"] = secret
    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, sys.argv[1])
        import dotenv
        dotenv.load_dotenv = lambda *a, **k: False
        import web_config
        print("RESULT=" + json.dumps({"key": web_config.SESSION_SECRET_KEY}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT)],
        capture_output=True, text=True, timeout=30, env=env,
    )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT="))
    result = json.loads(line[len("RESULT="):])
    assert result["key"] == secret


# --- MINOR: canonical token-shape hardening ---------------------------------


def test_many_real_generated_tokens_are_all_canonical():
    import secrets
    for _ in range(200):
        token = secrets.token_urlsafe(32)
        assert auth_session._is_canonical_token(token), f"a genuine secrets.token_urlsafe(32) output was rejected: {token!r}"


def test_impossible_final_quantum_is_rejected():
    """A 43-char, alphabet-valid string that decodes to the SAME 32 bytes
    as a real token but is not the CANONICAL encoding of those bytes
    (nonzero unused low bits in the final base64 character) — the exact
    class of value the auditor showed reaching DB lookup under the pass #1
    length+alphabet-only check."""
    import secrets
    real_token = secrets.token_urlsafe(32)
    padding = "=" * (-len(real_token) % 4)
    decoded = base64.urlsafe_b64decode(real_token + padding)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    found = None
    for cand in alphabet:
        if cand == real_token[-1]:
            continue
        candidate = real_token[:-1] + cand
        try:
            if base64.urlsafe_b64decode(candidate + padding) == decoded:
                found = candidate
                break
        except Exception:
            continue

    assert found is not None, "test setup failed to construct a non-canonical same-bytes candidate"
    assert not auth_session._is_canonical_token(found)
    # The canonical original must still be accepted.
    assert auth_session._is_canonical_token(real_token)


@pytest.mark.asyncio
async def test_noncanonical_43_char_token_is_rejected_before_db_access(monkeypatch):
    import secrets

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("db.auth_sessions was reached for a non-canonical token")

    monkeypatch.setattr(db_auth_sessions, "get_active_sync", _fail_if_called)

    real_token = secrets.token_urlsafe(32)
    padding = "=" * (-len(real_token) % 4)
    decoded = base64.urlsafe_b64decode(real_token + padding)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    noncanonical = None
    for cand in alphabet:
        if cand == real_token[-1]:
            continue
        candidate = real_token[:-1] + cand
        if base64.urlsafe_b64decode(candidate + padding) == decoded:
            noncanonical = candidate
            break
    assert noncanonical is not None

    assert await auth_session.resolve_session_user_id(noncanonical, expected_secure=True) is None


@pytest.mark.parametrize("bad_length_delta", [-1, 1])
def test_42_or_44_char_tokens_are_rejected(bad_length_delta):
    import secrets
    token = secrets.token_urlsafe(32)
    if bad_length_delta == -1:
        candidate = token[:-1]
    else:
        candidate = token + "a"
    assert not auth_session._is_canonical_token(candidate)


def test_invalid_characters_are_rejected():
    assert not auth_session._is_canonical_token("!" * 43)
    assert not auth_session._is_canonical_token("=" * 43)
    assert not auth_session._is_canonical_token("+" * 43)  # standard base64 char, not urlsafe
    assert not auth_session._is_canonical_token("/" * 43)  # standard base64 char, not urlsafe


def test_unicode_input_is_rejected():
    assert not auth_session._is_canonical_token("Ω" * 43)
    assert not auth_session._is_canonical_token("あ" * 15)  # different length entirely


def test_canonical_uuid_is_rejected():
    assert not auth_session._is_canonical_token(str(uuid.uuid4()))


def test_digest_hex_is_rejected():
    import hashlib
    digest_hex = hashlib.sha256(b"anything").hexdigest()  # 64 hex chars
    assert not auth_session._is_canonical_token(digest_hex)
    assert not auth_session._is_canonical_token(digest_hex[:43])  # right length, wrong alphabet-adjacent risk


def test_oversized_value_is_rejected():
    assert not auth_session._is_canonical_token("a" * 100_000)
