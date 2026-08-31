"""
Web-adapter-only configuration (Stage 6A) — mirrors telegram_config.py's
own isolation rationale: SESSION_SECRET_KEY is required ONLY by the
FastAPI adapter (web/*.py), so it lives in its own module rather than
config.py. This keeps config.py, db/*.py, and every Telegram-only code
path importable without it, exactly as telegram_config.py already does for
TELEGRAM_BOT_TOKEN. Session TTL policy lives in session_config.py instead
of here (independent-audit corrective pass #1, minor finding #1 — see that
module's own docstring): app/auth_session.py needs the TTL but not the
CSRF secret or cookie posture, so it imports session_config.py, never this
module.

Fails closed at import time if SESSION_SECRET_KEY is missing, too weak, or
contains any whitespace — same posture as config.py's OPENAI_API_KEY/
ANTHROPIC_API_KEY (never a silent insecure default for a value that signs
CSRF tokens), hardened further by independent-audit corrective pass #1
(Major 4: a one-character or whitespace-only secret used to pass this
check silently) and corrective pass #2 (Major 2: pass #1's "strip, then
check byte length" rule still let a WHITESPACE-PADDED secret through — a
value like `"a" + " " * 40 + "b"` has only 2 bytes of real material but
strips to 42 bytes, satisfying the old >=32 rule. The rule is now
"no whitespace anywhere, and the value AS CONFIGURED must be >=32 UTF-8
bytes" — see below).

COOKIE_SECURE and the cookie-name functions are read/computed FRESH at
each call site (never bound once at import time) — the same convention
db/settings.py's DATABASE_URL and rag/constants.py's DATA_DIR already use,
so tests can monkeypatch this module's COOKIE_SECURE attribute directly
(see tests/test_stage6a_cookies_csrf.py) and have every downstream cookie
helper observe the change immediately.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _parse_strict_bool(raw: str | None, *, var_name: str, default: bool) -> bool:
    """Strict boolean parsing for security-sensitive flags (Stage 6A
    independent-audit corrective pass #1, Blocker 2) — an unset variable
    uses `default`, but any variable that IS set must be an explicit,
    recognized boolean spelling or this raises. The pre-fix version treated
    any unrecognized string (e.g. a typo like "definitely-not-a-boolean")
    as false, silently disabling the Secure cookie flag — a fail-OPEN
    behavior for a security-critical setting. Never silently coerces an
    unrecognized value to either boolean."""
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise ValueError(
        f"{var_name} must be one of true/false/1/0/yes/no (case-insensitive), got {raw!r}"
    )


SESSION_SECRET_KEY: str = os.getenv("SESSION_SECRET_KEY", "")

# Minimum length for the HMAC key that signs CSRF tokens (web/csrf.py) —
# Stage 6A independent-audit corrective pass #1, Major 4, rule REVISED by
# corrective pass #2, Major 2. Not a statistical entropy estimate
# (deliberately, per pass #1's own guidance) — a plain minimum-byte-length
# floor is sufficient and simple to reason about. 32 bytes (256 bits)
# matches the entropy of the session token itself (app/auth_session.py's
# _TOKEN_BYTES) so the signing key is never the weaker link.
#
# The rule is now simple and unambiguous, with no "count only the
# meaningful part" logic to get subtly wrong:
#   1. SESSION_SECRET_KEY must contain NO Unicode whitespace character
#      anywhere (leading, trailing, OR internal) — checked with `str.
#      isspace()` per character (Unicode-aware: catches tabs, newlines,
#      and non-ASCII whitespace like U+00A0 NO-BREAK SPACE, not merely
#      ASCII space).
#   2. The value AS CONFIGURED (no stripping — there is nothing to strip a
#      whitespace-free string of) must encode to >= 32 UTF-8 bytes.
# Pass #1's rule instead stripped LEADING/TRAILING whitespace only, then
# measured the stripped value's byte length — which still let INTERNAL
# whitespace count as "material": `"a" + " " * 40 + "b"` stripped to 42
# bytes (>= 32, accepted) despite carrying only 2 real characters. The
# auditor's exact reproduction. Rejecting whitespace outright removes the
# ambiguity entirely rather than trying to define "meaningful" padding.
#
# The assigned value below is the ORIGINAL, UNMODIFIED secret — validation
# inspects it but never strips/normalizes the actual HMAC key material; the
# whitespace rule above already guarantees stripping would be a no-op for
# any value that passes validation.
_MIN_SESSION_SECRET_KEY_BYTES = 32

if not SESSION_SECRET_KEY:
    raise ValueError(
        "SESSION_SECRET_KEY is not set in .env file — required to sign CSRF tokens for the "
        "web adapter"
    )
if any(ch.isspace() for ch in SESSION_SECRET_KEY):
    raise ValueError(
        "SESSION_SECRET_KEY must not contain any whitespace character (space, tab, newline, or "
        "other Unicode whitespace), anywhere in the value — found at least one"
    )
if len(SESSION_SECRET_KEY.encode("utf-8")) < _MIN_SESSION_SECRET_KEY_BYTES:
    raise ValueError(
        f"SESSION_SECRET_KEY is too short ({len(SESSION_SECRET_KEY.encode('utf-8'))} bytes) — "
        f"at least {_MIN_SESSION_SECRET_KEY_BYTES} bytes are required. Generate one with: "
        f"python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )

# Fail-safe default: cookies are Secure unless a developer explicitly opts
# out for local plain-HTTP testing (WEB_COOKIE_SECURE=false). Production
# must never rely on this default being weakened silently.
COOKIE_SECURE: bool = _parse_strict_bool(
    os.getenv("WEB_COOKIE_SECURE"), var_name="WEB_COOKIE_SECURE", default=True
)

# Deployment posture guard (Stage 6A independent-audit corrective pass #1,
# Blocker 2 — "no production guard preventing insecure-cookie mode").
# Defaults to "production" (the fail-safe posture, consistent with
# COOKIE_SECURE's own fail-safe default above): a deployment that never set
# WEB_ENV at all is treated as production and therefore may never disable
# Secure cookies. A local/test insecure-cookie mode remains available, but
# only by EXPLICITLY setting WEB_ENV=development alongside
# WEB_COOKIE_SECURE=false — never implicitly.
WEB_ENV: str = os.getenv("WEB_ENV", "production").strip().lower()
_VALID_WEB_ENVS = ("production", "development")
if WEB_ENV not in _VALID_WEB_ENVS:
    raise ValueError(f"WEB_ENV must be one of {_VALID_WEB_ENVS!r}, got {WEB_ENV!r}")

if WEB_ENV == "production" and not COOKIE_SECURE:
    raise ValueError(
        "WEB_COOKIE_SECURE=false is not allowed when WEB_ENV=production — insecure "
        "(non-Secure) cookies must never be served in production. Set WEB_ENV=development "
        "for local/test insecure-cookie use."
    )


def session_cookie_name(secure: bool | None = None) -> str:
    """`__Host-` prefix requires Secure + Path=/ + no Domain attribute
    (browsers enforce this; see web/cookies.py) — only usable when
    `secure` is true, i.e. never for local plain-HTTP development.

    `secure` defaults to the CURRENT COOKIE_SECURE posture when omitted
    (every call site except web/cookies.py's clear_session_cookie() —
    Stage 6A independent-audit corrective pass #1, Major 3 — uses the
    default); passing it explicitly lets a caller ask for the OTHER
    posture's fixed name, e.g. to defensively clear a cookie issued under a
    since-changed WEB_COOKIE_SECURE configuration."""
    if secure is None:
        secure = COOKIE_SECURE
    return "__Host-session" if secure else "session"


def csrf_cookie_name(secure: bool | None = None) -> str:
    """See session_cookie_name()'s docstring — same `secure` parameter
    convention."""
    if secure is None:
        secure = COOKIE_SECURE
    return "__Host-csrf_token" if secure else "csrf_token"
