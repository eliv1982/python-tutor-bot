"""
CSRF defense (Stage 6A) — stateless double-submit cookie, cryptographically
tied to the exact session cookie value rather than to a separately-stored
CSRF secret.

derive_csrf_token(raw_session_token) = HMAC-SHA256(web_config.
SESSION_SECRET_KEY, raw_session_token). web/cookies.py sets this as a
second, non-HttpOnly cookie whenever the session cookie itself is set, so
same-origin JavaScript can read it and echo it back as a custom header
(X-CSRF-Token) on state-changing requests. A cross-site attacker page can
cause the browser to SEND our cookies automatically, but the Same-Origin
Policy prevents it from ever READING their values to also set as a custom
header — that is what makes the double-submit comparison in
web/dependencies.py's require_csrf() a real CSRF defense, not merely
SameSite-in-disguise. SameSite=Lax on both cookies (web/cookies.py) is
kept as an additional, independent layer, per Stage 6A's CSRF posture
requirement — never the sole mechanism.

No extra database row/state is required: deriving the token from the raw
session token itself (which only the legitimate browser and the server
ever see, and which rotates on every new session) means the CSRF token
automatically rotates whenever Stage 6B's OAuth flow rotates/creates a new
session — nothing here needs to change for that.
"""

import hmac
import hashlib

import web_config


def derive_csrf_token(raw_session_token: str) -> str:
    return hmac.new(
        web_config.SESSION_SECRET_KEY.encode("utf-8"),
        raw_session_token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def csrf_token_matches(*, raw_session_token: str, submitted_token: str) -> bool:
    expected = derive_csrf_token(raw_session_token)
    return hmac.compare_digest(expected, submitted_token)
