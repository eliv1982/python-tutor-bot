"""
GitHub OAuth-only configuration (Stage 6B) — mirrors telegram_config.py's/
web_config.py's own isolation rationale: these values are required ONLY by
the GitHub login routes (web/github_oauth.py), so they live in their own
module rather than web_config.py. web/app.py's create_app() imports
web/github_oauth.py unconditionally (GitHub login is a core route of this
adapter, not an optional feature flag), so in practice starting the web
adapter at all now requires this module's checks to pass — the same
fail-closed posture web_config.py's own SESSION_SECRET_KEY check already
established for every other web route, including ones (like /healthz)
that don't themselves touch a GitHub credential either. Telegram's
main.py/bot.py never imports web/*.py at all, so this has no effect on
the Telegram adapter's own startup requirements.

Reads web_config.WEB_ENV (never the reverse — web_config.py has no
knowledge of this module) to decide whether an http:// redirect URI is
ever acceptable, exactly mirroring web_config.py's own
WEB_ENV/COOKIE_SECURE production guard.

Fails closed at import time for a missing/empty/malformed value — same
posture as every other credential check in this codebase (config.py's
OPENAI_API_KEY/ANTHROPIC_API_KEY, telegram_config.py's
TELEGRAM_BOT_TOKEN, web_config.py's SESSION_SECRET_KEY).
"""

import os
from urllib.parse import urlparse

from dotenv import load_dotenv

import web_config

load_dotenv()

# Current GitHub OAuth web-flow endpoints (Section 28 of the Stage 6B
# spec) — isolated here as named constants rather than inlined at each
# call site, so services/github_oauth_client.py's tests can assert on them
# directly and a future endpoint change has exactly one place to edit.
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API_URL = "https://api.github.com/user"


def _no_whitespace(value: str) -> bool:
    return not any(ch.isspace() for ch in value)


GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "")
if not GITHUB_CLIENT_ID or not _no_whitespace(GITHUB_CLIENT_ID):
    raise ValueError(
        "GITHUB_CLIENT_ID is not set (or contains whitespace) in .env file — required for "
        "GitHub OAuth login (create a GitHub OAuth App to obtain one)"
    )

GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "")
if not GITHUB_CLIENT_SECRET or not _no_whitespace(GITHUB_CLIENT_SECRET):
    raise ValueError(
        "GITHUB_CLIENT_SECRET is not set (or contains whitespace) in .env file — required for "
        "GitHub OAuth login. This is a server-side-only secret: never exposed to the browser."
    )

GITHUB_REDIRECT_URI: str = os.getenv("GITHUB_REDIRECT_URI", "").strip()
if not GITHUB_REDIRECT_URI:
    raise ValueError(
        "GITHUB_REDIRECT_URI is not set in .env file — required for GitHub OAuth login, and "
        "must exactly match the callback URL registered on the GitHub OAuth App "
        "(e.g. http://127.0.0.1:8000/api/auth/github/callback for local development)"
    )

# The web adapter's callback route is a fixed path (web/github_oauth.py's
# `@router.get("/api/auth/github/callback")`) — hardcoded here rather than
# imported (this module must stay importable/side-effect-free independent
# of web/*.py, and web/github_oauth.py itself imports THIS module, so the
# reverse import would be circular) — same "frozen literal, not a live
# cross-module reference" posture alembic/versions/*.py already documents
# for its own hardcoded literals.
_REQUIRED_CALLBACK_PATH = "/api/auth/github/callback"

# Stage 6B independent-audit corrective pass #1, MINOR 3: tightened
# redirect-URI validation. The previous version only checked scheme/host —
# it accepted userinfo, a fragment, query parameters (our callback route
# accepts none), and any malformed port silently, and allowed `localhost`
# as a second dev-loopback spelling alongside `127.0.0.1`. Every rejection
# below runs BEFORE any GitHub HTTP interaction is even possible (this is
# still plain import-time configuration validation), so a misconfigured
# value fails closed at process startup, not at first login attempt.
_parsed_redirect_uri = urlparse(GITHUB_REDIRECT_URI)

if not _parsed_redirect_uri.netloc or not _parsed_redirect_uri.path:
    raise ValueError(
        f"GITHUB_REDIRECT_URI must be an absolute URL with a host and path, got "
        f"{GITHUB_REDIRECT_URI!r}"
    )
if _parsed_redirect_uri.username is not None or _parsed_redirect_uri.password is not None:
    raise ValueError("GITHUB_REDIRECT_URI must not contain userinfo (a username/password before the host)")
if _parsed_redirect_uri.fragment:
    raise ValueError("GITHUB_REDIRECT_URI must not contain a fragment (#...)")
if _parsed_redirect_uri.query:
    raise ValueError(
        "GITHUB_REDIRECT_URI must not contain query parameters — the callback route accepts none"
    )
try:
    # Merely ACCESSING .port is what triggers urllib's own out-of-range/
    # non-numeric port validation (raises ValueError) — evaluated here,
    # before any scheme-specific check below, so an invalid port fails
    # closed uniformly regardless of scheme.
    _parsed_redirect_uri.port
except ValueError as e:
    raise ValueError(f"GITHUB_REDIRECT_URI has an invalid port: {GITHUB_REDIRECT_URI!r}") from e
if _parsed_redirect_uri.path != _REQUIRED_CALLBACK_PATH:
    raise ValueError(
        f"GITHUB_REDIRECT_URI path must be exactly {_REQUIRED_CALLBACK_PATH!r} (the web "
        f"adapter's fixed callback route), got {_parsed_redirect_uri.path!r} — no other path is "
        f"ever accepted"
    )

if _parsed_redirect_uri.scheme == "https":
    if not _parsed_redirect_uri.hostname:
        raise ValueError("GITHUB_REDIRECT_URI must have a valid hostname")
elif _parsed_redirect_uri.scheme == "http":
    if web_config.WEB_ENV != "development":
        raise ValueError(
            "GITHUB_REDIRECT_URI may only use http:// when WEB_ENV=development — production "
            "must use an https:// callback URL"
        )
    # Only the two LITERAL loopback addresses — `urlparse(...).hostname`
    # normalizes bracketed IPv6 (`http://[::1]:8000/...`) to the bare
    # `::1` form, so this handles both IPv4/IPv6 loopback spellings
    # uniformly. `localhost` is deliberately NOT accepted (Stage 6B
    # independent-audit corrective pass #1, MINOR 3): this codebase's own
    # documented manual-testing callback (.env.example, README.md) is
    # `http://127.0.0.1:8000/...`, and accepting a second, DNS-resolved
    # spelling for "the same thing" only widens what a misconfigured
    # deployment could accidentally accept as "development-loopback"
    # without narrowing anything a real developer needs.
    if _parsed_redirect_uri.hostname not in ("127.0.0.1", "::1"):
        raise ValueError(
            "an http:// GITHUB_REDIRECT_URI is only allowed for the literal loopback hosts "
            "127.0.0.1 or ::1 in development (not 'localhost', and never an arbitrary external "
            f"host), got host {_parsed_redirect_uri.hostname!r}"
        )
else:
    raise ValueError(
        f"GITHUB_REDIRECT_URI must use the http or https scheme, got "
        f"{_parsed_redirect_uri.scheme!r} — dangerous/unexpected schemes are rejected"
    )

# Short-lived OAuth transaction TTL (Section 5 of the Stage 6B spec: "a
# short transaction TTL such as 5-10 minutes is appropriate... must not
# exceed GitHub authorization-code lifetime unnecessarily"). Default: 10
# minutes. Bounded above at 15 minutes — comfortably above the default
# while still rejecting an obviously-mistaken/excessive value, mirroring
# session_config.py's own WEB_SESSION_TTL_SECONDS upper-bound rationale.
_DEFAULT_OAUTH_TRANSACTION_TTL_SECONDS = 600
_MAX_OAUTH_TRANSACTION_TTL_SECONDS = 900

_raw_ttl = os.getenv("GITHUB_OAUTH_TRANSACTION_TTL_SECONDS")
if _raw_ttl is None:
    OAUTH_TRANSACTION_TTL_SECONDS: int = _DEFAULT_OAUTH_TRANSACTION_TTL_SECONDS
else:
    try:
        OAUTH_TRANSACTION_TTL_SECONDS = int(_raw_ttl.strip())
    except ValueError as e:
        raise ValueError(
            f"GITHUB_OAUTH_TRANSACTION_TTL_SECONDS must be an integer number of seconds, got "
            f"{_raw_ttl!r}"
        ) from e
    if OAUTH_TRANSACTION_TTL_SECONDS <= 0:
        raise ValueError(
            f"GITHUB_OAUTH_TRANSACTION_TTL_SECONDS must be a positive number of seconds, got "
            f"{OAUTH_TRANSACTION_TTL_SECONDS}"
        )
    if OAUTH_TRANSACTION_TTL_SECONDS > _MAX_OAUTH_TRANSACTION_TTL_SECONDS:
        raise ValueError(
            f"GITHUB_OAUTH_TRANSACTION_TTL_SECONDS={OAUTH_TRANSACTION_TTL_SECONDS} exceeds the "
            f"maximum allowed ({_MAX_OAUTH_TRANSACTION_TTL_SECONDS}, 15 minutes) — an OAuth "
            f"transaction must stay short-lived"
        )

# --- OAuth admission control (Stage 6B independent-audit corrective pass
# #1, MAJOR 2) ----------------------------------------------------------------
#
# Bounds db.oauth_transactions.create_sync()'s database-authoritative,
# GLOBAL admission control — see that function's own docstring and
# db/models.py's GithubOAuthAdmission docstring for the full transactional
# protocol this closes an unauthenticated storage-exhaustion attack with.
# Deliberately GLOBAL (not per-client-IP): this application has no
# trustworthy reverse-proxy client-IP contract yet (see README.md's
# production deployment invariants).
#
# The rate WINDOW LENGTH itself is intentionally not configurable — a
# fixed 60-second window (i.e. these two settings are both "per minute"/
# "at any moment") keeps the design and its tests simple, and nothing
# about this application's expected interactive login volume needs a
# different window length.
OAUTH_RATE_WINDOW_SECONDS: int = 60

# Fail-closed bounds for BOTH settings below (Section 7: "invalid/non-
# positive configuration must fail closed" / "do not allow unbounded
# configuration values"): each must be a positive integer, and each is
# capped well above any value a legitimate low-volume interactive login
# service would need, so a typo can never silently disable the bound
# entirely (e.g. an accidentally-huge value) while still leaving generous
# headroom for ordinary manual use (Section 7: "do not choose tiny values
# that make ordinary manual use fragile").
_DEFAULT_OAUTH_MAX_STARTS_PER_MINUTE = 30
_MAX_OAUTH_MAX_STARTS_PER_MINUTE = 10_000

_DEFAULT_OAUTH_MAX_OUTSTANDING_TRANSACTIONS = 200
_MAX_OAUTH_MAX_OUTSTANDING_TRANSACTIONS = 100_000


def _parse_positive_bounded_int(*, env_var: str, default: int, maximum: int) -> int:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError as e:
        raise ValueError(f"{env_var} must be a positive integer, got {raw!r}") from e
    if value <= 0:
        raise ValueError(f"{env_var} must be a positive integer, got {value}")
    if value > maximum:
        raise ValueError(f"{env_var}={value} exceeds the maximum allowed ({maximum})")
    return value


# Max GitHub OAuth login STARTS (successful GET /login admissions) per
# OAUTH_RATE_WINDOW_SECONDS window, GLOBAL across every process sharing
# this database — the primary defense against a flood of unauthenticated
# `/login` requests, independent of how quickly any of them expire.
OAUTH_MAX_STARTS_PER_MINUTE: int = _parse_positive_bounded_int(
    env_var="GITHUB_OAUTH_MAX_STARTS_PER_MINUTE",
    default=_DEFAULT_OAUTH_MAX_STARTS_PER_MINUTE,
    maximum=_MAX_OAUTH_MAX_STARTS_PER_MINUTE,
)

# Hard physical-storage bound: the maximum number of `github_oauth_transactions`
# rows allowed to exist AT ONCE (checked after each call's own expired-row
# cleanup — Section 9). This is what makes storage growth impossible even
# under a slow trickle of starts that individually stay under the rate
# limit but are never completed or allowed to expire.
OAUTH_MAX_OUTSTANDING_TRANSACTIONS: int = _parse_positive_bounded_int(
    env_var="GITHUB_OAUTH_MAX_OUTSTANDING_TRANSACTIONS",
    default=_DEFAULT_OAUTH_MAX_OUTSTANDING_TRANSACTIONS,
    maximum=_MAX_OAUTH_MAX_OUTSTANDING_TRANSACTIONS,
)
