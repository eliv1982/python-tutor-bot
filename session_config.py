"""
Server-side web-session TTL policy — shared configuration (Stage 6A
independent-audit corrective pass #1, minor finding #1).

Split out of web_config.py: app/auth_session.py (the session application
layer) needs ONLY the session TTL to compute `expires_at` — it has no need
for SESSION_SECRET_KEY (that signs CSRF tokens, a concern of the FastAPI
adapter's cookie/CSRF layer alone, see web/csrf.py) or COOKIE_SECURE/the
cookie-name functions (a concern of web/cookies.py alone). Importing the
whole of web_config.py merely to read a TTL therefore forced
app/auth_session.py to also satisfy web_config.py's SESSION_SECRET_KEY
fail-closed check at import time — a real coupling with no corresponding
need, exactly the kind of accidental import-time requirement
telegram_config.py/config.py's own split (see telegram_config.py's own
docstring) already established the precedent for avoiding in this
codebase. This module follows that same precedent: importable without any
web-adapter-only credential, so app/auth_session.py (and, transitively,
anything that only needs session-lifetime policy) stays importable
independent of whether SESSION_SECRET_KEY is configured.

web_config.py itself does not need anything from here — cookie
construction always receives `expires_at` as an explicit parameter (see
web/cookies.py), never reads a TTL of its own.

SESSION_TTL_SECONDS is read/validated fresh at this module's own import
time (module-level, fail-closed) — same convention as config.py's/
telegram_config.py's own credential checks — never re-validated per call.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# No sliding expiration (Stage 6A explicitly avoids it) — fixed lifetime
# from creation time, in seconds. Default: 14 days.
_DEFAULT_SESSION_TTL_SECONDS = 60 * 60 * 24 * 14

# Upper bound (Stage 6A independent-audit corrective pass #1, Major 4): a
# session that never meaningfully expires defeats the whole point of a
# bounded-lifetime bearer credential. 180 days is comfortably above the
# 14-day default (room for a deliberately longer-lived deployment) while
# still rejecting an obviously-mistaken/absurd value (e.g. a stray extra
# zero, or a value expressed in the wrong unit).
_MAX_SESSION_TTL_SECONDS = 60 * 60 * 24 * 180

_raw_ttl = os.getenv("WEB_SESSION_TTL_SECONDS")
if _raw_ttl is None:
    SESSION_TTL_SECONDS: int = _DEFAULT_SESSION_TTL_SECONDS
else:
    try:
        SESSION_TTL_SECONDS = int(_raw_ttl.strip())
    except ValueError as e:
        raise ValueError(
            f"WEB_SESSION_TTL_SECONDS must be an integer number of seconds, got {_raw_ttl!r}"
        ) from e
    if SESSION_TTL_SECONDS <= 0:
        raise ValueError(
            f"WEB_SESSION_TTL_SECONDS must be a positive number of seconds, got {SESSION_TTL_SECONDS}"
        )
    if SESSION_TTL_SECONDS > _MAX_SESSION_TTL_SECONDS:
        raise ValueError(
            f"WEB_SESSION_TTL_SECONDS={SESSION_TTL_SECONDS} exceeds the maximum allowed "
            f"({_MAX_SESSION_TTL_SECONDS}, 180 days) — an unbounded/excessive session "
            f"lifetime is never a valid configuration"
        )
