"""
GitHub OAuth 2.0 Authorization Code + PKCE login (Stage 6B) — the ONLY
place a GitHub identity is ever turned into an authenticated Stage 6A web
session. See app/oauth_transaction.py (state/PKCE lifecycle),
services/github_oauth_client.py (the actual GitHub HTTP calls), and
app/github_identity.py (GitHub numeric id -> canonical UUID) for the
mechanics this module orchestrates; this module itself holds no business
logic beyond that orchestration, matching web/routes.py's own "thin
adapter" convention.

Stage 6C non-merge boundary: GitHub is only ever an EXTERNAL identity
provider — the canonical identity remains `users.id` (db/models.py). This
module never infers that a GitHub identity and an existing Telegram-
backed canonical user are the same human (by email, username, browser
state, or any other heuristic): a previously unseen GitHub identity always
resolves to its OWN canonical user (app/github_identity.py), even if the
same human already has a Telegram-backed one elsewhere. That is an
accepted, INTENTIONAL duality — Stage 6C will add explicit, deliberate
account linking; nothing here ever silently merges or moves document
ownership.

Login-CSRF / session-fixation posture — read this before touching either
route:
  - GET /api/auth/github/login sets a short-lived, HttpOnly cookie (the
    "OAuth-state cookie") carrying the exact SAME raw `state` value handed
    to GitHub, with `SameSite=Lax`. Lax is required, not merely
    acceptable: GitHub's redirect back to our callback is a top-level
    CROSS-SITE GET navigation, and Lax is precisely the SameSite policy
    that still attaches a cookie to that kind of request (Strict would
    not, and the flow would break for every real browser).
  - GET /api/auth/github/callback requires the query-string `state` to
    match this cookie's value (constant-time comparison) BEFORE the
    database transaction is even claimed. Without this check, the
    database-side `state` validation ALONE does not stop a classic OAuth
    "login CSRF": an attacker can legitimately complete their OWN GitHub
    consent up to (but not including) the callback step — producing a
    fully valid, unconsumed state+PKCE transaction that GitHub and our
    database both consider completely legitimate — and then lure the
    VICTIM's browser into merely visiting that exact callback URL (e.g.
    via an auto-submitting link or an <img> tag). Without a browser-bound
    check, our server would happily complete the attacker's OWN GitHub
    identity and place the resulting session cookie in the VICTIM's
    browser: the victim ends up silently, unknowingly authenticated AS
    THE ATTACKER — a stored/silent login-CSRF outcome, and a well-known
    OAuth callback vulnerability class (see RFC 6749 section 10.12). Since
    the victim's browser never visited /login for the attacker's
    transaction, it never received a matching cookie, and the mismatch
    fails closed here — BEFORE claiming (consuming) the transaction, so a
    forged, cookie-less callback attempt can never grief a legitimate
    holder's still-unused transaction either.
  - A fresh session (app.auth_session.create_session_for_github() — Stage
    6C corrective pass, independent-audit MAJOR 1: generation-aware, see
    below) is ALWAYS minted on success, and this module never reads,
    inspects, or reuses any *session* cookie the browser may already be
    carrying — an existing session's identity can never leak into, or be
    confused with, a new GitHub login, and there is no session identifier
    here for an attacker to plant: the raw bearer token is generated
    fresh, entirely server-side, every single time. An old, pre-existing
    browser session (if any) is left exactly as it was — neither adopted,
    merged, nor silently revoked by an unrelated login elsewhere; see
    tests/test_stage6b_github_oauth_routes.py's "existing session"
    coverage for the exact proof of this choice.
  - Stage 6C corrective pass, independent-audit MAJOR 1 (see
    db/telegram_link.py's own module docstring for the full protocol):
    each OAuth transaction created by GET /login captures the current
    global unlink-generation counter (`claimed.auth_generation`,
    propagated from app.oauth_transaction.create_transaction() through to
    this callback). The callback resolves identity through
    app.github_identity.resolve_user_uuid_for_oauth() (never the plain,
    non-generation-aware resolve_user_uuid()) — it rejects the callback
    outright if a concurrent/prior unlink has since tombstoned this GitHub
    identity at a generation newer than the one this transaction captured,
    closing a race where a callback that had already authenticated with
    GitHub, but had not yet resolved/created its mapping, could otherwise
    recreate access an unlink just tore down. Session issuance itself then
    re-resolves the CURRENT mapping a second time, under lock, inside
    create_session_for_github() — see that function's own docstring for
    why the generation check alone is not sufficient. See
    tests/test_stage6c_oauth_generation_race.py for the real-PostgreSQL,
    real-thread proof of every required race, including a stale, already-
    in-flight callback arriving strictly AFTER its identity's unlink.

Callback error handling (Section 15 of the Stage 6B spec): every failure
path — GitHub denial, missing/malformed/mismatched/invalid/expired/
replayed state, missing code, a provider HTTP/network/JSON error, an
invalid GitHub identity payload, a stale OAuth generation (Stage 6C
corrective pass, independent-audit MAJOR 1 — see
app.github_identity.resolve_user_uuid_for_oauth()'s own docstring), or a
StalePostureError from session minting — returns a generic 4xx/5xx
response and mints no session, never leaking the client secret, the PKCE
verifier, the access token, the raw authorization code, or internal
database error detail.

Callback privacy headers and OAuth-binding-cookie lifecycle (Stage 6B
independent-audit corrective pass #1, MAJOR 1 / MINOR 4A/4B/4C):

  - EVERY callback response (success redirect and every error response
    alike) carries `Referrer-Policy: no-referrer` and `Cache-Control:
    no-store` — defense-in-depth against `code`/`state` leaking via a
    Referer header on whatever the browser navigates to next, or via a
    shared/cached copy of this response (see _PRIVACY_HEADERS below).
  - The raw query-string `state` is validated for canonical SHAPE
    (app.oauth_transaction.is_canonical_state() — length, alphabet,
    canonical encoding) BEFORE it is ever passed to `hmac.compare_digest`,
    which raises TypeError (-> an uncaught 500) for a non-ASCII `str`
    operand. A malformed/Unicode `state` therefore always fails safely
    with a 400, and touches neither the OAuth-binding cookie nor the
    database.
  - The OAuth-binding cookie is cleared ONLY once the query-string
    `state` is confirmed to match it (`binding_matches` below) — from
    that point on, this callback is, by definition, the CURRENTLY bound
    transaction for this browser, so EVERY subsequent outcome (success,
    GitHub denial, invalid/expired/replayed state, provider failure,
    stale posture) is terminal for it and clears the cookie. A callback
    whose state does NOT match the cookie (or arrives with no cookie at
    all) never touches it — this is what keeps one tab's failed/forged
    callback from ever clearing a DIFFERENT, still-valid transaction's
    binding cookie in a multi-tab scenario (the cookie always reflects
    whichever `/login` call happened most recently).
  - A GitHub denial (`error=...`) DOES consume (claim-and-discard) the
    matching transaction once binding is confirmed — a denied transaction
    is dead either way, so leaving it claimable would only ever enable a
    pointless replay of the same denial.
  - A callback with a MATCHING binding but NO `code` and no `error` is
    structurally incomplete (never a shape GitHub itself produces) — it
    is rejected WITHOUT claiming the transaction and WITHOUT clearing the
    cookie, so it can never destroy a legitimate, still-pending
    transaction as a side effect of a malformed/adversarial request.
"""

import hmac
import logging

from fastapi import APIRouter, Request, Response, status
from starlette.responses import JSONResponse, RedirectResponse

import app.auth_session as auth_session
import app.github_identity as github_identity
import app.oauth_transaction as oauth_transaction
import github_oauth_config
import services.github_oauth_client as github_oauth_client
import web_config
from web.cookies import set_session_cookie

logger = logging.getLogger(__name__)

router = APIRouter()

_OAUTH_STATE_COOKIE_BASENAME = "github_oauth_state"

# Fixed, same-origin, constant post-login destination (Section 14: "Do NOT
# implement arbitrary user-controlled post-login redirects. No open
# redirect."). Never derived from any request input.
_POST_LOGIN_REDIRECT_PATH = "/api/me"

# Applied to EVERY callback response, success or failure alike (Stage 6B
# independent-audit corrective pass #1, MAJOR 1 item 4 / this module's own
# docstring) — never a substitute for the upstream reverse-proxy/access-
# log invariants documented in README.md, only defense-in-depth against
# this response itself ever leaking `code`/`state` onward.
_PRIVACY_HEADERS = {"Referrer-Policy": "no-referrer", "Cache-Control": "no-store"}


def _oauth_state_cookie_name(secure: bool | None = None) -> str:
    """Same `__Host-`-prefix-when-secure convention as
    web_config.session_cookie_name()/csrf_cookie_name() — see those
    docstrings for why the prefix requires Secure+Path=/+no Domain, and is
    therefore only ever used when the current cookie posture is secure."""
    if secure is None:
        secure = web_config.COOKIE_SECURE
    return f"__Host-{_OAUTH_STATE_COOKIE_BASENAME}" if secure else _OAUTH_STATE_COOKIE_BASENAME


def _clear_oauth_state_cookie(response: Response) -> None:
    response.delete_cookie(
        key=_oauth_state_cookie_name(),
        path="/",
        secure=web_config.COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )


def _error_response(status_code: int, detail: str, *, clear_oauth_cookie: bool = False) -> JSONResponse:
    """Builds a callback error response carrying the same `{"detail":
    ...}` shape FastAPI's own default HTTPException handler would have
    produced, plus this module's privacy headers, plus (when
    `clear_oauth_cookie` is True — see this module's own docstring for
    exactly when that is) the OAuth-binding cookie's deletion. Replaces
    raising HTTPException directly: an exception's own `headers=` param
    cannot carry the two Set-Cookie header instances required for
    web/cookies.py's session-cookie shape, and this keeps every callback
    exit — success or failure — going through one explicit Response
    object that always gets the privacy headers applied."""
    response = JSONResponse({"detail": detail}, status_code=status_code)
    response.headers.update(_PRIVACY_HEADERS)
    if clear_oauth_cookie:
        _clear_oauth_state_cookie(response)
    return response


@router.get("/api/auth/github/login")
async def github_login() -> Response:
    """
    Starts a fresh OAuth transaction and redirects the browser to GitHub's
    authorization endpoint. No CSRF protection is needed for this route
    itself (Section 13's login-CSRF concern is about the CALLBACK, not
    this GET): merely starting a login flow authenticates nothing and
    grants nothing by itself — `state` (verified at the callback, see this
    module's own docstring) is the actual anti-CSRF mechanism for the flow
    as a whole, per RFC 6749 section 10.12.

    Subject to database-authoritative OAuth admission control (Stage 6B
    independent-audit corrective pass #1, MAJOR 2 — see
    app.oauth_transaction.create_transaction()/db.oauth_transactions.
    create_sync()): a rejected attempt returns a safe, generic 429 and
    creates no transaction row at all.
    """
    try:
        transaction = await oauth_transaction.create_transaction()
    except oauth_transaction.OAuthAdmissionRejected:
        return JSONResponse(
            {"detail": "Too many login attempts, please try again shortly"},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    authorize_url = github_oauth_client.build_authorize_url(
        state=transaction.state, code_challenge=transaction.code_challenge
    )

    response = RedirectResponse(authorize_url, status_code=status.HTTP_302_FOUND)
    response.headers.update(_PRIVACY_HEADERS)
    response.set_cookie(
        key=_oauth_state_cookie_name(),
        value=transaction.state,
        max_age=github_oauth_config.OAUTH_TRANSACTION_TTL_SECONDS,
        path="/",
        httponly=True,
        secure=web_config.COOKIE_SECURE,
        samesite="lax",
    )
    return response


@router.get("/api/auth/github/callback")
async def github_callback(request: Request) -> Response:
    query_state = request.query_params.get("state")
    provider_error = request.query_params.get("error")
    code = request.query_params.get("code")

    # Cheap, pre-hmac/pre-DB canonical-shape validation (Stage 6B
    # independent-audit corrective pass #1, MINOR 4B) — see this module's
    # own docstring for why this must run BEFORE hmac.compare_digest.
    if query_state is not None and not oauth_transaction.is_canonical_state(query_state):
        return _error_response(status.HTTP_400_BAD_REQUEST, "Malformed OAuth state")

    if not query_state:
        return _error_response(status.HTTP_400_BAD_REQUEST, "Missing OAuth state")

    cookie_state = request.cookies.get(_oauth_state_cookie_name())
    binding_matches = (
        cookie_state is not None
        and oauth_transaction.is_canonical_state(cookie_state)
        and hmac.compare_digest(cookie_state, query_state)
    )
    if not binding_matches:
        # See this module's own docstring: this is the login-CSRF defense,
        # checked BEFORE the transaction is claimed so a forged, cookie-
        # less request can never consume a legitimate holder's still-valid
        # transaction — and, symmetrically, never clears a DIFFERENT,
        # still-valid transaction's binding cookie either.
        return _error_response(status.HTTP_400_BAD_REQUEST, "OAuth state mismatch")

    # From here on, binding_matches is True: this callback corresponds to
    # the CURRENTLY bound transaction, so every remaining outcome below is
    # terminal for it and clears the OAuth-binding cookie (see this
    # module's own docstring) — except the "missing code" case, which
    # deliberately leaves both the cookie and the transaction untouched.

    if provider_error is not None:
        # A user declining authorization (or any other GitHub-reported
        # error) is a normal, expected outcome — never transformed into
        # authenticated success (Section 15). Consumed here (claim-and-
        # discard) since a denied transaction is dead either way — see
        # this module's own docstring.
        await oauth_transaction.claim_transaction(query_state)
        return _error_response(
            status.HTTP_400_BAD_REQUEST, "GitHub authorization was not granted", clear_oauth_cookie=True
        )

    if not code:
        # Matching binding but no authorization code and no provider
        # error — a structurally incomplete request GitHub itself never
        # produces. Fails safely WITHOUT claiming (consuming) the still-
        # valid transaction (Section 15/MINOR 4C) and without clearing the
        # binding cookie, so a legitimate follow-up callback can still
        # complete normally.
        return _error_response(status.HTTP_400_BAD_REQUEST, "Missing authorization code")

    claimed = await oauth_transaction.claim_transaction(query_state)
    if claimed is None:
        # Unknown, expired, or already-consumed — indistinguishable by
        # design (see app.oauth_transaction.claim_transaction()'s own
        # docstring); a second callback replaying an already-used `state`
        # lands here too.
        return _error_response(
            status.HTTP_400_BAD_REQUEST, "Invalid or expired OAuth state", clear_oauth_cookie=True
        )

    try:
        access_token = await github_oauth_client.exchange_code_for_token(
            code=code,
            code_verifier=claimed.code_verifier,
            redirect_uri=github_oauth_config.GITHUB_REDIRECT_URI,
        )
        github_user_id = await github_oauth_client.fetch_github_user_id(access_token=access_token)
    except github_oauth_client.GithubOAuthError:
        logger.warning("GitHub OAuth provider interaction failed")
        return _error_response(
            status.HTTP_502_BAD_GATEWAY, "GitHub authentication failed", clear_oauth_cookie=True
        )

    # Generation-aware resolution (Stage 6C corrective pass, independent-
    # audit MAJOR 1) — guarantees a github_accounts row exists (creating
    # both it and the canonical user on first login), exactly like the
    # plain resolve_user_uuid() this replaces, but ALSO rejects this
    # transaction outright if a concurrent/prior unlink has since
    # tombstoned this GitHub identity at a generation newer than the one
    # this transaction captured at login start (`claimed.auth_generation`)
    # — see app.github_identity.resolve_user_uuid_for_oauth()'s own
    # docstring for the exact race this closes. Its RETURN VALUE is still
    # deliberately not what session issuance trusts: create_session_for_
    # github() below re-resolves github_user_id -> canonical UUID itself,
    # under a lock, in the SAME transaction as the session insert, and
    # that freshly-locked resolution is the one this callback ultimately
    # relies on for WHICH user the session is minted for (see
    # app/auth_session.py's create_session_for_github() and
    # db/auth_sessions.py's create_for_github_sync() docstrings for the
    # exact race THAT closes) — this call's only job here is the
    # generation gate: None means "reject", anything else means "at least
    # not stale as of this check".
    gated_user_id = await github_identity.resolve_user_uuid_for_oauth(
        github_user_id=github_user_id, auth_generation=claimed.auth_generation
    )
    if gated_user_id is None:
        # This GitHub identity was unlinked at a generation newer than
        # this login flow's own — the claimed transaction stays consumed
        # either way (Section D: "the user must start a genuinely new
        # login"); never resurrect a user/mapping/session for it.
        logger.warning("GitHub OAuth login rejected: stale generation (unlinked since this login began)")
        return _error_response(status.HTTP_400_BAD_REQUEST, "Login must be restarted", clear_oauth_cookie=True)

    try:
        issued = await auth_session.create_session_for_github(github_user_id, issued_secure=web_config.COOKIE_SECURE)
    except auth_session.StalePostureError:
        # Fail closed (Section 12/18): this process's own cookie posture
        # is no longer authoritative — never silently retry under another
        # posture, never issue a cookie anyway.
        logger.warning("GitHub OAuth login rejected: stale session-cookie posture")
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Login temporarily unavailable, please retry",
            clear_oauth_cookie=True,
        )

    if issued is None:
        # Extremely tight race (Stage 6C): the github_accounts row resolved
        # moments ago no longer exists by the time issuance re-resolved it
        # under lock (e.g. a concurrent unlink). Fail closed exactly like
        # StalePostureError above — never fall back to the earlier,
        # possibly-stale resolution.
        logger.warning("GitHub OAuth login rejected: GitHub mapping no longer current at session issuance")
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Login temporarily unavailable, please retry",
            clear_oauth_cookie=True,
        )

    redirect_response = RedirectResponse(_POST_LOGIN_REDIRECT_PATH, status_code=status.HTTP_302_FOUND)
    redirect_response.headers.update(_PRIVACY_HEADERS)
    set_session_cookie(redirect_response, raw_token=issued.raw_token, expires_at=issued.expires_at)
    _clear_oauth_state_cookie(redirect_response)
    return redirect_response
