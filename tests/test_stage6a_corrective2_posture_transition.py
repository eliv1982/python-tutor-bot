"""
Stage 6A independent-audit corrective pass #2 — Major 1 real-FastAPI,
real-PostgreSQL end-to-end regression proof: a web session must not
silently become valid again after the application's cookie posture
(WEB_COOKIE_SECURE) changes and later reverts.

Pre-fix reproduction (the independent auditor's exact steps):
  1. create a real valid session in secure mode;
  2. browser holds the `__Host-session`/`__Host-csrf_token` cookies;
  3. application posture switches to insecure;
  4. call POST /api/logout;
  5. the auth dependency looks only for the bare `session` cookie name;
  6. request returns 401 BEFORE the logout route ever executes;
  7. no cleanup headers are emitted;
  8. the server-side secure session remains untouched/active the whole time;
  9. posture switches back to secure;
  10. the ORIGINAL secure bearer authenticates again — a session silently
      "revived" purely because of a temporary posture change nobody ever
      explicitly logged out of.

Fix (see db/models.py's WebSession docstring, db/auth_sessions.py's
get_active_sync()/apply_startup_posture_sync(), and web/app.py's
lifespan):
  - every web_sessions row now records `issued_secure` — the cookie
    posture it was created under;
  - ordinary authentication (get_active_sync) only resolves a session
    whose `issued_secure` matches the CURRENTLY EXPECTED posture;
  - AUTHORITATIVELY, every FastAPI startup (lifespan, before serving any
    request) PERMANENTLY revokes every still-active session bound to
    whichever posture is NOT the one the app is starting under. This is
    what makes a transition irreversible — a session dead this way can
    never "come back" on a later revert, regardless of whether logout was
    ever successfully called on it.

This file proves the SEQUENTIAL regression (the auditor's original,
single-process-at-a-time reproduction above). Stage 6A independent-audit
corrective pass #3 discovered and closed a remaining CONCURRENT race in
this same mechanism — an old, still-running opposite-posture PROCESS
could commit a brand-new incompatible session after a newer process's
one-time startup revocation had already run — by making
db.auth_sessions.create_sync() itself transactionally check the
database's own authoritative posture (`web_session_policy`, see
db/models.py's WebSessionPolicy docstring) before ever inserting. See
tests/test_stage6a_corrective3_policy_race.py for that real-thread,
real-lock-contention proof; nothing in this file changed its OWN
assertions for pass #3 — only call sites needed updating for
create_sync()'s new "must match the authoritative DB posture" contract
(every session creation below is now preceded by whatever transition is
needed to make the DB posture match, exactly mirroring what a real
process boundary would have already done).

"Posture transition" is simulated the way it actually happens in a real
deployment: WEB_COOKIE_SECURE is read once at process start
(web_config.py's own module-level import), so it can only ever change
between process restarts — never live, mid-process. Each simulated
transition below is therefore a fresh `with TestClient(create_app()) as
client:` block (which runs the ASGI lifespan startup, including the new
revocation) entered AFTER monkeypatching web_config.COOKIE_SECURE to the
new value — not a live in-process flag flip with no corresponding
restart, which would not reflect how this setting can actually change in
production.
"""

import random
import uuid

import pytest
from starlette.testclient import TestClient

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.identity as db_identity
import web_config
from web.app import create_app
from web.dependencies import CSRF_HEADER_NAME


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — every test here
    needs REAL `users` rows and REAL sessions."""
    yield


def _real_user() -> uuid.UUID:
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def _set_cookie_headers(response) -> list[str]:
    return response.headers.get_list("set-cookie")


def _set_posture(monkeypatch, *, secure: bool) -> None:
    """Simulates the app being (re)started under a given cookie posture —
    see this module's own docstring on why this is monkeypatch + a fresh
    TestClient context, not a live flag flip."""
    monkeypatch.setattr(web_config, "COOKIE_SECURE", secure)
    monkeypatch.setattr(web_config, "WEB_ENV", "production" if secure else "development")


# --- Scenario A: secure -> insecure -> secure -------------------------------


@pytest.mark.asyncio
async def test_scenario_a_secure_session_does_not_revive_after_insecure_then_secure_transition(
    monkeypatch, postgres_db
):
    # 1-2: create + authenticate a real session under SECURE posture.
    _set_posture(monkeypatch, secure=True)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=True)

    with TestClient(create_app()) as client:
        client.cookies.set(web_config.session_cookie_name(secure=True), issued.raw_token)
        response = client.get("/api/me")
        assert response.status_code == 200
        assert response.json()["id"] == str(user_id)

    # 3: application posture transitions to insecure — entering a NEW
    # TestClient context runs the lifespan startup, which revokes every
    # session bound to the now-incompatible (secure) posture. This is the
    # AUTHORITATIVE closure — proven directly against the DB below, before
    # any logout attempt.
    _set_posture(monkeypatch, secure=False)
    with TestClient(create_app()):
        pass

    assert db_auth_sessions.get_active_sync(
        token_hash=auth_session._hash_token(issued.raw_token), expected_secure=True
    ) is None, "the secure-posture session must already be revoked by the startup transition policy"

    # 4-8: the auditor's exact reproduction — call POST /api/logout while
    # insecure, presenting the OLD secure cookie NAME (a real browser would
    # not even transmit a Secure-flagged cookie over plain HTTP; simulated
    # here as a best-effort attempt regardless). The auth dependency only
    # ever looks for the CURRENT posture's cookie name ("session"), so this
    # must 401 before the logout route runs, with zero cleanup headers —
    # this is expected AND fine, because the session was already killed by
    # the authoritative startup revocation above, not by this logout call.
    with TestClient(create_app()) as insecure_client:
        insecure_client.cookies.set("__Host-session", issued.raw_token)
        logout_response = insecure_client.post("/api/logout", headers={CSRF_HEADER_NAME: "irrelevant"})
        assert logout_response.status_code == 401
        assert _set_cookie_headers(logout_response) == []

        me_response = insecure_client.get("/api/me")
        assert me_response.status_code == 401

    # 9-10: posture reverts to secure — the ORIGINAL bearer must NOT
    # authenticate again. This is the actual regression the audit flagged.
    _set_posture(monkeypatch, secure=True)
    with TestClient(create_app()) as client:
        client.cookies.set(web_config.session_cookie_name(secure=True), issued.raw_token)
        response = client.get("/api/me")
        assert response.status_code == 401


# --- Scenario B: insecure -> secure -> insecure (symmetric) -----------------


@pytest.mark.asyncio
async def test_scenario_b_insecure_session_does_not_revive_after_secure_then_insecure_transition(
    monkeypatch, postgres_db
):
    _set_posture(monkeypatch, secure=False)
    # Establish insecure as the DB-authoritative posture BEFORE creating a
    # session under it — create_sync() now transactionally requires this
    # (Stage 6A independent-audit corrective pass #3); a real insecure
    # process would already have done this via its own startup. The
    # fixture's DB default is secure=True, so without this the create_session()
    # call below would raise StalePostureError.
    with TestClient(create_app()):
        pass

    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=False)

    with TestClient(create_app()) as client:
        client.cookies.set(web_config.session_cookie_name(secure=False), issued.raw_token)
        response = client.get("/api/me")
        assert response.status_code == 200
        assert response.json()["id"] == str(user_id)

    # Transition to secure — startup revokes every insecure-posture session.
    _set_posture(monkeypatch, secure=True)
    with TestClient(create_app()):
        pass

    assert db_auth_sessions.get_active_sync(
        token_hash=auth_session._hash_token(issued.raw_token), expected_secure=False
    ) is None

    # Revert to insecure — the original bearer must not revive.
    _set_posture(monkeypatch, secure=False)
    with TestClient(create_app()) as client:
        client.cookies.set(web_config.session_cookie_name(secure=False), issued.raw_token)
        response = client.get("/api/me")
        assert response.status_code == 401


# --- Scenario C: secure production never accepts the bare cookie -----------


@pytest.mark.asyncio
async def test_scenario_c_secure_production_never_authenticates_the_bare_session_cookie(monkeypatch, postgres_db):
    """A REAL, currently-active insecure-posture session's bearer, submitted
    under the bare `session` cookie name while the app is running secure
    production, must not authenticate — neither structurally (the auth
    dependency only ever reads `__Host-session` while secure) nor at the DB
    layer (issued_secure mismatch), even if it somehow got read.

    The posture-FILTER proof (steps below, before any secure-posture
    lifespan ever runs) is deliberately kept separate from the HTTP-level
    proof at the end: entering a TestClient under secure posture also
    triggers the startup revocation policy (Scenario A/B), which would
    itself kill this insecure session as a SEPARATE, additional effect —
    correct, but not what this test is isolating. Proving the filter first
    (via direct db.auth_sessions calls, no lifespan involved) confirms
    get_active_sync()'s `expected_secure` mismatch rejects it on its own
    merits, not merely because it was revoked by something else."""
    _set_posture(monkeypatch, secure=False)
    # Establish insecure as the DB-authoritative posture first — see
    # Scenario B's own comment on why create_sync() now requires this.
    with TestClient(create_app()):
        pass

    user_id = _real_user()
    insecure_issued = await auth_session.create_session(user_id, issued_secure=False)

    # Sanity: genuinely valid under its OWN posture, before anything else
    # has touched it.
    assert (
        await auth_session.resolve_session_user_id(insecure_issued.raw_token, expected_secure=False) == user_id
    )
    # Posture-FILTER proof, in isolation: a currently-active, unexpired,
    # un-revoked insecure-posture session must still be rejected by a
    # secure-posture lookup.
    assert (
        await auth_session.resolve_session_user_id(insecure_issued.raw_token, expected_secure=True) is None
    )

    # HTTP-level proof: entering secure posture (which additionally revokes
    # this session via the startup policy — Scenario A/B's own mechanism,
    # not what's under test here) and submitting the bearer under the bare
    # `session` cookie name must not authenticate.
    _set_posture(monkeypatch, secure=True)
    with TestClient(create_app()) as client:
        client.cookies.set("session", insecure_issued.raw_token)
        response = client.get("/api/me")
        assert response.status_code == 401


# --- Scenario D: ordinary logout still works and cleans up -----------------


@pytest.mark.asyncio
async def test_scenario_d_ordinary_logout_revokes_and_clears_cookies(monkeypatch, postgres_db):
    from web.csrf import derive_csrf_token

    _set_posture(monkeypatch, secure=True)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=True)

    with TestClient(create_app()) as client:
        client.cookies.set(web_config.session_cookie_name(secure=True), issued.raw_token)
        response = client.post(
            "/api/logout", headers={CSRF_HEADER_NAME: derive_csrf_token(issued.raw_token)}
        )
        assert response.status_code == 204
        names = sorted(h.split("=", 1)[0] for h in _set_cookie_headers(response))
        assert names == ["__Host-csrf_token", "__Host-session", "csrf_token", "session"]

    assert await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=True) is None


# --- Scenario E: mismatched-posture logout is handled by startup/session- ---
# --- store revocation, NOT by an alternate-cookie logout-auth fallback -----


@pytest.mark.asyncio
async def test_scenario_e_mismatched_posture_logout_relies_on_startup_revocation_not_alt_cookie_auth(
    monkeypatch, postgres_db
):
    """Explicit design assertion (permitted by this pass's own spec): this
    codebase deliberately does NOT implement a specialized "logout may also
    authenticate the other posture's cookie name" fallback. A logout
    attempt made during a posture mismatch is expected to 401 with zero
    cleanup headers — exactly the auditor's raw reproduction — and that is
    ACCEPTABLE because the authoritative closure already happened at
    startup (revoke_sessions_with_incompatible_posture), independent of
    whether logout is ever called at all. This test exists to make that
    design choice explicit and regression-checked, not merely implied by
    Scenario A."""
    _set_posture(monkeypatch, secure=True)
    user_id = _real_user()
    issued = await auth_session.create_session(user_id, issued_secure=True)

    _set_posture(monkeypatch, secure=False)
    with TestClient(create_app()) as client:
        # The startup that just ran already revoked this session — confirm
        # that BEFORE ever touching the logout endpoint.
        assert db_auth_sessions.get_active_sync(
            token_hash=auth_session._hash_token(issued.raw_token), expected_secure=True
        ) is None

        client.cookies.set("__Host-session", issued.raw_token)
        client.cookies.set("__Host-csrf_token", "irrelevant-would-not-match-anyway")
        response = client.post("/api/logout", headers={CSRF_HEADER_NAME: "irrelevant-would-not-match-anyway"})
        assert response.status_code == 401
        assert _set_cookie_headers(response) == []


# --- Low-level posture-filter and revocation-policy proofs -----------------


def test_get_active_sync_rejects_a_session_created_under_the_other_posture(postgres_db):
    import hashlib
    import secrets
    from datetime import datetime, timedelta, timezone

    user_id = _real_user()
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).digest()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    db_auth_sessions.create_sync(token_hash=token_hash, user_id=user_id, issued_secure=True, expires_at=expires_at)

    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=True) is not None
    assert db_auth_sessions.get_active_sync(token_hash=token_hash, expected_secure=False) is None


def test_apply_startup_posture_only_revokes_the_mismatched_ones(postgres_db):
    """Stage 6A independent-audit corrective pass #3 renamed this
    function's predecessor (revoke_sessions_with_incompatible_posture_sync)
    to apply_startup_posture_sync() and made it transactional against
    create_sync()'s own posture check (see db/models.py's
    WebSessionPolicy docstring). Under correct operation, create_sync()
    itself now refuses to create a session incompatible with the CURRENT
    database posture — so a "True row + False row coexisting" state can
    only legitimately arise from the exact cross-process race window
    tests/test_stage6a_corrective3_policy_race.py reproduces directly.
    Here, a raw INSERT (bypassing create_sync()'s check, same technique
    tests/test_stage6a_corrective1_digest_constraint.py already uses for
    proving DB-level invariants the app layer wouldn't normally produce)
    stands in for that late/incompatible row, so this test can focus on
    apply_startup_posture_sync()'s own revoke-selectivity in isolation."""
    import hashlib
    import secrets
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import insert

    from db.engine import get_sync_engine
    from db.models import WebSession

    user_id = _real_user()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    secure_hash = hashlib.sha256(secrets.token_urlsafe(32).encode("utf-8")).digest()
    insecure_hash = hashlib.sha256(secrets.token_urlsafe(32).encode("utf-8")).digest()

    # Establish insecure as authoritative, then create a COMPATIBLE
    # insecure session through the real API.
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    db_auth_sessions.create_sync(token_hash=insecure_hash, user_id=user_id, issued_secure=False, expires_at=expires_at)

    # Stand in for a late, already-incompatible (secure) row.
    engine = get_sync_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(WebSession).values(
                session_token_hash=secure_hash, user_id=user_id, issued_secure=True, expires_at=expires_at
            )
        )

    revoked_count = db_auth_sessions.apply_startup_posture_sync(requested_secure=False)

    assert revoked_count == 1
    assert db_auth_sessions.get_active_sync(token_hash=secure_hash, expected_secure=True) is None
    assert db_auth_sessions.get_active_sync(token_hash=insecure_hash, expected_secure=False) is not None


def test_apply_startup_posture_never_touches_the_user_row(postgres_db):
    """Deliberately narrow: only web_sessions is affected — users,
    telegram_accounts, ownership are untouched."""
    import hashlib
    import secrets
    from datetime import datetime, timedelta, timezone

    user_id = _real_user()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    insecure_hash = hashlib.sha256(secrets.token_urlsafe(32).encode("utf-8")).digest()

    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    db_auth_sessions.create_sync(token_hash=insecure_hash, user_id=user_id, issued_secure=False, expires_at=expires_at)

    db_auth_sessions.apply_startup_posture_sync(requested_secure=True)  # revokes the insecure session above

    # The canonical user row created by _real_user() must still resolve
    # unchanged — revoking incompatible sessions must never touch `users`.
    resolved_again = db_identity.get_user_by_id_sync(user_id)
    assert resolved_again is not None
    assert resolved_again.id == user_id


def test_apply_startup_posture_is_idempotent_and_safe_with_no_incompatible_sessions(postgres_db):
    assert db_auth_sessions.apply_startup_posture_sync(requested_secure=True) == 0
    assert db_auth_sessions.apply_startup_posture_sync(requested_secure=False) == 0
    assert db_auth_sessions.apply_startup_posture_sync(requested_secure=True) == 0
