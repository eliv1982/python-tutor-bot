"""
Stage 6C regression tests: POST /api/unlink/github (Section L) — the three
branches (Telegram-linked kept, GitHub-only deleted, data-bearing-without-
Telegram rejected atomically), cookie/CSRF behavior, and the underlying
db.telegram_link.unlink_github_sync() contract directly. Real disposable
PostgreSQL via tests/conftest.py's postgres_db.
"""

import random
import uuid

import pytest
from starlette.testclient import TestClient

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.documents as db_documents
import db.github_identity as db_github_identity
import db.identity as db_identity
import db.preferences as db_preferences
import db.telegram_link as db_telegram_link
import web_config
from concurrency_helpers import capture
from db.engine import get_sync_engine
from db.models import GithubAccount, TelegramLinkAttempt, User, WebSession
from sqlalchemy import select
from sqlalchemy.orm import Session
from web.app import create_app
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME

UNLINK_PATH = "/api/unlink/github"


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


@pytest.fixture(autouse=True)
def _insecure_posture_for_testing(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    yield


def _github_only_user() -> uuid.UUID:
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    return db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)


async def _session_for(user_id: uuid.UUID):
    return await auth_session.create_session(user_id, issued_secure=False)


def _authed_client(issued) -> TestClient:
    client = TestClient(create_app())
    client.cookies.set(web_config.session_cookie_name(), issued.raw_token)
    return client


def _csrf_headers(issued) -> dict:
    return {CSRF_HEADER_NAME: derive_csrf_token(issued.raw_token)}


# ---------------------------------------------------------------------------
# A. Direct db.telegram_link.unlink_github_sync() branch proofs
# ---------------------------------------------------------------------------


def test_github_only_unlink_deletes_the_empty_user(postgres_db):
    user_id = _github_only_user()
    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.USER_DELETED

    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, user_id) is None
        assert session.execute(select(GithubAccount).where(GithubAccount.user_id == user_id)).first() is None


def test_telegram_linked_unlink_preserves_user_and_data(postgres_db):
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    user_id = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    from db.models import GithubAccount as GA

    engine = get_sync_engine()
    with Session(engine) as session:
        session.add(GA(github_user_id=github_id, user_id=user_id))
        session.commit()
    db_preferences.set_mode_sync(user_id, "text")

    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.TELEGRAM_KEPT

    with Session(engine) as session:
        assert session.get(User, user_id) is not None
        assert session.execute(select(GithubAccount).where(GithubAccount.user_id == user_id)).first() is None
    mode, _ = db_preferences.get_preferences_sync(user_id)
    assert mode == "text"  # Telegram-owned data survives untouched


def test_data_bearing_without_telegram_unlink_rejects_atomically(postgres_db):
    user_id = _github_only_user()
    db_preferences.set_mode_sync(user_id, "voice")

    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.REJECTED

    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(User, user_id) is not None
        assert session.execute(select(GithubAccount).where(GithubAccount.user_id == user_id)).first() is not None
    mode, _ = db_preferences.get_preferences_sync(user_id)
    assert mode == "voice"


def test_no_current_mapping_returns_rejected_without_mutation(postgres_db):
    user_id = _github_only_user()
    db_telegram_link.unlink_github_sync(user_id=user_id)  # first call deletes it (USER_DELETED)
    # user is gone now; call again is meaningless for the SAME id, so use a
    # fresh telegram-backed user with no github mapping at all instead.
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    fresh_user = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    outcome = db_telegram_link.unlink_github_sync(user_id=fresh_user)
    assert outcome == db_telegram_link.UnlinkOutcome.REJECTED


def test_unlink_deletes_an_outstanding_attempt_in_every_branch(postgres_db):
    import hashlib
    import secrets as _secrets
    from datetime import datetime, timedelta, timezone

    user_id = _github_only_user()
    raw_secret = _secrets.token_urlsafe(32)
    db_telegram_link.create_attempt_sync(
        web_user_id=user_id,
        link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.get(TelegramLinkAttempt, user_id) is not None

    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.USER_DELETED
    with Session(engine) as session:
        assert session.get(TelegramLinkAttempt, user_id) is None


# ---------------------------------------------------------------------------
# B. HTTP endpoint: auth, CSRF, cookies, response shape
# ---------------------------------------------------------------------------


def test_no_session_is_unauthenticated():
    client = TestClient(create_app())
    response = client.post(UNLINK_PATH, headers={CSRF_HEADER_NAME: "irrelevant"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_valid_session_without_csrf_is_rejected():
    user_id = _github_only_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)
    response = client.post(UNLINK_PATH)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_successful_unlink_clears_session_and_csrf_cookies():
    user_id = _github_only_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    response = client.post(UNLINK_PATH, headers=_csrf_headers(issued))

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    set_cookie_headers = response.headers.get_list("set-cookie")
    assert any(h.startswith(web_config.session_cookie_name() + "=") for h in set_cookie_headers)
    assert any(h.startswith(web_config.csrf_cookie_name() + "=") for h in set_cookie_headers)
    for h in set_cookie_headers:
        assert "Max-Age=0" in h or "expires" in h.lower()

    # And the session itself is now genuinely gone server-side.
    assert await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=False) is None


@pytest.mark.asyncio
async def test_rejected_unlink_does_not_clear_cookies_or_revoke_the_session():
    user_id = _github_only_user()
    db_preferences.set_mode_sync(user_id, "text")  # forces the REJECTED branch
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    response = client.post(UNLINK_PATH, headers=_csrf_headers(issued))

    assert response.status_code == 409
    set_cookie_headers = response.headers.get_list("set-cookie")
    assert set_cookie_headers == []
    # The session is still valid — nothing was revoked.
    assert await auth_session.resolve_session_user_id(issued.raw_token, expected_secure=False) == user_id


@pytest.mark.asyncio
async def test_telegram_kept_branch_via_http_returns_ok_and_preserves_user():
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    user_id = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    engine = get_sync_engine()
    with Session(engine) as session:
        session.add(GithubAccount(github_user_id=github_id, user_id=user_id))
        session.commit()

    issued = await _session_for(user_id)
    client = _authed_client(issued)
    response = client.post(UNLINK_PATH, headers=_csrf_headers(issued))

    assert response.status_code == 200
    with Session(engine) as session:
        assert session.get(User, user_id) is not None


# ---------------------------------------------------------------------------
# D. Generation/tombstone protocol atomicity (Stage 6C corrective pass,
# independent-audit MAJOR 1, Section E)
# ---------------------------------------------------------------------------


def _admission_generation() -> int:
    from db.models import GITHUB_OAUTH_ADMISSION_ID, GithubOAuthAdmission

    with Session(get_sync_engine()) as session:
        return session.execute(
            select(GithubOAuthAdmission.unlink_generation).where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
        ).scalar_one()


def _tombstone_generation(github_id: int):
    from db.models import GithubUnlinkTombstone

    with Session(get_sync_engine()) as session:
        return session.execute(
            select(GithubUnlinkTombstone.unlink_generation).where(GithubUnlinkTombstone.github_user_id == github_id)
        ).scalar_one_or_none()


def test_successful_telegram_kept_unlink_bumps_generation_and_writes_tombstone(postgres_db):
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    user_id = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    with Session(get_sync_engine()) as session:
        session.add(GithubAccount(github_user_id=github_id, user_id=user_id))
        session.commit()

    generation_before = _admission_generation()
    assert _tombstone_generation(github_id) is None

    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.TELEGRAM_KEPT

    assert _admission_generation() == generation_before + 1
    assert _tombstone_generation(github_id) == generation_before + 1


def test_successful_user_deleted_unlink_bumps_generation_and_writes_tombstone(postgres_db):
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    user_id = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)

    generation_before = _admission_generation()
    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.USER_DELETED

    assert _admission_generation() == generation_before + 1
    assert _tombstone_generation(github_id) == generation_before + 1


def test_repeated_unlink_relink_advances_the_same_tombstone_row_not_a_new_one(postgres_db):
    """A SECOND unlink of the same (re-linked) GitHub identity advances the
    existing tombstone row's generation rather than accumulating history —
    only the latest generation ever matters for the staleness check."""
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    user_a = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    assert db_telegram_link.unlink_github_sync(user_id=user_a) == db_telegram_link.UnlinkOutcome.USER_DELETED
    first_generation = _tombstone_generation(github_id)
    assert first_generation is not None

    user_b = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    assert user_b != user_a
    assert db_telegram_link.unlink_github_sync(user_id=user_b) == db_telegram_link.UnlinkOutcome.USER_DELETED

    with Session(get_sync_engine()) as session:
        from db.models import GithubUnlinkTombstone

        rows = list(
            session.execute(
                select(GithubUnlinkTombstone).where(GithubUnlinkTombstone.github_user_id == github_id)
            ).scalars()
        )
    assert len(rows) == 1  # advanced in place, never a second row
    assert rows[0].unlink_generation == first_generation + 1


def test_ambiguous_domain_data_rejection_never_touches_generation_or_tombstone(postgres_db):
    user_id = _github_only_user()
    with Session(get_sync_engine()) as session:
        github_id_value = session.execute(
            select(GithubAccount.github_user_id).where(GithubAccount.user_id == user_id)
        ).scalar_one()
    db_preferences.set_mode_sync(user_id, "voice")

    generation_before = _admission_generation()

    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.REJECTED

    assert _admission_generation() == generation_before
    assert _tombstone_generation(github_id_value) is None


def test_no_current_mapping_rejection_never_touches_generation(postgres_db):
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    fresh_user = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    generation_before = _admission_generation()
    outcome = db_telegram_link.unlink_github_sync(user_id=fresh_user)
    assert outcome == db_telegram_link.UnlinkOutcome.REJECTED
    assert _admission_generation() == generation_before


def test_mapping_moved_by_concurrent_merge_while_unlink_waits_fails_safely(postgres_db):
    """Unlink's unlocked probe learns a candidate github_user_id for
    `source`, then acquires the per-GitHub advisory lock for it — a lock
    redemption/merge never contends on (merge only ever locks
    `github_accounts`/`users` rows directly). This lets a genuinely
    concurrent merge run to completion WHILE unlink is paused holding only
    the advisory lock, moving the mapping from `source` onto `target`
    before unlink ever revalidates the row. Unlink must then re-read
    locked state, discover the mapping no longer belongs to `source`, and
    fail safely — REJECTED, with NO generation bump, NO tombstone, and no
    mutation of any kind."""
    import hashlib
    import secrets
    import threading
    from datetime import datetime, timedelta, timezone

    source = _github_only_user()
    with Session(get_sync_engine()) as session:
        github_id = session.execute(
            select(GithubAccount.github_user_id).where(GithubAccount.user_id == source)
        ).scalar_one()

    raw_secret = secrets.token_urlsafe(32)
    db_telegram_link.create_attempt_sync(
        web_user_id=source,
        link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    target = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    paused = threading.Event()
    release = threading.Event()

    def _pause_after_advisory():
        paused.set()
        assert release.wait(timeout=5), "test never released unlink"

    unlink_outcome = {}

    def _run_unlink():
        unlink_outcome["record"] = capture(lambda: db_telegram_link.unlink_github_sync(
            user_id=source, _test_hook_after_advisory_lock=_pause_after_advisory
        ))

    unlink_thread = threading.Thread(target=_run_unlink)
    unlink_thread.start()
    assert paused.wait(timeout=5), "unlink never reached the advisory lock"

    generation_before = _admission_generation()

    # Concurrent merge completes fully while unlink is paused — moves the
    # mapping from source onto target.
    merge_result = db_telegram_link.redeem_attempt_sync(
        link_secret_hash=hashlib.sha256(raw_secret.encode()).digest(), telegram_user_id=telegram_id
    )
    assert merge_result.outcome == db_telegram_link.RedemptionOutcome.MERGED

    release.set()
    unlink_thread.join(timeout=5)
    assert not unlink_thread.is_alive()

    assert unlink_outcome["record"].exception is None
    assert unlink_outcome["record"].result == db_telegram_link.UnlinkOutcome.REJECTED
    assert _admission_generation() == generation_before  # never bumped
    assert _tombstone_generation(github_id) is None  # never written

    with Session(get_sync_engine()) as session:
        row = session.execute(
            select(GithubAccount).where(GithubAccount.github_user_id == github_id)
        ).scalar_one()
        assert row.user_id == target  # mapping genuinely moved, untouched by the failed unlink
