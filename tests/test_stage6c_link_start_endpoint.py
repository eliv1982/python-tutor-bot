"""
Stage 6C regression tests: POST /api/link/telegram/start — auth, CSRF,
Cache-Control: no-store, the generic 503-when-unavailable posture (Section
H), the generic 409-when-no-github-mapping posture, and attempt-replacement
semantics (Section G). Real disposable PostgreSQL via tests/conftest.py's
postgres_db, exactly mirroring tests/test_stage6a_fastapi_app.py's own
TestClient conventions.
"""

import random
import uuid

import pytest
from starlette.testclient import TestClient

import app.auth_session as auth_session
import db.auth_sessions as db_auth_sessions
import db.github_identity as db_github_identity
import telegram_link_config
import web_config
from concurrency_helpers import assert_all_terminated, assert_no_exceptions, run_workers
from web.app import create_app
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME

START_PATH = "/api/link/telegram/start"


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


@pytest.fixture(autouse=True)
def _insecure_posture_for_testing(monkeypatch, postgres_db):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)
    db_auth_sessions.apply_startup_posture_sync(requested_secure=False)
    yield


@pytest.fixture(autouse=True)
def _configured_bot_username(monkeypatch):
    monkeypatch.setattr(telegram_link_config, "TELEGRAM_BOT_USERNAME", "my_tutor_bot")


def _real_github_user() -> uuid.UUID:
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


# --- authentication / CSRF ---------------------------------------------------


def test_no_session_is_unauthenticated():
    client = TestClient(create_app())
    response = client.post(START_PATH, headers={CSRF_HEADER_NAME: "irrelevant"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_valid_session_without_csrf_header_is_rejected():
    user_id = _real_github_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    response = client.post(START_PATH)

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_valid_session_and_csrf_succeeds():
    user_id = _real_github_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    response = client.post(START_PATH, headers=_csrf_headers(issued))

    assert response.status_code == 200
    body = response.json()
    assert body["deep_link"].startswith("https://t.me/my_tutor_bot?start=link_")
    assert "expires_at" in body


# --- Cache-Control: no-store -------------------------------------------------


@pytest.mark.asyncio
async def test_success_response_carries_no_store_header():
    user_id = _real_github_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    response = client.post(START_PATH, headers=_csrf_headers(issued))

    assert response.headers.get("cache-control") == "no-store"


# --- generic 503 when linking is unavailable (Section H) --------------------


@pytest.mark.asyncio
async def test_missing_bot_username_returns_generic_503(monkeypatch):
    monkeypatch.setattr(telegram_link_config, "TELEGRAM_BOT_USERNAME", None)
    user_id = _real_github_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    response = client.post(START_PATH, headers=_csrf_headers(issued))

    assert response.status_code == 503
    assert "my_tutor_bot" not in response.text


# --- generic 409 when there is no current GitHub mapping --------------------


@pytest.mark.asyncio
async def test_no_current_github_mapping_returns_generic_409():
    import db.identity as db_identity

    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    user_id = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    response = client.post(START_PATH, headers=_csrf_headers(issued))

    assert response.status_code == 409


# --- attempt-replacement semantics (Section G) -------------------------------


@pytest.mark.asyncio
async def test_repeated_requests_supersede_the_previous_attempt_not_accumulate():
    from db.engine import get_sync_engine
    from db.models import TelegramLinkAttempt
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    user_id = _real_github_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    first = client.post(START_PATH, headers=_csrf_headers(issued))
    second = client.post(START_PATH, headers=_csrf_headers(issued))

    assert first.status_code == 200 and second.status_code == 200
    # A bare `assert first_deep_link != second_deep_link` would print BOTH
    # live raw-secret-bearing deep links on failure (pytest's own
    # assertion-rewriting introspection) — compute the boolean first and
    # fail with a fixed, secret-free message instead (Stage 6C corrective
    # pass, independent-audit MINOR 2).
    if first.json()["deep_link"] == second.json()["deep_link"]:
        pytest.fail(
            "two consecutive link-start requests produced identical deep links (secret collision)",
            pytrace=False,
        )

    with Session(get_sync_engine()) as session:
        count = session.execute(
            select(func.count()).select_from(TelegramLinkAttempt).where(TelegramLinkAttempt.web_user_id == user_id)
        ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_superseded_secret_no_longer_redeems():
    """The FIRST issued secret must stop working once a second start
    request supersedes it — proven end to end via app.telegram_link."""
    import app.telegram_link as telegram_link

    user_id = _real_github_user()
    issued = await _session_for(user_id)
    client = _authed_client(issued)

    first = client.post(START_PATH, headers=_csrf_headers(issued))
    first_secret = first.json()["deep_link"].rsplit("link_", 1)[1]
    client.post(START_PATH, headers=_csrf_headers(issued))

    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    import db.identity as db_identity
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    outcome = await telegram_link.redeem_link(telegram_user_id=telegram_id, raw_secret=first_secret)
    assert outcome == telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED


@pytest.mark.asyncio
async def test_repeated_requests_cannot_create_unbounded_rows_under_concurrency():
    """Section G: "repeated requests cannot create unbounded rows" — fires
    six GENUINELY simultaneous starts (a shared threading.Barrier, never
    launch order — concurrency_helpers.run_workers) for the SAME user, and
    proves exactly one row survives regardless of interleaving. Every
    worker's own result-or-exception is captured exactly once under
    synchronization, every thread is joined with a bounded timeout and
    confirmed no longer alive, the exact six-worker id set and terminal-
    record count are asserted explicitly, and — beyond "exactly one row
    exists" — the FINAL persisted digest is proven to be the only one of
    the six that still redeems, with every other returned secret already
    superseded end to end through app.telegram_link.redeem_link() (Stage
    6C corrective pass, independent-audit MAJOR 1). No assertion below
    ever compares a raw secret/deep-link against another value directly —
    every check is either a count/id/enum comparison or a digest-bytes
    match, so a failure here can never print a live bearer secret."""
    import hashlib

    import app.telegram_link as telegram_link
    import db.identity as db_identity
    from db.engine import get_sync_engine
    from db.models import TelegramLinkAttempt
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    user_id = _real_github_user()
    issued = await _session_for(user_id)
    worker_ids = list(range(6))

    def _start(_worker_id):
        client = _authed_client(issued)
        response = client.post(START_PATH, headers=_csrf_headers(issued))
        if response.status_code != 200:
            raise AssertionError(f"unexpected status_code={response.status_code}")
        return response.json()["deep_link"]

    records, threads = run_workers(worker_ids, _start, timeout=15.0)

    assert_all_terminated(threads)
    assert set(records.keys()) == set(worker_ids)
    assert len(records) == 6
    assert_no_exceptions(records)

    secrets_by_worker = {
        worker_id: record.result.rsplit("link_", 1)[1] for worker_id, record in records.items()
    }

    with Session(get_sync_engine()) as session:
        count = session.execute(
            select(TelegramLinkAttempt.link_secret_hash).where(TelegramLinkAttempt.web_user_id == user_id)
        ).all()
    assert len(count) == 1
    persisted_hash = count[0][0]

    matching_workers = [
        worker_id
        for worker_id, secret in secrets_by_worker.items()
        if hashlib.sha256(secret.encode()).digest() == persisted_hash
    ]
    assert len(matching_workers) == 1
    winning_worker = matching_workers[0]

    # Every OTHER worker's secret must already be superseded — proven by
    # actually redeeming each one against a fresh Telegram target (the
    # real, end-to-end redemption contract), never by re-deriving the
    # digest a second time.
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    for worker_id, secret in secrets_by_worker.items():
        if worker_id == winning_worker:
            continue
        outcome = await telegram_link.redeem_link(telegram_user_id=telegram_id, raw_secret=secret)
        assert outcome == telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED

    winning_outcome = await telegram_link.redeem_link(
        telegram_user_id=telegram_id, raw_secret=secrets_by_worker[winning_worker]
    )
    assert winning_outcome == telegram_link.RedemptionOutcome.MERGED
