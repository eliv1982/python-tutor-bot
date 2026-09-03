"""
Stage 6C regression tests: db.auth_sessions.create_for_github_sync() (Section
K) — GitHub-backed session issuance racing against db.telegram_link.
unlink_github_sync()/redeem_attempt_sync(), both orderings, sequential and
genuinely concurrent (real threads, real PostgreSQL row-lock contention).
Real disposable PostgreSQL via tests/conftest.py's postgres_db.
"""

import random
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

import db.auth_sessions as db_auth_sessions
import db.github_identity as db_github_identity
import db.identity as db_identity
import db.telegram_link as db_telegram_link
from concurrency_helpers import capture, wait_until_blocked_on
from db.engine import get_sync_engine
from db.models import GithubAccount, WebSession


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


def _github_only_user():
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    user_id = db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)
    return github_id, user_id


def _future_expiry():
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _hash(raw: str) -> bytes:
    import hashlib

    return hashlib.sha256(raw.encode()).digest()


def _create_and_redeem_merge(source_user_id) -> uuid.UUID:
    raw_secret = secrets.token_urlsafe(32)
    db_telegram_link.create_attempt_sync(
        web_user_id=source_user_id, link_secret_hash=_hash(raw_secret), expires_at=_future_expiry()
    )
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    result = db_telegram_link.redeem_attempt_sync(link_secret_hash=_hash(raw_secret), telegram_user_id=telegram_id)
    assert result.outcome == db_telegram_link.RedemptionOutcome.MERGED
    return result.target_user_id


def _issue(github_id: int):
    raw_token = secrets.token_urlsafe(32)
    user_id = db_auth_sessions.create_for_github_sync(
        github_user_id=github_id, token_hash=_hash(raw_token), issued_secure=True, expires_at=_future_expiry()
    )
    return raw_token, user_id


# ---------------------------------------------------------------------------
# A. Sequential orderings (Section K's four required race outcomes)
# ---------------------------------------------------------------------------


def test_issuance_before_unlink_unlink_later_revokes_it(postgres_db):
    github_id, user_id = _github_only_user()
    raw_token, issued_user_id = _issue(github_id)
    assert issued_user_id == user_id
    assert db_auth_sessions.get_active_sync(token_hash=_hash(raw_token), expected_secure=True) is not None

    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.USER_DELETED

    assert db_auth_sessions.get_active_sync(token_hash=_hash(raw_token), expected_secure=True) is None


def test_unlink_before_issuance_no_session_is_created(postgres_db):
    github_id, user_id = _github_only_user()
    outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
    assert outcome == db_telegram_link.UnlinkOutcome.USER_DELETED

    raw_token, issued_user_id = _issue(github_id)
    assert issued_user_id is None
    assert db_auth_sessions.get_active_sync(token_hash=_hash(raw_token), expected_secure=True) is None


def test_issuance_before_merge_merge_deletes_the_source_session(postgres_db):
    github_id, source_user_id = _github_only_user()
    raw_token, issued_user_id = _issue(github_id)
    assert issued_user_id == source_user_id

    target = _create_and_redeem_merge(source_user_id)

    assert db_auth_sessions.get_active_sync(token_hash=_hash(raw_token), expected_secure=True) is None
    engine = get_sync_engine()
    with Session(engine) as session:
        assert session.execute(select(WebSession).where(WebSession.user_id == target)).first() is None


def test_merge_before_issuance_fresh_session_issued_for_surviving_telegram_uuid(postgres_db):
    github_id, source_user_id = _github_only_user()
    target = _create_and_redeem_merge(source_user_id)

    raw_token, issued_user_id = _issue(github_id)
    assert issued_user_id == target
    record = db_auth_sessions.get_active_sync(token_hash=_hash(raw_token), expected_secure=True)
    assert record is not None
    assert record.user_id == target


# ---------------------------------------------------------------------------
# B. Genuine concurrent contention (real threads, real PostgreSQL row lock)
# ---------------------------------------------------------------------------


def test_concurrent_issuance_and_unlink_never_deadlock_and_resolve_consistently(postgres_db):
    """Real-thread proof: issuance holds the github_accounts row lock
    first (via a test hook), unlink is confirmed GENUINELY blocked on that
    same row (pg_stat_activity), then issuance is released — the session
    must complete first, and unlink (unblocked next) must still correctly
    observe/act on the fully-committed state (revoking the just-created
    session, since this github-only user has no Telegram identity to
    preserve)."""
    github_id, user_id = _github_only_user()

    issuer_holds_lock = threading.Event()
    release_issuer = threading.Event()

    def _pause_issuer():
        issuer_holds_lock.set()
        assert release_issuer.wait(timeout=5), "test never released the issuer"

    issuer_outcome = {}
    raw_token = secrets.token_urlsafe(32)

    def _run_issuer():
        issuer_outcome["record"] = capture(lambda: db_auth_sessions.create_for_github_sync(
            github_user_id=github_id,
            token_hash=_hash(raw_token),
            issued_secure=True,
            expires_at=_future_expiry(),
            _test_hook_after_lock=_pause_issuer,
        ))

    issuer_thread = threading.Thread(target=_run_issuer)
    issuer_thread.start()
    assert issuer_holds_lock.wait(timeout=5), "issuer never reached the provider-row lock"

    unlink_outcome = {}

    def _run_unlink():
        unlink_outcome["record"] = capture(lambda: db_telegram_link.unlink_github_sync(user_id=user_id))

    unlink_thread = threading.Thread(target=_run_unlink)
    unlink_thread.start()

    assert wait_until_blocked_on(table_substring="github_accounts"), (
        "unlink never showed up as genuinely blocked on the provider-row lock"
    )

    release_issuer.set()
    issuer_thread.join(timeout=5)
    unlink_thread.join(timeout=5)
    assert not issuer_thread.is_alive() and not unlink_thread.is_alive()

    assert issuer_outcome["record"].exception is None
    assert unlink_outcome["record"].exception is None
    assert issuer_outcome["record"].result == user_id
    assert unlink_outcome["record"].result == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert db_auth_sessions.get_active_sync(token_hash=_hash(raw_token), expected_secure=True) is None
