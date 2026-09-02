"""
Stage 6B regression tests: GitHub OAuth transaction persistence
(db.oauth_transactions) against a REAL disposable PostgreSQL container —
see tests/conftest.py's postgres_container()/postgres_db() fixtures.
Proves genuine database behavior (the digest-length CHECK constraint, the
atomic single-use DELETE ... RETURNING claim, real concurrent-thread
replay-safety) — never pretend/mocked behavior.

Stage 6B independent-audit corrective pass #1, MAJOR 2/Section 8: create_sync()
is now admission-controlled (see db/oauth_transactions.py's own docstring)
and claim_sync() DELETEs the row it claims instead of merely marking it
consumed — `consumed_at` no longer exists on this table at all. Generous,
effectively-unlimited admission parameters (`_GENEROUS_*` below) are used
throughout this file so ordinary functional tests never accidentally hit
the admission control itself; tests/test_stage6b_corrective1_oauth_admission.py
is where admission control (the rate window, the outstanding-row hard
cap, and their real-PostgreSQL concurrency proofs) is exercised directly.
"""

import hashlib
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError

import db.oauth_transactions as db_oauth_transactions
from db.engine import get_sync_engine
from db.models import GithubOAuthTransaction

# Effectively unlimited for this file's purposes — every functional test
# here creates at most a handful of transactions, far below either bound.
_GENEROUS_MAX_STARTS_PER_WINDOW = 10_000
_GENEROUS_MAX_OUTSTANDING = 10_000
_GENEROUS_WINDOW_SECONDS = 60


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture — this module
    exercises the REAL db.oauth_transactions functions against
    postgres_db, never any offline fake."""
    yield


def _hash(raw: str) -> bytes:
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _future_expiry(seconds: float = 600) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _past_expiry(seconds: float = 1) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def _create(*, state_hash: bytes, code_verifier: str, expires_at: datetime) -> bool:
    return db_oauth_transactions.create_sync(
        state_hash=state_hash,
        code_verifier=code_verifier,
        expires_at=expires_at,
        max_starts_per_window=_GENEROUS_MAX_STARTS_PER_WINDOW,
        max_outstanding=_GENEROUS_MAX_OUTSTANDING,
        window_seconds=_GENEROUS_WINDOW_SECONDS,
    )


def _new_transaction(*, verifier: str = None, ttl_seconds: float = 600) -> tuple[str, bytes, str]:
    raw_state = secrets.token_urlsafe(32)
    code_verifier = verifier or secrets.token_urlsafe(48)
    state_hash = _hash(raw_state)
    admitted = _create(state_hash=state_hash, code_verifier=code_verifier, expires_at=_future_expiry(ttl_seconds))
    assert admitted is True
    return raw_state, state_hash, code_verifier


# --- create_sync / basic persistence ----------------------------------------


def test_create_sync_persists_a_row_with_the_expected_columns(postgres_db):
    raw_state, state_hash, code_verifier = _new_transaction()

    engine = get_sync_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(GithubOAuthTransaction.code_verifier).where(GithubOAuthTransaction.state_hash == state_hash)
        ).one()

    assert row.code_verifier == code_verifier


def test_create_sync_returns_true_on_admission(postgres_db):
    raw_state = secrets.token_urlsafe(32)
    admitted = _create(state_hash=_hash(raw_state), code_verifier=secrets.token_urlsafe(48), expires_at=_future_expiry())
    assert admitted is True


def test_raw_state_itself_is_never_persisted_anywhere(postgres_db):
    """Only the digest is a column in this table at all — the raw state
    string cannot appear as a value of ANY column, since no column could
    hold it as itself (state_hash is a 32-byte digest, code_verifier is a
    separate independently-generated value)."""
    raw_state, state_hash, code_verifier = _new_transaction()
    assert raw_state != code_verifier

    engine = get_sync_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(GithubOAuthTransaction).where(GithubOAuthTransaction.state_hash == state_hash)
        ).one()
    assert raw_state.encode("utf-8") not in bytes(row.state_hash)
    assert row.code_verifier != raw_state


# --- claim_sync: happy path, single-use, replay -----------------------------


def test_claim_sync_returns_the_original_code_verifier(postgres_db):
    raw_state, state_hash, code_verifier = _new_transaction()

    claimed = db_oauth_transactions.claim_sync(state_hash=state_hash)

    assert claimed == code_verifier


def test_claim_sync_deletes_the_row(postgres_db):
    """Stage 6B corrective pass #1, Section 8: claim no longer marks a row
    consumed — it DELETEs it. The row must be entirely gone afterward."""
    raw_state, state_hash, code_verifier = _new_transaction()
    db_oauth_transactions.claim_sync(state_hash=state_hash)

    engine = get_sync_engine()
    with engine.connect() as conn:
        count = conn.execute(
            select(func.count()).select_from(GithubOAuthTransaction).where(GithubOAuthTransaction.state_hash == state_hash)
        ).scalar_one()
    assert count == 0


def test_claim_sync_is_single_use_a_second_claim_fails_closed(postgres_db):
    raw_state, state_hash, code_verifier = _new_transaction()

    first = db_oauth_transactions.claim_sync(state_hash=state_hash)
    second = db_oauth_transactions.claim_sync(state_hash=state_hash)

    assert first == code_verifier
    assert second is None


def test_claim_sync_on_unknown_state_hash_returns_none(postgres_db):
    unknown_hash = hashlib.sha256(b"never-issued").digest()
    assert db_oauth_transactions.claim_sync(state_hash=unknown_hash) is None


def test_claim_sync_on_expired_transaction_returns_none(postgres_db):
    raw_state = secrets.token_urlsafe(32)
    state_hash = _hash(raw_state)
    assert _create(state_hash=state_hash, code_verifier=secrets.token_urlsafe(48), expires_at=_past_expiry()) is True

    assert db_oauth_transactions.claim_sync(state_hash=state_hash) is None


def test_claim_sync_on_expired_transaction_does_not_delete_it(postgres_db):
    """An expired-but-never-claimed row is left untouched by claim_sync
    itself (the WHERE predicate excludes it, matching zero rows) — it is
    swept up by create_sync()'s own expired-row cleanup instead (proven in
    tests/test_stage6b_corrective1_oauth_admission.py), not by a claim
    attempt."""
    raw_state = secrets.token_urlsafe(32)
    state_hash = _hash(raw_state)
    assert _create(state_hash=state_hash, code_verifier=secrets.token_urlsafe(48), expires_at=_past_expiry()) is True
    db_oauth_transactions.claim_sync(state_hash=state_hash)

    engine = get_sync_engine()
    with engine.connect() as conn:
        count = conn.execute(
            select(func.count()).select_from(GithubOAuthTransaction).where(GithubOAuthTransaction.state_hash == state_hash)
        ).scalar_one()
    assert count == 1


def test_two_different_transactions_do_not_interfere(postgres_db):
    _, hash_a, verifier_a = _new_transaction()
    _, hash_b, verifier_b = _new_transaction()

    assert db_oauth_transactions.claim_sync(state_hash=hash_a) == verifier_a
    assert db_oauth_transactions.claim_sync(state_hash=hash_b) == verifier_b
    # Each remains individually single-use.
    assert db_oauth_transactions.claim_sync(state_hash=hash_a) is None
    assert db_oauth_transactions.claim_sync(state_hash=hash_b) is None


# --- concurrent double-claim: the core replay-safety proof ------------------


def test_concurrent_double_claim_only_one_thread_ever_receives_the_verifier(postgres_db):
    """The required real-PostgreSQL, real-thread proof (Section 6/19 of the
    Stage 6B spec): many concurrent claim attempts against the SAME
    state_hash must yield the verifier to EXACTLY ONE caller, everyone
    else gets None — proving the atomic DELETE ... RETURNING claim is a
    genuine single-use gate under real concurrency, not merely correct in
    a single-threaded call sequence."""
    raw_state, state_hash, code_verifier = _new_transaction()

    results = []
    lock = threading.Lock()
    start_barrier = threading.Barrier(20)

    def _attempt():
        start_barrier.wait(timeout=5)
        outcome = db_oauth_transactions.claim_sync(state_hash=state_hash)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=_attempt) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(not t.is_alive() for t in threads)
    successes = [r for r in results if r is not None]
    assert successes == [code_verifier], f"expected exactly one success, got: {results}"
    assert len(results) == 20


# --- digest-length CHECK constraint (mirrors WebSession's own proof) -------


def test_digest_length_check_constraint_rejects_a_short_state_hash(postgres_db):
    engine = get_sync_engine()
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                insert(GithubOAuthTransaction).values(
                    state_hash=b"too-short",
                    code_verifier="x" * 43,
                    expires_at=_future_expiry(),
                )
            )


def test_digest_length_check_constraint_rejects_a_long_state_hash(postgres_db):
    engine = get_sync_engine()
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                insert(GithubOAuthTransaction).values(
                    state_hash=b"x" * 40,
                    code_verifier="x" * 43,
                    expires_at=_future_expiry(),
                )
            )


def test_digest_length_check_constraint_accepts_exactly_32_bytes(postgres_db):
    engine = get_sync_engine()
    with engine.begin() as conn:
        conn.execute(
            insert(GithubOAuthTransaction).values(
                state_hash=b"y" * 32,
                code_verifier="x" * 43,
                expires_at=_future_expiry(),
            )
        )
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(GithubOAuthTransaction)).scalar_one()
    assert count == 1


def test_duplicate_state_hash_insert_is_rejected_by_the_primary_key(postgres_db):
    raw_state, state_hash, code_verifier = _new_transaction()
    engine = get_sync_engine()
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                insert(GithubOAuthTransaction).values(
                    state_hash=state_hash,
                    code_verifier="a-different-verifier",
                    expires_at=_future_expiry(),
                )
            )


def test_expires_at_index_exists(postgres_db):
    """Section 9: create_sync()'s admission-path cleanup deletes expired
    rows on every call — proves the supporting index actually exists
    (rather than only asserting cleanup behavior indirectly)."""
    from sqlalchemy import text

    engine = get_sync_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'github_oauth_transactions'")
        ).all()
    index_names = {row[0] for row in rows}
    assert "ix_github_oauth_transactions_expires_at" in index_names
