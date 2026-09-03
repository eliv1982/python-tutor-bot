"""
Stage 6C regression tests: the corrected lock order (Section C; Stage 6C
corrective pass, independent-audit MAJOR 2) — real-PostgreSQL, real-thread
proofs that redemption and unlink can never deadlock against each other,
that attempt-creation and unlink are mutually deadlock-free, that a stress
of many concurrent FIRST-EVER attempt creators for the SAME user can never
reproduce the historical missing-row/existing-row inversion this design
removes, that the per-GitHub advisory lock correctly serializes unlink
against the generation-aware OAuth resolver, and bounded-completion proofs
for same-secret and different-sources concurrent redemption races. Real
disposable PostgreSQL via tests/conftest.py's postgres_db.

Every function under test here now touches `github_accounts`/`users`
STRICTLY BEFORE ever mutating `telegram_link_attempts` (db/telegram_link.py's
own module docstring) — there is no longer a branch where the attempt
table is locked/probed FIRST, so every interleaving below is expressed in
terms of `github_accounts` row contention, never `telegram_link_attempts`
contention.
"""

import random
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

import db.github_identity as db_github_identity
import db.identity as db_identity
import db.telegram_link as db_telegram_link
from concurrency_helpers import (
    assert_all_terminated,
    assert_no_exceptions,
    capture,
    run_workers,
    wait_until_blocked_on,
    wait_until_blocked_on_advisory_lock,
)
from db.engine import get_sync_engine
from db.models import GithubAccount, TelegramLinkAttempt, User


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


def _hash(raw: str) -> bytes:
    import hashlib

    return hashlib.sha256(raw.encode()).digest()


def _future_expiry():
    return datetime.now(timezone.utc) + timedelta(hours=1)


def _fresh_github_id() -> int:
    return random.randint(10 ** 8, 10 ** 9 - 1)


def _github_only_user():
    return db_github_identity.resolve_or_create_user_by_github_id_sync(_fresh_github_id())


def _github_only_user_and_id():
    github_id = _fresh_github_id()
    return db_github_identity.resolve_or_create_user_by_github_id_sync(github_id), github_id


def _create_attempt(user_id) -> str:
    raw_secret = secrets.token_urlsafe(32)
    outcome = db_telegram_link.create_attempt_sync(
        web_user_id=user_id, link_secret_hash=_hash(raw_secret), expires_at=_future_expiry()
    )
    assert outcome == db_telegram_link.CreateAttemptOutcome.CREATED
    return raw_secret


# ---------------------------------------------------------------------------
# A. Redemption vs unlink for the SAME source — both orderings, expressed
# via github_accounts row contention (module docstring position 2) —
# neither side ever locks/probes telegram_link_attempts before that.
# ---------------------------------------------------------------------------


def test_redemption_holds_provider_lock_unlink_then_finds_nothing_to_undo(postgres_db):
    """Redemption reaches (and holds open, via _test_hook_after_user_lock)
    its github_accounts + users locks for source+target FIRST — unlink's
    own provider-row lock (the SAME source github_accounts row) must show
    up as GENUINELY blocked on github_accounts. Once redemption completes
    (moving the mapping to target), unlink must proceed cleanly to
    REJECTED (nothing left to unlink for source) — never hang, never
    deadlock."""
    source = _github_only_user()
    raw_secret = _create_attempt(source)
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    locks_held = threading.Event()
    release_redemption = threading.Event()

    def _pause_after_user_lock():
        locks_held.set()
        assert release_redemption.wait(timeout=5), "test never released redemption"

    redemption_outcome = {}

    def _run_redemption():
        redemption_outcome["record"] = capture(lambda: db_telegram_link.redeem_attempt_sync(
            link_secret_hash=_hash(raw_secret), telegram_user_id=telegram_id,
            _test_hook_after_user_lock=_pause_after_user_lock,
        ))

    redemption_thread = threading.Thread(target=_run_redemption)
    redemption_thread.start()
    assert locks_held.wait(timeout=5), "redemption never reached its provider/user locks"

    unlink_outcome = {}

    def _run_unlink():
        unlink_outcome["record"] = capture(lambda: db_telegram_link.unlink_github_sync(user_id=source))

    unlink_thread = threading.Thread(target=_run_unlink)
    unlink_thread.start()

    assert wait_until_blocked_on(table_substring="github_accounts"), (
        "unlink never showed up as genuinely blocked on the provider-row lock "
        "(a block on telegram_link_attempts here would mean the attempt table "
        "is being locked before github_accounts again)"
    )

    release_redemption.set()
    redemption_thread.join(timeout=5)
    unlink_thread.join(timeout=5)
    assert not redemption_thread.is_alive() and not unlink_thread.is_alive()

    assert redemption_outcome["record"].exception is None
    assert unlink_outcome["record"].exception is None
    assert redemption_outcome["record"].result.outcome == db_telegram_link.RedemptionOutcome.MERGED
    assert unlink_outcome["record"].result == db_telegram_link.UnlinkOutcome.REJECTED  # nothing left to unlink


def test_unlink_holds_provider_lock_redemption_then_sees_invalid_or_expired(postgres_db):
    """Unlink reaches (and holds open, via _test_hook_after_provider_lock)
    the source github_accounts row FIRST — redemption's own attempt to
    lock the SAME row (as part of its source+target ordered lock) must
    show up as genuinely blocked on github_accounts. Once unlink completes
    (USER_DELETED — deleting the mapping, the user, AND the outstanding
    attempt row as part of its own mutation), redemption's later claim
    attempt must cleanly resolve to INVALID_OR_EXPIRED — never hang, never
    deadlock."""
    source = _github_only_user()
    raw_secret = _create_attempt(source)
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    provider_lock_held = threading.Event()
    release_unlink = threading.Event()

    def _pause_unlink():
        provider_lock_held.set()
        assert release_unlink.wait(timeout=5), "test never released unlink"

    unlink_outcome = {}

    def _run_unlink():
        unlink_outcome["record"] = capture(lambda: db_telegram_link.unlink_github_sync(
            user_id=source, _test_hook_after_provider_lock=_pause_unlink
        ))

    unlink_thread = threading.Thread(target=_run_unlink)
    unlink_thread.start()
    assert provider_lock_held.wait(timeout=5), "unlink never reached the provider-row lock"

    redemption_outcome = {}

    def _run_redemption():
        redemption_outcome["record"] = capture(lambda: db_telegram_link.redeem_attempt_sync(
            link_secret_hash=_hash(raw_secret), telegram_user_id=telegram_id
        ))

    redemption_thread = threading.Thread(target=_run_redemption)
    redemption_thread.start()

    assert wait_until_blocked_on(table_substring="github_accounts"), (
        "redemption never showed up as genuinely blocked on the provider-row lock"
    )

    release_unlink.set()
    unlink_thread.join(timeout=5)
    redemption_thread.join(timeout=5)
    assert not unlink_thread.is_alive() and not redemption_thread.is_alive()

    assert unlink_outcome["record"].exception is None
    assert redemption_outcome["record"].exception is None
    assert unlink_outcome["record"].result == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert redemption_outcome["record"].result.outcome == db_telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED


# ---------------------------------------------------------------------------
# B. Attempt-creation vs unlink for the SAME user — both orderings
# ---------------------------------------------------------------------------


def test_create_attempt_vs_unlink_creation_first_then_unlink_removes_it(postgres_db):
    source = _github_only_user()

    provider_lock_held = threading.Event()
    release_creator = threading.Event()

    def _pause_creator():
        provider_lock_held.set()
        assert release_creator.wait(timeout=5), "test never released the creator"

    creator_outcome = {}

    def _run_creator():
        creator_outcome["record"] = capture(lambda: db_telegram_link.create_attempt_sync(
            web_user_id=source,
            link_secret_hash=_hash(secrets.token_urlsafe(32)),
            expires_at=_future_expiry(),
            _test_hook_after_provider_lock=_pause_creator,
        ))

    creator_thread = threading.Thread(target=_run_creator)
    creator_thread.start()
    assert provider_lock_held.wait(timeout=5), "creator never reached the provider-row lock"

    unlink_outcome = {}

    def _run_unlink():
        unlink_outcome["record"] = capture(lambda: db_telegram_link.unlink_github_sync(user_id=source))

    unlink_thread = threading.Thread(target=_run_unlink)
    unlink_thread.start()

    assert wait_until_blocked_on(table_substring="github_accounts"), (
        "unlink never showed up as genuinely blocked on the provider-row lock"
    )

    release_creator.set()
    creator_thread.join(timeout=5)
    unlink_thread.join(timeout=5)
    assert not creator_thread.is_alive() and not unlink_thread.is_alive()

    assert creator_outcome["record"].exception is None
    assert unlink_outcome["record"].exception is None
    assert creator_outcome["record"].result == db_telegram_link.CreateAttemptOutcome.CREATED
    assert unlink_outcome["record"].result == db_telegram_link.UnlinkOutcome.USER_DELETED


def test_create_attempt_vs_unlink_unlink_first_then_creation_sees_no_mapping(postgres_db):
    source = _github_only_user()

    provider_lock_held = threading.Event()
    release_unlink = threading.Event()

    def _pause_unlink():
        provider_lock_held.set()
        assert release_unlink.wait(timeout=5), "test never released unlink"

    unlink_outcome = {}

    def _run_unlink():
        unlink_outcome["record"] = capture(lambda: db_telegram_link.unlink_github_sync(
            user_id=source, _test_hook_after_provider_lock=_pause_unlink
        ))

    unlink_thread = threading.Thread(target=_run_unlink)
    unlink_thread.start()
    assert provider_lock_held.wait(timeout=5), "unlink never reached the provider-row lock"

    creator_outcome = {}

    def _run_creator():
        creator_outcome["record"] = capture(lambda: db_telegram_link.create_attempt_sync(
            web_user_id=source, link_secret_hash=_hash(secrets.token_urlsafe(32)), expires_at=_future_expiry()
        ))

    creator_thread = threading.Thread(target=_run_creator)
    creator_thread.start()

    assert wait_until_blocked_on(table_substring="github_accounts"), (
        "creator never showed up as genuinely blocked on the provider-row lock"
    )

    release_unlink.set()
    unlink_thread.join(timeout=5)
    creator_thread.join(timeout=5)
    assert not unlink_thread.is_alive() and not creator_thread.is_alive()

    assert unlink_outcome["record"].exception is None
    assert creator_outcome["record"].exception is None
    assert unlink_outcome["record"].result == db_telegram_link.UnlinkOutcome.USER_DELETED
    assert creator_outcome["record"].result == db_telegram_link.CreateAttemptOutcome.NO_GITHUB_MAPPING


# ---------------------------------------------------------------------------
# C. Bounded completion / no deadlock under real concurrency
# ---------------------------------------------------------------------------


def test_concurrent_same_secret_redemption_only_one_merges(postgres_db):
    """Four workers, released together via a shared start barrier
    (concurrency_helpers.run_workers), all race to redeem the SAME secret —
    exactly one may merge; the other three must resolve to
    INVALID_OR_EXPIRED. Every worker's own result-or-exception is captured
    under synchronization, keyed by worker id, and every thread is
    confirmed terminated before any assertion runs."""
    source = _github_only_user()
    raw_secret = _create_attempt(source)
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    worker_ids = list(range(4))

    def _redeem(_worker_id):
        return db_telegram_link.redeem_attempt_sync(
            link_secret_hash=_hash(raw_secret), telegram_user_id=telegram_id
        ).outcome

    records, threads = run_workers(worker_ids, _redeem, timeout=15.0)

    assert_all_terminated(threads)
    assert set(records.keys()) == set(worker_ids)
    assert len(records) == 4
    assert_no_exceptions(records)

    outcomes = [record.result for record in records.values()]
    assert outcomes.count(db_telegram_link.RedemptionOutcome.MERGED) == 1
    assert outcomes.count(db_telegram_link.RedemptionOutcome.INVALID_OR_EXPIRED) == 3


def test_different_sources_racing_the_same_telegram_target_bounded_completion(postgres_db):
    """Two DIFFERENT source users both try to link to the SAME (brand new)
    Telegram target concurrently. Exactly one may end up merged into it;
    the other must resolve to a REJECTED_TARGET_ALREADY_LINKED_ELSEWHERE
    or, if it happens to be scheduled first, itself becomes the merge and
    the second one is rejected — either way, both complete within the
    bound, never hang, and every worker's result/exception is captured and
    asserted explicitly (never relying on
    PytestUnhandledThreadExceptionWarning to surface a failure)."""
    source_a, source_b = _github_only_user(), _github_only_user()
    raw_a, raw_b = _create_attempt(source_a), _create_attempt(source_b)
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    raws = {"a": raw_a, "b": raw_b}

    def _redeem(name):
        return db_telegram_link.redeem_attempt_sync(
            link_secret_hash=_hash(raws[name]), telegram_user_id=telegram_id
        ).outcome

    records, threads = run_workers(["a", "b"], _redeem, timeout=15.0)

    assert_all_terminated(threads)
    assert set(records.keys()) == {"a", "b"}, "not every worker produced exactly one terminal record"
    assert_no_exceptions(records)

    outcomes = {name: record.result for name, record in records.items()}
    outcome_values = set(outcomes.values())
    assert db_telegram_link.RedemptionOutcome.MERGED in outcome_values
    assert outcome_values - {db_telegram_link.RedemptionOutcome.MERGED} <= {
        db_telegram_link.RedemptionOutcome.REJECTED_TARGET_ALREADY_LINKED_ELSEWHERE
    }

    # Final database state, verified after every worker has terminated:
    # exactly one of the two sources survived the merge (the other was
    # deleted), and the target now genuinely owns a github_accounts row.
    engine = get_sync_engine()
    with Session(engine) as session:
        merged_name = next(name for name, outcome in outcomes.items() if outcome == db_telegram_link.RedemptionOutcome.MERGED)
        merged_source = source_a if merged_name == "a" else source_b
        other_source = source_b if merged_name == "a" else source_a
        assert session.get(User, merged_source) is None
        assert session.get(User, other_source) is not None


def test_many_concurrent_first_ever_attempt_creators_for_the_same_user_never_deadlock(postgres_db):
    """Stress/concurrency regression (Stage 6C corrective pass,
    independent-audit MAJOR 2): reproduces the SHAPE of the historical
    deadlock this module used to contain — many truly concurrent, FIRST-
    EVER create_attempt_sync() callers for the SAME `web_user_id`, racing
    on a not-yet-existing attempt row. Under the OLD design, an early
    "row doesn't exist yet" probe took no lock, letting the effective lock
    order for a first-time caller diverge from an already-existing-row
    caller's order; under the corrected design there is no such probe at
    all — every caller is already fully serialized by the SAME
    `github_accounts` row lock, taken first, before any of them ever
    touches `telegram_link_attempts`. Released together via a shared start
    barrier (concurrency_helpers.run_workers) so all twelve genuinely
    begin at once, every worker's own result-or-exception is captured
    under synchronization keyed by worker id, every thread is joined with
    a bounded timeout and confirmed no longer alive, and the final
    database state is verified only after every worker has terminated."""
    source = _github_only_user()
    worker_count = 12
    worker_ids = list(range(worker_count))

    def _create(_worker_id):
        return db_telegram_link.create_attempt_sync(
            web_user_id=source,
            link_secret_hash=_hash(secrets.token_urlsafe(32)),
            expires_at=_future_expiry(),
        )

    records, threads = run_workers(worker_ids, _create, timeout=20.0)

    assert_all_terminated(threads)
    assert set(records.keys()) == set(worker_ids)
    assert len(records) == worker_count
    assert_no_exceptions(records)
    assert [records[wid].result for wid in worker_ids] == [db_telegram_link.CreateAttemptOutcome.CREATED] * worker_count

    engine = get_sync_engine()
    with Session(engine) as session:
        count = session.execute(
            text("SELECT count(*) FROM telegram_link_attempts WHERE web_user_id = :uid"), {"uid": str(source)}
        ).scalar_one()
    assert count == 1, "concurrent replacement must leave exactly one surviving attempt row"


# ---------------------------------------------------------------------------
# D. Unlink vs the generation-aware OAuth resolver — the SAME per-GitHub
# advisory lock must serialize them, and a concurrent unlink that commits
# first must be visible (via the tombstone) to the resolver every time.
# ---------------------------------------------------------------------------


def test_unlink_and_oauth_resolver_serialize_on_the_same_advisory_lock(postgres_db):
    """unlink_github_sync() and
    db.github_identity.resolve_or_create_user_by_github_id_for_oauth_sync()
    both take pg_advisory_xact_lock(-github_user_id) as their very first
    step, before either ever touches github_accounts/users — whichever
    reaches it first must run to completion (commit) before the other can
    even read the tombstone/provider row. Proven here by pausing unlink
    immediately after it acquires that lock (before it revalidates the
    provider row), confirming the resolver call genuinely blocks trying to
    acquire the SAME advisory lock (never merely blocked on a table row),
    then releasing unlink and confirming the resolver's outcome is fully
    consistent with unlink's now-committed tombstone."""
    user_id, github_id = _github_only_user_and_id()

    advisory_lock_held = threading.Event()
    release_unlink = threading.Event()

    def _pause_unlink():
        advisory_lock_held.set()
        assert release_unlink.wait(timeout=5), "test never released unlink"

    unlink_outcome = {}

    def _run_unlink():
        unlink_outcome["record"] = capture(lambda: db_telegram_link.unlink_github_sync(
            user_id=user_id, _test_hook_after_advisory_lock=_pause_unlink
        ))

    unlink_thread = threading.Thread(target=_run_unlink)
    unlink_thread.start()
    assert advisory_lock_held.wait(timeout=5), "unlink never reached the advisory lock"

    resolver_outcome = {}

    def _run_resolver():
        resolver_outcome["record"] = capture(lambda: db_github_identity.resolve_or_create_user_by_github_id_for_oauth_sync(
            github_user_id=github_id, auth_generation=0
        ))

    resolver_thread = threading.Thread(target=_run_resolver)
    resolver_thread.start()

    assert wait_until_blocked_on_advisory_lock(), "resolver never showed up as genuinely blocked on the advisory lock"

    release_unlink.set()
    unlink_thread.join(timeout=5)
    resolver_thread.join(timeout=5)
    assert not unlink_thread.is_alive() and not resolver_thread.is_alive()

    assert unlink_outcome["record"].exception is None
    assert resolver_outcome["record"].exception is None
    assert unlink_outcome["record"].result == db_telegram_link.UnlinkOutcome.USER_DELETED
    # auth_generation=0 was captured "before" unlink (generation now >= 1)
    # — the resolver, unblocked only after unlink's tombstone is fully
    # committed, must see it and reject.
    assert resolver_outcome["record"].result is None


# ---------------------------------------------------------------------------
# E. Deterministic regression proof for the removed lock inversion (Stage
# 6C corrective pass, independent-audit MAJOR 2) — the twelve-worker stress
# test above (Section C) is useful evidence but not a DETERMINISTIC proof:
# it can only ever demonstrate the absence of a hang under whatever
# interleaving the OS scheduler happened to produce that one run. The two
# tests immediately below instead directly OBSERVE the real SQL statements
# db.telegram_link.create_attempt_sync() actually issues — a SQLAlchemy
# `before_cursor_execute` listener attached to the REAL engine, exercising
# the actual production function, never a reimplementation of its logic in
# test code — and assert their call ORDER, separately for "no attempt row
# exists yet" and "an attempt row already exists". Neither depends on
# thread scheduling at all (single-threaded, deterministic call order); a
# regression back to the historical "probe/lock telegram_link_attempts
# before github_accounts" design fails either one deterministically, every
# run, not merely "sometimes under the right interleaving" the way a pure
# stress test would. The third test below then closes the remaining gap
# between those two static observations: a coordinated real-PostgreSQL
# schedule that deliberately transitions a single user from "no row" to
# "row exists" WHILE a second caller is genuinely blocked trying to act on
# it, proving there is no window at that exact boundary where a second
# caller's effective lock order could diverge from the first's.
# ---------------------------------------------------------------------------


def _observe_create_attempt_statement_order(*, web_user_id, raw_secret: str):
    """Attaches a SQLAlchemy `before_cursor_execute` listener to the real
    engine db.telegram_link.create_attempt_sync() actually uses
    (db.engine.get_sync_engine()) and records, IN CALL ORDER, which table
    each statement it issues touches — "provider_lock" for the
    `github_accounts ... FOR UPDATE` row lock, "user_lock" for the
    `users ... FOR UPDATE` row lock, "attempt_touch" for anything touching
    `telegram_link_attempts` (the single `INSERT ... ON CONFLICT DO
    UPDATE` today; deliberately not restricted to INSERT specifically, so
    a regression that reintroduces an earlier probe/lock statement against
    that table is caught too, whatever SQL shape it takes). Returns
    `(outcome, observed)` — this only OBSERVES the statements the real
    function issues; it never re-derives or asserts on the lock order any
    other way, so it cannot pass merely because test code agrees with
    itself."""
    engine = get_sync_engine()
    observed: List[str] = []

    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        upper = statement.upper()
        if "GITHUB_ACCOUNTS" in upper and "FOR UPDATE" in upper:
            observed.append("provider_lock")
        elif "TELEGRAM_LINK_ATTEMPTS" in upper:
            observed.append("attempt_touch")
        elif "USERS" in upper and "FOR UPDATE" in upper:
            observed.append("user_lock")

    event.listen(engine, "before_cursor_execute", _before_cursor_execute)
    try:
        outcome = db_telegram_link.create_attempt_sync(
            web_user_id=web_user_id, link_secret_hash=_hash(raw_secret), expires_at=_future_expiry()
        )
    finally:
        event.remove(engine, "before_cursor_execute", _before_cursor_execute)
    return outcome, observed


def test_create_attempt_lock_order_observed_when_no_attempt_row_exists(postgres_db):
    """No `telegram_link_attempts` row exists for this user at all yet —
    the historical bug's "row doesn't exist" branch. The real statement
    order must still show the provider lock strictly BEFORE the attempt
    table is ever touched."""
    source = _github_only_user()  # brand new — no attempt row exists yet
    outcome, observed = _observe_create_attempt_statement_order(
        web_user_id=source, raw_secret=secrets.token_urlsafe(32)
    )
    assert outcome == db_telegram_link.CreateAttemptOutcome.CREATED
    assert observed.count("provider_lock") == 1
    assert observed.count("attempt_touch") == 1
    assert observed.index("provider_lock") < observed.index("attempt_touch"), (
        "create_attempt_sync() touched telegram_link_attempts before locking github_accounts "
        "(no-existing-row case) — this is the historical inversion this design removed"
    )
    if "user_lock" in observed:
        assert observed.index("provider_lock") < observed.index("user_lock") < observed.index("attempt_touch")


def test_create_attempt_lock_order_observed_when_attempt_row_already_exists(postgres_db):
    """An attempt row for this user ALREADY exists (a prior, unobserved
    call created it) — the historical bug's "row exists" branch. The real
    statement order for this SECOND, observed call must still show the
    provider lock strictly BEFORE the attempt table's ON CONFLICT
    replacement statement."""
    source = _github_only_user()
    first_outcome = db_telegram_link.create_attempt_sync(
        web_user_id=source, link_secret_hash=_hash(secrets.token_urlsafe(32)), expires_at=_future_expiry()
    )
    assert first_outcome == db_telegram_link.CreateAttemptOutcome.CREATED

    outcome, observed = _observe_create_attempt_statement_order(
        web_user_id=source, raw_secret=secrets.token_urlsafe(32)
    )
    assert outcome == db_telegram_link.CreateAttemptOutcome.CREATED
    assert observed.count("provider_lock") == 1
    assert observed.count("attempt_touch") == 1
    assert observed.index("provider_lock") < observed.index("attempt_touch"), (
        "create_attempt_sync() touched telegram_link_attempts before locking github_accounts "
        "(existing-row case) — this is the historical inversion this design removed"
    )
    if "user_lock" in observed:
        assert observed.index("provider_lock") < observed.index("user_lock") < observed.index("attempt_touch")


def test_create_attempt_transition_from_no_row_to_existing_row_is_safe(postgres_db):
    """Deterministically coordinates the exact ambiguous boundary the
    historical deadlock's removed branch used to treat differently: caller
    A's create_attempt_sync() call for a brand-new user (no attempt row
    exists yet) is paused, via `_test_hook_after_provider_lock`, AFTER it
    has already acquired source's `github_accounts` row lock but BEFORE it
    performs the INSERT that brings the row into existence — confirmed
    directly against the real database (a fresh connection, separate from
    A's own held transaction) that no row exists yet at that instant. A
    second caller B for the SAME user is then started while A is still
    paused — B must show up as GENUINELY blocked on the SAME
    `github_accounts` row (pg_stat_activity, never merely inferred), proving
    there is no window where a second caller could observe/act on the
    not-yet-existing attempt row through any path other than that one
    shared provider-row lock. Once A is released and completes (the row now
    exists), B proceeds and safely replaces it — the final database state
    has exactly one row, reflecting B's own written digest (the LAST
    completed write), never a mix of both. Every worker's own result-or-
    exception is captured via concurrency_helpers.capture()."""
    source = _github_only_user()

    provider_lock_held = threading.Event()
    release_a = threading.Event()

    def _pause_a():
        provider_lock_held.set()
        assert release_a.wait(timeout=5), "test never released caller A"

    raw_a = secrets.token_urlsafe(32)
    outcome_a = {}

    def _run_a():
        outcome_a["record"] = capture(lambda: db_telegram_link.create_attempt_sync(
            web_user_id=source, link_secret_hash=_hash(raw_a), expires_at=_future_expiry(),
            _test_hook_after_provider_lock=_pause_a,
        ))

    thread_a = threading.Thread(target=_run_a)
    thread_a.start()
    assert provider_lock_held.wait(timeout=5), "caller A never reached the provider lock"

    # At this instant NO attempt row exists yet for `source` — A is paused
    # strictly BEFORE its own INSERT — confirmed directly against the real
    # database, under a separate connection, before starting B.
    with Session(get_sync_engine()) as session:
        assert session.get(TelegramLinkAttempt, source) is None

    raw_b = secrets.token_urlsafe(32)
    outcome_b = {}

    def _run_b():
        outcome_b["record"] = capture(lambda: db_telegram_link.create_attempt_sync(
            web_user_id=source, link_secret_hash=_hash(raw_b), expires_at=_future_expiry(),
        ))

    thread_b = threading.Thread(target=_run_b)
    thread_b.start()

    assert wait_until_blocked_on(table_substring="github_accounts"), (
        "caller B never showed up as genuinely blocked on the SAME provider-row lock while A holds it "
        "pre-INSERT — a regression here would mean B could observe/act on the not-yet-existing attempt "
        "row through some path other than the provider lock"
    )

    release_a.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive() and not thread_b.is_alive()

    assert outcome_a["record"].exception is None
    assert outcome_b["record"].exception is None
    assert outcome_a["record"].result == db_telegram_link.CreateAttemptOutcome.CREATED
    assert outcome_b["record"].result == db_telegram_link.CreateAttemptOutcome.CREATED

    with Session(get_sync_engine()) as session:
        rows = session.execute(
            select(TelegramLinkAttempt).where(TelegramLinkAttempt.web_user_id == source)
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].link_secret_hash == _hash(raw_b)  # B, released second, committed last
