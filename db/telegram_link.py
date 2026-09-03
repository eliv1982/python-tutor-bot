"""
Server-side Telegram <-> GitHub/web identity-linking persistence (Stage
6C) — SYNC, deliberately (see db/engine.py's module docstring). Wrapped in
asyncio.to_thread() by app/telegram_link.py, the same idiom every other
db/*.py module in this codebase uses for its blocking-I/O boundary.

Corrected lock order (Stage 6C corrective pass, independent-audit MAJOR 2
— mandatory for every function in this module that touches more than one
table, and for db.auth_sessions.create_for_github_sync() in the sibling
module, and for db.github_identity.
resolve_or_create_user_by_github_id_for_oauth_sync() in that sibling
module):

    1. Advisory lock, where one applies to this operation at all:
       - redemption/merge uses `pg_advisory_xact_lock(telegram_user_id)` —
         the EXACT SAME key db.identity.
         resolve_or_create_user_by_telegram_id_sync() already uses, so
         redemption is always serialized against concurrent first-
         creation/resolution of the same Telegram identity.
       - unlink uses `pg_advisory_xact_lock(-github_user_id)` — the EXACT
         SAME (negated) key db.github_identity.
         resolve_or_create_user_by_github_id_sync()/
         resolve_or_create_user_by_github_id_for_oauth_sync() already use,
         so a successful unlink's generation bump + tombstone write (see
         unlink_github_sync() below) can never interleave unsafely with a
         concurrent OAuth resolution for the SAME GitHub identity.
       - attempt creation/replacement takes NO advisory lock — it never
         resolves or mutates a GitHub/Telegram identity, only this user's
         own already-established GitHub mapping.
    2. The relevant `github_accounts` row(s), locked/revalidated, ordered
       by `github_user_id` when more than one is locked in the same
       transaction.
    3. The relevant `users` row(s), locked/revalidated, ordered by `id`
       when more than one is locked in the same transaction.
    4. `web_session_policy` — session-issuance only (db.auth_sessions.
       create_for_github_sync(); never touched by anything in THIS
       module) / `github_oauth_admission` — unlink's generation bump only
       (see unlink_github_sync() below); never touched by attempt
       creation or redemption.
    5. The `telegram_link_attempts` row itself — ALWAYS LAST, and always
       via the single mutating statement that claims/creates/removes it
       (an atomic `DELETE ... RETURNING`, an `INSERT ... ON CONFLICT DO
       UPDATE`, or a plain `DELETE`) — NEVER a separate prior `SELECT ...
       FOR UPDATE` lock step. This is the actual fix for the deadlock this
       module used to contain (see the historical note below): every
       function used to lock/probe the attempt row FIRST, before
       provider/user — for a row that already existed, this genuinely
       locked it first; for a row that did not exist yet, a `SELECT ...
       FOR UPDATE` matching zero rows takes no lock at all, so the
       EFFECTIVE order silently became provider/user-first instead. Two
       concurrent callers landing on opposite sides of that existing-vs-
       missing branch could form a genuine two-resource wait cycle.
       Making the attempt table strictly the LAST resource touched, via
       ONE mutating statement, in EVERY function, removes that
       inconsistency by construction: there is no longer a branch where
       the attempt table is ever locked/touched before `github_accounts`/
       `users`.
    6. Commit.

No function here ever locks/touches a `github_accounts`/`users` row and
THEN waits on a `telegram_link_attempts` row that already exists, because
no function here ever waits on `telegram_link_attempts` at all — its one
mutating statement always runs last and is inherently non-blocking with
respect to itself (an atomic conditional `DELETE`/`INSERT ... ON
CONFLICT`/`DELETE` either matches-and-mutates or matches nothing; a
genuinely concurrent claim of the identical row simply serializes at that
one statement, which can never form a cycle with a resource acquired
BEFORE it). This is what makes attempt-creation, redemption, and unlink
mutually deadlock-free regardless of interleaving — see
tests/test_stage6c_lock_ordering.py for the real-PostgreSQL, real-thread
proof of every required interleaving, including a stress test reproducing
the shape of the old missing-row/existing-row inversion this design
removes.

This module previously wrapped every function here in a bounded retry for
PostgreSQL's own deadlock-detector SQLSTATE (40P01), specifically to
paper over the inconsistency described above. That retry wrapper has been
REMOVED entirely (Stage 6C corrective pass, independent-audit MAJOR 2):
the corrected lock order above removes the deadlock class it existed to
mitigate, so a bounded retry is no longer a substitute for correct
ordering — it is simply unnecessary.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import (
    GITHUB_OAUTH_ADMISSION_ID,
    Document,
    GithubAccount,
    GithubOAuthAdmission,
    GithubUnlinkTombstone,
    TelegramAccount,
    TelegramLinkAttempt,
    User,
    UserPreference,
    WebSession,
)


class CreateAttemptOutcome(Enum):
    """Returned by create_attempt_sync() — never raised as an exception:
    "no current GitHub mapping" is an ordinary, expected outcome (the
    caller's GitHub account may have been unlinked concurrently), not a
    persistence error."""

    CREATED = "created"
    NO_GITHUB_MAPPING = "no_github_mapping"


class RedemptionOutcome(Enum):
    """Every deterministic outcome redeem_attempt_sync() can reach — see
    that function's own docstring for exactly which condition produces
    each one. Every REJECTED_* value is presented identically (the same
    generic failure message) to the Telegram user by app/telegram_link.py
    — the distinct values exist for internal logging/testing only, never
    surfaced verbatim to an end user."""

    MERGED = "merged"
    ALREADY_LINKED = "already_linked"
    INVALID_OR_EXPIRED = "invalid_or_expired"
    REJECTED_TARGET_ALREADY_LINKED_ELSEWHERE = "rejected_target_already_linked_elsewhere"
    REJECTED_SOURCE_ALREADY_LINKED_ELSEWHERE = "rejected_source_already_linked_elsewhere"
    REJECTED_AMBIGUOUS_MERGE = "rejected_ambiguous_merge"
    REJECTED_SOURCE_MAPPING_GONE = "rejected_source_mapping_gone"


@dataclass(frozen=True)
class RedemptionResult:
    """`target_user_id` is populated only for MERGED/ALREADY_LINKED (the
    two outcomes where a caller might legitimately want to know which
    canonical user the Telegram sender now resolves to); None for every
    other outcome. Never carries the raw secret or the source user id —
    see this module's own docstring / Section F of the Stage 6C spec for
    the secret-boundary rule this respects."""

    outcome: RedemptionOutcome
    target_user_id: Optional[uuid.UUID] = None


class UnlinkOutcome(Enum):
    """REJECTED covers BOTH "nothing to unlink" (no current GitHub mapping)
    and "data-bearing account without a Telegram identity" alike — see
    unlink_github_sync()'s own docstring for why these are deliberately
    indistinguishable to a caller."""

    TELEGRAM_KEPT = "telegram_kept"
    USER_DELETED = "user_deleted"
    REJECTED = "rejected"


def create_attempt_sync(
    *,
    web_user_id: uuid.UUID,
    link_secret_hash: bytes,
    expires_at: datetime,
    _test_hook_after_provider_lock: Optional[Callable[[], None]] = None,
    _test_hook_after_user_lock: Optional[Callable[[], None]] = None,
) -> CreateAttemptOutcome:
    """
    Create (or atomically supersede) `web_user_id`'s one outstanding link
    attempt. Called by app/telegram_link.py's start_link() AFTER a separate,
    prior, short cleanup transaction (see cleanup_expired_attempts_sync()
    below — never in the same transaction as this function's locks).

    `expires_at` is computed by the caller (app/telegram_link.py, from
    telegram_link_config.LINK_ATTEMPT_TTL_SECONDS) — mirrors
    db.auth_sessions.create_sync()'s own "the caller resolves TTL policy,
    this function performs none of its own" split of concerns.

    Lock order (module docstring positions 2/3/5): this user's
    `github_accounts` row -> this user's `users` row -> the attempt-table
    mutation itself -> commit. No separate lock step for the attempt row —
    see this module's own docstring for why that used to be a source of
    deadlock.

    The replacement write itself is `INSERT ... ON CONFLICT (web_user_id)
    DO UPDATE` (never a separate DELETE-then-INSERT): concurrent callers
    for the SAME `web_user_id` are already fully serialized by the
    `github_accounts` row lock above (the same single row, taken first, by
    every one of them) before any of them ever reaches this statement, so
    by the time it runs there is at most one in-flight writer for this
    exact key — a genuinely different, unrelated caller could still race
    this exact statement only if it bypassed the provider-row lock
    entirely, which no caller in this module does. `ON CONFLICT` remains
    the correct choice regardless (it is simply never contended in
    practice here): it is what safely handles "an attempt row from an
    earlier, already-completed call still exists" without a separate
    existence check.

    Returns NO_GITHUB_MAPPING (no row written, transaction rolled back) if
    `web_user_id`'s GitHub mapping is missing at the moment the provider
    row lock is acquired — Section G's "still-current GitHub mapping
    required and revalidated under lock".
    """
    with Session(get_sync_engine()) as session:
        github_row = session.execute(
            select(GithubAccount.github_user_id).where(GithubAccount.user_id == web_user_id).with_for_update()
        ).first()
        if _test_hook_after_provider_lock is not None:
            _test_hook_after_provider_lock()
        if github_row is None:
            session.rollback()
            return CreateAttemptOutcome.NO_GITHUB_MAPPING

        session.execute(select(User.id).where(User.id == web_user_id).with_for_update()).scalar_one()
        if _test_hook_after_user_lock is not None:
            _test_hook_after_user_lock()

        stmt = pg_insert(TelegramLinkAttempt).values(
            web_user_id=web_user_id, link_secret_hash=link_secret_hash, expires_at=expires_at
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[TelegramLinkAttempt.web_user_id],
            set_={"link_secret_hash": link_secret_hash, "expires_at": expires_at, "created_at": func.now()},
        )
        session.execute(stmt)
        session.commit()
        return CreateAttemptOutcome.CREATED


def cleanup_expired_attempts_sync(*, batch_size: int = 200) -> int:
    """
    Bounded, best-effort cleanup of expired `telegram_link_attempts` rows —
    run in its OWN short transaction, strictly BEFORE (never alongside) the
    main attempt-creation transaction above (Section D). Deliberately never
    touches `github_accounts`/`users`, and therefore never participates in
    this module's provider/user lock graph at all — the only way to
    guarantee cleanup can never itself become a deadlock party to
    create_attempt_sync()/redeem_attempt_sync()/unlink_github_sync().

    `FOR UPDATE SKIP LOCKED` + a bounded batch + deterministic ordering
    (`ORDER BY web_user_id`): a row a concurrent redemption is actively
    claiming is silently skipped this pass (never blocked on) — a skipped-
    but-still-expired row is simply picked up by the next cleanup pass.
    This is what keeps cleanup non-blocking with respect to every other
    operation in this module, and bounded (never an unbounded sweep) with
    respect to table size.

    Returns the number of rows deleted (0 is a normal, common result).
    Never raises for "nothing to clean up"; a genuine database error here
    is caught by the caller (app/telegram_link.py's start_link()) and never
    allowed to block or corrupt the main creation attempt that follows it
    (Section D: "cleanup failure must not expose a bearer or corrupt the
    main operation").
    """
    with Session(get_sync_engine()) as session:
        expired_ids: List[uuid.UUID] = list(
            session.execute(
                select(TelegramLinkAttempt.web_user_id)
                .where(TelegramLinkAttempt.expires_at <= func.now())
                .order_by(TelegramLinkAttempt.web_user_id)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            ).scalars()
        )
        if not expired_ids:
            session.commit()
            return 0
        session.execute(delete(TelegramLinkAttempt).where(TelegramLinkAttempt.web_user_id.in_(expired_ids)))
        session.commit()
        return len(expired_ids)


def redeem_attempt_sync(
    *,
    link_secret_hash: bytes,
    telegram_user_id: int,
    _test_hook_after_advisory_lock: Optional[Callable[[], None]] = None,
    _test_hook_after_provider_lock: Optional[Callable[[], None]] = None,
    _test_hook_after_user_lock: Optional[Callable[[], None]] = None,
    _test_hook_after_claim: Optional[Callable[[], None]] = None,
) -> RedemptionResult:
    """
    Atomically redeem one link attempt for the sender of a validated
    `/start link_<secret>` message. Called by app/telegram_link.py's
    redeem_link() AFTER handlers/start.py has already resolved/created
    `telegram_user_id`'s canonical UUID via db.identity.
    resolve_or_create_user_by_telegram_id_sync() (a separate, already-
    committed prior transaction under the SAME advisory-lock key) — see
    this function's own step 3 below for why that ordering matters.

    Lock order (module docstring positions 1/2/3/5):
      1. `pg_advisory_xact_lock(telegram_user_id)` — the exact key
         db.identity's own Telegram resolution uses, so this transaction
         can never interleave unsafely with a concurrent first-creation of
         the same Telegram identity.
      2. A plain, non-locking read of the not-yet-expired attempt matching
         `link_secret_hash`, to learn the CANDIDATE source user id — never
         mutated or locked here; this is only a "what would I be claiming"
         probe. INVALID_OR_EXPIRED (nothing else touched) if nothing
         matches — an unknown, already-claimed, superseded, or genuinely
         expired secret are all indistinguishable by design, mirroring
         db.auth_sessions.get_active_sync()'s own "fail-closed lookup"
         philosophy.
      3. Resolve the target Telegram UUID (`telegram_accounts.user_id`)
         under the advisory lock — safe to trust as current precisely
         because that lock is the same one db.identity's resolve/create
         path takes, and the caller already guaranteed a row exists before
         calling this function.
      4. Fast path: if the candidate source equals the target, the claim
         is attempted directly (no provider/user rows to lock — there is
         nothing to merge) and resolves to ALREADY_LINKED or
         INVALID_OR_EXPIRED depending on whether the claim still matches.
      5. Otherwise (a genuine merge candidate): lock BOTH source's and
         target's `github_accounts` rows together, ordered by
         `github_user_id`, then BOTH `users` rows together, ordered by
         `id` (module docstring positions 2/3) — deterministic ordering
         across every function in this module that ever locks more than
         one row of either table in one transaction.
      6. ONLY NOW attempt the atomic claim: `DELETE ... WHERE
         link_secret_hash = :hash AND web_user_id = :candidate_source AND
         expires_at > now() ... RETURNING web_user_id` — requires BOTH the
         exact digest AND the exact candidate learned in step 2, so a
         replacement (a NEW create_attempt_sync() call for the same user,
         which changes `link_secret_hash`), an unlink (which deletes the
         row outright), a genuine expiry, or another concurrent claimant
         racing the identical secret all make this match zero rows —
         INVALID_OR_EXPIRED, nothing further mutated.
      7. Re-read every remaining decision from the rows LOCKED in step 5
         (never from any value read before step 5) and resolve to exactly
         one outcome — see RedemptionOutcome's own docstring for what each
         one means.

    MERGED mutations (Section I): move the source's `github_accounts` row
    onto the target UUID; delete every source `web_sessions` row; delete
    every remaining source `telegram_link_attempts` row (normally none —
    already consumed by step 6's claim; a defensive no-op DELETE); delete
    the now-empty source `users` row. Never touches Qdrant, document
    ownership, or preference ownership (Section I/N) — nothing in this
    function imports rag/* or touches `documents`/`user_preferences` except
    to CHECK (never write) them in the REJECTED_AMBIGUOUS_MERGE gate below.

    Every deterministic outcome (MERGED, ALREADY_LINKED, every REJECTED_*)
    commits — the attempt claim from step 6 is consumed either way (Section
    J). Only an unexpected/transient failure (an uncaught exception) rolls
    the whole transaction — claim included — back, via this Session's
    ordinary context-manager close-on-exception behavior; no domain outcome
    below ever raises.
    """

    with Session(get_sync_engine()) as session:
        session.execute(text("SELECT pg_advisory_xact_lock(CAST(:tid AS bigint))"), {"tid": telegram_user_id})
        if _test_hook_after_advisory_lock is not None:
            _test_hook_after_advisory_lock()

        candidate_source_user_id = session.execute(
            select(TelegramLinkAttempt.web_user_id).where(
                TelegramLinkAttempt.link_secret_hash == link_secret_hash,
                TelegramLinkAttempt.expires_at > func.now(),
            )
        ).scalar_one_or_none()
        if candidate_source_user_id is None:
            session.commit()
            return RedemptionResult(outcome=RedemptionOutcome.INVALID_OR_EXPIRED)

        target_user_id = session.execute(
            select(TelegramAccount.user_id).where(TelegramAccount.telegram_user_id == telegram_user_id)
        ).scalar_one_or_none()
        if target_user_id is None:
            # Unreachable in normal operation — handlers/start.py always
            # resolves/creates this row, under this exact advisory lock,
            # before calling this function. Treated as transient: rolls
            # back (nothing was claimed yet) so the secret remains
            # redeemable.
            raise RuntimeError(
                "telegram_link redemption: no telegram_accounts row for an already-authorized sender"
            )

        if candidate_source_user_id == target_user_id:
            claimed = session.execute(
                delete(TelegramLinkAttempt)
                .where(
                    TelegramLinkAttempt.link_secret_hash == link_secret_hash,
                    TelegramLinkAttempt.web_user_id == candidate_source_user_id,
                    TelegramLinkAttempt.expires_at > func.now(),
                )
                .returning(TelegramLinkAttempt.web_user_id)
            ).first()
            if _test_hook_after_claim is not None:
                _test_hook_after_claim()
            session.commit()
            if claimed is None:
                return RedemptionResult(outcome=RedemptionOutcome.INVALID_OR_EXPIRED)
            return RedemptionResult(outcome=RedemptionOutcome.ALREADY_LINKED, target_user_id=target_user_id)

        github_rows = list(
            session.execute(
                select(GithubAccount)
                .where(GithubAccount.user_id.in_((candidate_source_user_id, target_user_id)))
                .order_by(GithubAccount.github_user_id)
                .with_for_update()
            ).scalars()
        )
        if _test_hook_after_provider_lock is not None:
            _test_hook_after_provider_lock()
        source_github = next((r for r in github_rows if r.user_id == candidate_source_user_id), None)
        target_github = next((r for r in github_rows if r.user_id == target_user_id), None)

        session.execute(
            select(User.id)
            .where(User.id.in_((candidate_source_user_id, target_user_id)))
            .order_by(User.id)
            .with_for_update()
        ).all()
        if _test_hook_after_user_lock is not None:
            _test_hook_after_user_lock()

        claimed = session.execute(
            delete(TelegramLinkAttempt)
            .where(
                TelegramLinkAttempt.link_secret_hash == link_secret_hash,
                TelegramLinkAttempt.web_user_id == candidate_source_user_id,
                TelegramLinkAttempt.expires_at > func.now(),
            )
            .returning(TelegramLinkAttempt.web_user_id)
        ).first()
        if _test_hook_after_claim is not None:
            _test_hook_after_claim()
        if claimed is None:
            session.commit()
            return RedemptionResult(outcome=RedemptionOutcome.INVALID_OR_EXPIRED)
        source_user_id: uuid.UUID = claimed[0]

        if source_github is None:
            session.commit()
            return RedemptionResult(outcome=RedemptionOutcome.REJECTED_SOURCE_MAPPING_GONE)

        source_owns_a_telegram_account = session.execute(
            select(TelegramAccount.telegram_user_id).where(TelegramAccount.user_id == source_user_id).limit(1)
        ).first()
        if source_owns_a_telegram_account is not None:
            session.commit()
            return RedemptionResult(outcome=RedemptionOutcome.REJECTED_SOURCE_ALREADY_LINKED_ELSEWHERE)

        if target_github is not None:
            session.commit()
            return RedemptionResult(outcome=RedemptionOutcome.REJECTED_TARGET_ALREADY_LINKED_ELSEWHERE)

        source_has_documents = session.execute(
            select(Document.id).where(Document.owner_user_id == source_user_id).limit(1)
        ).first()
        source_has_preferences = session.execute(
            select(UserPreference.user_id).where(UserPreference.user_id == source_user_id).limit(1)
        ).first()
        if source_has_documents is not None or source_has_preferences is not None:
            session.commit()
            return RedemptionResult(outcome=RedemptionOutcome.REJECTED_AMBIGUOUS_MERGE)

        session.execute(
            update(GithubAccount)
            .where(GithubAccount.github_user_id == source_github.github_user_id)
            .values(user_id=target_user_id)
        )
        session.execute(delete(WebSession).where(WebSession.user_id == source_user_id))
        session.execute(delete(TelegramLinkAttempt).where(TelegramLinkAttempt.web_user_id == source_user_id))
        session.execute(delete(User).where(User.id == source_user_id))
        session.commit()
        return RedemptionResult(outcome=RedemptionOutcome.MERGED, target_user_id=target_user_id)


def unlink_github_sync(
    *,
    user_id: uuid.UUID,
    _test_hook_after_advisory_lock: Optional[Callable[[], None]] = None,
    _test_hook_after_provider_lock: Optional[Callable[[], None]] = None,
    _test_hook_after_user_lock: Optional[Callable[[], None]] = None,
    _test_hook_after_admission_lock: Optional[Callable[[], None]] = None,
) -> UnlinkOutcome:
    """
    Removes `user_id`'s GitHub mapping (Section L), and — for every
    SUCCESSFUL unlink — atomically establishes finality via the durable
    generation/tombstone protocol (Stage 6C corrective pass, independent-
    audit MAJOR 1): a NEWER OAuth login (captured generation strictly
    greater than the one this unlink writes) may recreate a mapping for
    this same GitHub identity, but any OLDER, already-in-flight OAuth
    callback for it — even one that had already authenticated with
    GitHub before this unlink ran — must be rejected, never resurrecting
    access. See db.github_identity.
    resolve_or_create_user_by_github_id_for_oauth_sync() for the other
    half of this protocol.

    Because this function initially knows only the canonical `user_id`, it
    first performs an UNLOCKED, non-blocking probe of `github_accounts` to
    learn the CANDIDATE `github_user_id` to unlink — never trusted as
    final. It then acquires the per-GitHub advisory lock
    (`pg_advisory_xact_lock(-github_user_id)`, the exact same key
    db.github_identity's resolution functions use) keyed on that
    candidate, locks/REVALIDATES the actual `github_accounts` row under
    that lock, and fails safely (REJECTED, no mutation at all) if the row
    is gone or its `user_id` no longer equals the caller's `user_id` — a
    concurrent redemption/merge could have moved this exact GitHub mapping
    onto a different canonical user while this call was waiting for the
    lock, and this function must never unlink or tombstone a GitHub
    identity that no longer belongs to the authenticated caller.

    Lock order (module docstring positions 1/2/3/4/5): per-GitHub advisory
    lock -> `github_accounts` row (locked + revalidated) -> `users` row ->
    (success path only) `github_oauth_admission` singleton -> the attempt-
    table mutation itself -> commit.

    Three outcomes:
      - TELEGRAM_KEPT: `user_id` has a `telegram_accounts` row — the
        canonical user and every Telegram-owned row (documents,
        preferences, the Telegram mapping itself) are retained; only the
        GitHub mapping is removed and every active web session is
        REVOKED (not deleted — the user row survives, so there is nothing
        forcing session rows to be removed, and revoking rather than
        deleting preserves the same audit-row convention
        db.auth_sessions.revoke_sync() already uses for logout).
      - USER_DELETED: no `telegram_accounts` row AND no domain data
        (`documents`/`user_preferences`) — the canonical user is a bare
        GitHub-only identity with nothing left to strand, so it is deleted
        outright. Web sessions are DELETED here (not merely revoked) —
        required, not stylistic: the `users` row this function is about to
        delete still has an active FK from `web_sessions.user_id`, and this
        schema's FKs are NOT cascading (see db/models.py's
        TelegramLinkAttempt docstring on why the new FK in this stage is
        explicit `ON DELETE RESTRICT`, and every other FK in this schema
        defaults to the equivalent NO ACTION) — a mere revoke would leave a
        referencing row behind and the final `DELETE FROM users` would
        fail closed with an IntegrityError instead of succeeding.
      - REJECTED: covers "no current GitHub mapping to unlink" (probe
        found nothing, or the recheck under lock found the mapping gone or
        moved to someone else), AND "no Telegram account but unexpected
        domain data" alike, and performs NO mutation whatsoever in any of
        these cases — no generation bump, no tombstone write, no attempt/
        session/mapping/user change (Section L/E: "reject atomically;
        preserve mapping, sessions, attempts, user, and data; do not
        strand inaccessible content") — deliberately a single,
        indistinguishable outcome for all of them, so this function's
        return value alone never reveals which condition applied to a
        caller.
    """
    with Session(get_sync_engine()) as session:
        candidate_github_user_id = session.execute(
            select(GithubAccount.github_user_id).where(GithubAccount.user_id == user_id)
        ).scalar_one_or_none()
        if candidate_github_user_id is None:
            session.rollback()
            return UnlinkOutcome.REJECTED

        session.execute(
            text("SELECT pg_advisory_xact_lock(CAST(:key AS bigint))"), {"key": -candidate_github_user_id}
        )
        if _test_hook_after_advisory_lock is not None:
            _test_hook_after_advisory_lock()

        github_row = session.execute(
            select(GithubAccount).where(GithubAccount.github_user_id == candidate_github_user_id).with_for_update()
        ).scalar_one_or_none()
        if _test_hook_after_provider_lock is not None:
            _test_hook_after_provider_lock()
        if github_row is None or github_row.user_id != user_id:
            # Vanished, or moved to a different canonical user while this
            # call waited for the lock — never unlink/tombstone a mapping
            # that no longer belongs to the authenticated caller.
            session.rollback()
            return UnlinkOutcome.REJECTED

        session.execute(select(User.id).where(User.id == user_id).with_for_update()).scalar_one()
        if _test_hook_after_user_lock is not None:
            _test_hook_after_user_lock()

        has_telegram = session.execute(
            select(TelegramAccount.telegram_user_id).where(TelegramAccount.user_id == user_id).limit(1)
        ).first()

        if has_telegram is None:
            has_documents = session.execute(
                select(Document.id).where(Document.owner_user_id == user_id).limit(1)
            ).first()
            has_preferences = session.execute(
                select(UserPreference.user_id).where(UserPreference.user_id == user_id).limit(1)
            ).first()
            if has_documents is not None or has_preferences is not None:
                session.rollback()
                return UnlinkOutcome.REJECTED

        # Successful unlink from here on — atomically establish finality
        # (generation bump + tombstone) before performing the mutation
        # itself, all in this same transaction/commit.
        current_generation = session.execute(
            select(GithubOAuthAdmission.unlink_generation)
            .where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
            .with_for_update()
        ).scalar_one()
        if _test_hook_after_admission_lock is not None:
            _test_hook_after_admission_lock()
        new_generation = current_generation + 1
        session.execute(
            update(GithubOAuthAdmission)
            .where(GithubOAuthAdmission.id == GITHUB_OAUTH_ADMISSION_ID)
            .values(unlink_generation=new_generation)
        )
        tombstone_stmt = pg_insert(GithubUnlinkTombstone).values(
            github_user_id=candidate_github_user_id, unlink_generation=new_generation
        )
        tombstone_stmt = tombstone_stmt.on_conflict_do_update(
            index_elements=[GithubUnlinkTombstone.github_user_id],
            set_={"unlink_generation": new_generation, "unlinked_at": func.now()},
        )
        session.execute(tombstone_stmt)

        session.execute(delete(TelegramLinkAttempt).where(TelegramLinkAttempt.web_user_id == user_id))

        if has_telegram is not None:
            session.execute(
                update(WebSession)
                .where(WebSession.user_id == user_id, WebSession.revoked_at.is_(None))
                .values(revoked_at=func.now())
            )
            session.execute(
                delete(GithubAccount).where(GithubAccount.github_user_id == candidate_github_user_id)
            )
            session.commit()
            return UnlinkOutcome.TELEGRAM_KEPT

        session.execute(delete(WebSession).where(WebSession.user_id == user_id))
        session.execute(delete(GithubAccount).where(GithubAccount.github_user_id == candidate_github_user_id))
        session.execute(delete(User).where(User.id == user_id))
        session.commit()
        return UnlinkOutcome.USER_DELETED
