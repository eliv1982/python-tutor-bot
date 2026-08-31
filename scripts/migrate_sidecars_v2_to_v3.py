#!/usr/bin/env python
"""
Operator-facing one-time migration: rewrite v2 (Telegram-integer-owned)
managed-upload sidecars to v3 (canonical-UUID-owned) in place.

Stage 5C replaces Telegram numeric id with the internal user UUID as
canonical RAG ownership identity. A v2 sidecar's `owner_user_id` (Telegram
int) is no longer a valid ownership identity for scripts/rebuild_qdrant.py's
direct rebuild planning — it fail-closed skips v2/v1 sidecars entirely (see
its own `_plan_upload_documents()`). This script is the explicit, one-time
bridge: for each valid v2 sidecar, resolve its Telegram owner id to the
real internal user UUID (creating the mapping if this is the very first
time that Telegram id has ever been resolved — legitimate here, since this
is migrating already-existing, already-owned data, never a live Telegram
authorization decision) and rewrite the sidecar in place as v3, preserving
the exact same document_id/stored_name/content_sha256/display_name.

The PHYSICAL FILE is never touched — only its `.meta.json` sidecar. The
old v2 JSON is never silently discarded: write_sidecar_atomic() (same
primitive every other sidecar write in this codebase uses) means a
migrated sidecar is only ever observed either fully v2 or fully v3, never
partially written; a failure partway through this script leaves every
already-migrated sidecar migrated and every not-yet-reached one still v2,
safe to rerun.

Fail-closed candidates (never migrated, always skipped with a safe
reason): v1 (no owner ever recorded — nothing to resolve), any sidecar
that fails containment/secure-read/schema validation, and any sidecar
whose `document_id`/`stored_name` don't correspond to the physical file
it's paired with. A skip is never a guess.

Default behavior is a safe dry run: enumerate and validate every v2
sidecar, resolve what its target owner UUID WOULD be, make ZERO writes.
Pass --apply to actually rewrite sidecars in place (this DOES resolve/
create internal users in PostgreSQL and DOES rewrite sidecar files).

After migrating, run `python -m scripts.rebuild_qdrant --apply` to
reconcile the now-v3 sidecars into the current (UUID-owned) Qdrant
collection.

Supported invocation (module form, from the repository root):

    python -m scripts.migrate_sidecars_v2_to_v3            # dry run
    python -m scripts.migrate_sidecars_v2_to_v3 --apply     # migrate for real
"""

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from rag.identity import upload_document_id
from rag.safe_files import SecureReadError, read_regular_file_secure
from rag.sidecar import (
    PathContainmentError,
    SidecarError,
    build_sidecar,
    parse_sidecar_bytes,
    resolve_managed_upload_path,
    resolve_sidecar_path,
    secure_read_sidecar_bytes,
    sidecar_path_for,
    write_sidecar_atomic,
)
from utils.logging import logger

# db.documents/db.identity/utils.access_control (the fail-closed Telegram
# allowlist) are all imported lazily, inside apply_plan()/main()'s --apply
# branch (and never at all on a dry run) — a dry run must never require
# PostgreSQL or a Telegram credential to be reachable, and this module must
# stay importable in a minimal environment with no `db` package present at
# all, mirroring scripts/rebuild_qdrant.py's own credential/dependency-
# independent dry-run contract.


@dataclass(frozen=True)
class MigrationCandidate:
    physical_path: Path
    sidecar_path: Path
    document_id: str
    display_name: str
    stored_name: str
    content_sha256: str
    telegram_owner_id: int


@dataclass
class MigrationPlan:
    candidates: List[MigrationCandidate]
    skipped_reasons: List[str]


def _validate_candidate(
    physical_path: Path, uploads_dir: Path, uploads_root: Path
) -> Tuple[Optional[MigrationCandidate], Optional[str]]:
    """
    Validate ONE upload as strictly as scripts/rebuild_qdrant.py's own
    `_plan_upload_documents()` does — containment, secure read, schema,
    and physical-file/sidecar identity correspondence — and return either
    `(candidate, None)` or `(None, skip_reason)`. Purely local filesystem
    reads + hashing; makes NO PostgreSQL call and NO Qdrant call.

    Stage 5C corrective pass #2 (Section 5): this is the SAME validation
    `build_plan()` uses to construct a plan and `apply_plan()` uses to
    REVALIDATE a candidate immediately before its irreversible durable
    transition — a single implementation, so a plan can never be trusted
    as still-accurate stale data at apply time. Returns `(None, None)`
    for anything that is silently not a migration candidate at all (not a
    regular file, or a sidecar name) rather than a genuine skip reason.
    """
    if not physical_path.is_file():
        return None, None
    if physical_path.name.endswith(".meta.json"):
        return None, None

    candidate_sidecar_path = sidecar_path_for(physical_path)
    if candidate_sidecar_path.is_symlink():
        logger.warning("Sidecar migration: skipping upload whose sidecar path is a symlink")
        return None, "path_containment_violation"
    if not candidate_sidecar_path.is_file():
        return None, "missing_sidecar"

    try:
        sidecar_path = resolve_sidecar_path(uploads_dir, physical_path)
    except PathContainmentError as e:
        logger.warning("Sidecar migration: skipping upload whose sidecar failed path containment | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    try:
        sidecar_bytes = secure_read_sidecar_bytes(uploads_root, sidecar_path)
    except SidecarError as e:
        logger.warning("Sidecar migration: skipping upload whose sidecar failed secure read (possible race) | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    try:
        sidecar = parse_sidecar_bytes(sidecar_bytes)
    except SidecarError as e:
        logger.warning("Sidecar migration: skipping upload with invalid sidecar | error_type=%s", type(e).__name__)
        return None, "invalid_sidecar"

    if sidecar["schema_version"] != 2:
        # Not migration input: v1 has no owner to resolve (never
        # migrated — see module docstring); v3 is already migrated.
        if sidecar["schema_version"] == 1:
            return None, "v1_no_owner_to_migrate"
        return None, "already_v3"

    try:
        resolve_managed_upload_path(uploads_root, sidecar["stored_name"])
        resolve_managed_upload_path(uploads_root, physical_path.name)
    except PathContainmentError as e:
        logger.warning("Sidecar migration: skipping upload that failed path containment | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    expected_document_id = upload_document_id(physical_path.stem)
    if sidecar["document_id"] != expected_document_id or sidecar["stored_name"] != physical_path.name:
        logger.warning("Sidecar migration: skipping upload with sidecar/physical-file identity mismatch")
        return None, "sidecar_identity_mismatch"

    try:
        source_bytes = read_regular_file_secure(physical_path, root=uploads_root)
    except SecureReadError as e:
        logger.warning("Sidecar migration: skipping upload whose source failed secure read (possible race) | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    from rag.identity import sha256_hex
    actual_content_sha256 = sha256_hex(source_bytes)
    if actual_content_sha256 != sidecar["content_sha256"]:
        logger.warning("Sidecar migration: skipping upload whose content no longer matches its sidecar hash")
        return None, "sidecar_content_hash_mismatch"

    candidate = MigrationCandidate(
        physical_path=physical_path,
        sidecar_path=sidecar_path,
        document_id=sidecar["document_id"],
        display_name=sidecar["display_name"],
        stored_name=sidecar["stored_name"],
        content_sha256=actual_content_sha256,
        telegram_owner_id=sidecar["owner_user_id"],
    )
    return candidate, None


def build_plan(uploads_dir: Path) -> MigrationPlan:
    """
    Enumerate every v2 sidecar under `uploads_dir` and validate it via
    `_validate_candidate()` before it is ever considered a migration
    candidate. Purely local filesystem reads + hashing; makes NO
    PostgreSQL call and NO Qdrant call, so it is always safe to run.

    PLANNING IS ADVISORY (Stage 5C corrective pass #2, Section 5): the
    candidates returned here describe the state observed AT THIS MOMENT.
    An operator may inspect this plan before deciding to apply it, and the
    physical upload the plan describes could change in the meantime —
    apply_plan() never trusts these values as still-accurate; it
    revalidates each one fresh, immediately before its own irreversible
    durable transition.
    """
    uploads_dir = Path(uploads_dir)
    candidates: List[MigrationCandidate] = []
    skipped_reasons: List[str] = []

    if not uploads_dir.exists():
        return MigrationPlan(candidates=[], skipped_reasons=[])

    uploads_root = uploads_dir.resolve()

    for physical_path in sorted(uploads_dir.iterdir()):
        candidate, skip_reason = _validate_candidate(physical_path, uploads_dir, uploads_root)
        if candidate is not None:
            candidates.append(candidate)
        elif skip_reason is not None:
            skipped_reasons.append(skip_reason)

    return MigrationPlan(candidates=candidates, skipped_reasons=skipped_reasons)


@dataclass(frozen=True)
class CatalogReconciliation:
    """
    Provenance of one `_reconcile_catalog_row()` call (Stage 5C corrective
    pass #7 — Blocker: "a late-aborted legacy migration may delete an
    identical PostgreSQL pending catalog row that existed before the
    current migration attempt").

    `created_by_this_attempt`:
      - True  — this call's own INSERT is the one that put the row there.
        Only in this case may a later abort's cleanup ever remove it.
      - False — a row already existed (either observed up front, or lost a
        race to insert one — see below) and its complete metadata agreed
        with this migration candidate, so it was safely REUSED rather than
        created. This attempt did not make it, and must never claim it for
        cleanup on a later abort — an identical pre-existing row must
        survive this attempt regardless of what this attempt does next.
      - None  — authorship could not be conclusively established (an
        ambiguous database outcome — see the `except Exception` branch
        below). UNKNOWN ORIGIN must never be treated as CREATED BY THIS
        ATTEMPT: a later abort must preserve the row rather than risk
        deleting state that predates this attempt.

    A mismatching existing row never reaches this dataclass at all —
    `_reconcile_catalog_row()` raises `CatalogConsistencyError` for that
    case instead, exactly as before this corrective pass.
    """
    created_by_this_attempt: Optional[bool]


def _reconcile_catalog_row(
    *, document_id: uuid.UUID, owner_user_id: uuid.UUID, stored_name: str, display_name: str, content_sha256: str
) -> CatalogReconciliation:
    """
    Create (or, on a rerun, verify) the PostgreSQL `documents` catalog row
    for a migrated legacy upload — Stage 5C corrective pass: after Stage
    5C, a v2->v3 sidecar migration is incomplete without this (the
    catalog is the canonical durable ownership/lifecycle record; a v3
    sidecar with no corresponding row fails scripts/rebuild_qdrant.py's
    ownership gate closed and can never be indexed).

    Status is deliberately 'pending', never 'active': Qdrant has not been
    rebuilt yet at this point in the migration (that is a separate,
    subsequent `python -m scripts.rebuild_qdrant --apply` step — see this
    module's own docstring) — claiming 'active' here would itself be an
    inconsistent catalog claim, the exact defect this corrective pass
    closes elsewhere. `scripts/rebuild_qdrant.py`'s own catalog gate
    accepts a 'pending' row for planning and flips it to 'active' only
    once it has genuinely reconciled real Qdrant content for it.

    On a rerun (this document_id already has a row — e.g. a previous
    migration run already created it, or rebuild already activated it),
    the existing row must agree on owner/stored_name/display_name/
    content_sha256 or this raises db.documents.CatalogConsistencyError:
    never overwrite a disagreeing row, and never silently treat a
    mismatch as success.

    Stage 5C corrective pass #3 (Blocker 4): `display_name` is now part
    of this comparison — a prior version of this check omitted it (the
    same gap normal ingestion's own final verification had before Stage
    5C corrective pass #2 fixed it there), so migration could accept and
    proceed past a catalog row whose display_name genuinely disagreed
    with the legacy sidecar being migrated. Compared with plain string
    equality — the same semantic normal ingestion verification uses; no
    normalization is applied anywhere in this contract, so none is
    introduced here either.

    Stage 5C corrective pass #7: returns a `CatalogReconciliation`
    recording whether THIS call created the row, reused a pre-existing
    one, or could not conclusively tell — see that dataclass's own
    docstring. Previously this returned nothing, and the caller's abort
    path treated "the row still matches the candidate" as proof that this
    attempt created it; an independent real-PostgreSQL reproduction showed
    that is false whenever a pre-existing, unrelated migration attempt
    (or an already-existing identical row from any other source) happens
    to occupy the same document_id: this attempt would reuse it here, the
    v2 source would mutate before the final barrier, migration would
    abort correctly, and the caller's cleanup — driven only by "does it
    still match?", never "did *I* make it?" — would delete a row that
    predates this attempt entirely. Establishing provenance HERE, at the
    point where the database itself proves it (an INSERT either succeeds,
    meaning nothing was there before, or fails on the primary-key
    constraint, meaning something already was), is the one place this can
    be answered with certainty — reconstructing it later from "the row's
    current values happen to match" can never distinguish reused-and-
    untouched from created-by-this-attempt.
    """
    import db.documents as db_documents
    from sqlalchemy.exc import IntegrityError

    def _verify_matches_or_raise(existing) -> CatalogReconciliation:
        if (
            existing.owner_user_id != owner_user_id
            or existing.stored_name != stored_name
            or existing.display_name != display_name
            or existing.content_sha256 != content_sha256
        ):
            raise db_documents.CatalogConsistencyError(
                "existing catalog row disagrees with the legacy sidecar being migrated"
            )
        return CatalogReconciliation(created_by_this_attempt=False)

    existing = db_documents.get_sync(document_id=document_id)
    if existing is not None:
        # Pre-existing row, observed up front — reused, never created by
        # this attempt (item 1/6 of the concurrency semantics this
        # corrective pass reviewed: "existing row observed before insert").
        return _verify_matches_or_raise(existing)

    try:
        db_documents.create_pending_sync(
            document_id=document_id, owner_user_id=owner_user_id,
            stored_name=stored_name, display_name=display_name, content_sha256=content_sha256,
        )
    except IntegrityError:
        # Item 3 of the concurrency semantics reviewed: another
        # transaction inserted an identical (or disagreeing) row for this
        # exact document_id in the window between our lookup above and
        # this INSERT. PostgreSQL's own primary-key constraint is the
        # atomic arbiter here — there is no reconstruction from later
        # state involved: losing this race conclusively proves we did NOT
        # create the row now occupying this id, regardless of what its
        # values turn out to be.
        try:
            existing = db_documents.get_sync(document_id=document_id)
        except Exception:
            # The DB round trip to even find out what's there now failed —
            # authorship is genuinely unknowable either way.
            return CatalogReconciliation(created_by_this_attempt=None)
        if existing is None:
            # Vanishingly unlikely (deleted again in the interim) — still
            # not proof of anything this attempt did.
            return CatalogReconciliation(created_by_this_attempt=None)
        return _verify_matches_or_raise(existing)
    except Exception:
        # Item 4: create_pending_sync()'s own session.commit() can, in
        # principle, durably commit its INSERT and still raise back to
        # this caller (the same ambiguous-acknowledgement window
        # db.documents.reconcile_ambiguous_create_pending_sync()'s own
        # docstring documents for app/documents.py's normal ingestion
        # path). Whether the INSERT actually landed or not cannot be
        # proven from here — per this corrective pass's required
        # invariant, UNKNOWN ORIGIN must never be treated as CREATED BY
        # THIS ATTEMPT, so this is reported as unknown rather than
        # guessed either way. The caller leaves this candidate unmigrated;
        # if a row was in fact left behind, it is safe to revisit on a
        # rerun (the well-known, already-documented orphaned-pending-row
        # follow-up — not a correctness defect).
        return CatalogReconciliation(created_by_this_attempt=None)

    return CatalogReconciliation(created_by_this_attempt=True)


def apply_plan(
    plan: MigrationPlan,
    resolve_owner_uuid: Callable[[int], Optional[uuid.UUID]],
    uploads_dir: Path,
    *,
    _test_pre_sidecar_write_hook: Optional[Callable[["MigrationCandidate"], None]] = None,
) -> int:
    """
    For every candidate: SECURELY REVALIDATE it against the actual current
    filesystem state (Stage 5C corrective pass #2, Section 5 — see below),
    resolve its legacy Telegram owner to an internal UUID (fail closed —
    `resolve_owner_uuid` returns None for a Telegram id that is neither an
    existing mapping nor currently allowlisted, and that candidate is
    skipped, never guessed), reconcile the PostgreSQL `documents` catalog
    row for it, and ONLY THEN rewrite its sidecar in place as v3 — in that
    order, so the v2 sidecar (the only record of this upload's legacy
    ownership) is never touched until the new durable catalog state is
    already proven consistent. `resolve_owner_uuid` is a
    `Callable[[int], Optional[uuid.UUID]]` — production callers pass
    `_resolve_owner_uuid_fail_closed` (below); tests pass a deterministic
    fake so this function itself makes no PostgreSQL call in that case
    (db.documents' own functions are still called, but tests without a
    real database rely on the same fixture-faked db.documents used
    throughout this codebase's offline suite — see tests/conftest.py).

    Revalidation (Section 5 — TOCTOU closed): `plan.candidates` describes
    filesystem state observed at PLAN time, which may be stale by the time
    this function actually runs (an operator can review a dry-run plan
    before applying it; a caller can also hold a plan across an arbitrary
    gap for any other reason). Trusting the plan's own recorded
    content_sha256/stored_name/document_id at apply time — the previous
    corrective pass's defect — let a physical file mutated after planning
    still be migrated under its OLD, no-longer-accurate hash: a
    PostgreSQL catalog row and a v3 sidecar would both get created from
    stale metadata, an inconsistency no later rerun could detect (`already_
    v3` short-circuits before any content check). Every candidate is
    therefore re-run through the EXACT SAME `_validate_candidate()`
    `build_plan()` itself uses — securely reopening the physical file
    (`read_regular_file_secure()`), recomputing its content_sha256, and
    reconfirming the v2 sidecar's document_id/stored_name relationship —
    immediately before this function's own irreversible durable
    transition (the catalog write below). If the freshly revalidated
    candidate disagrees with the one the caller passed in (content
    changed, sidecar changed/removed, no longer schema_version 2, ...),
    this candidate is skipped closed: no catalog row is created/updated
    from stale metadata, no v3 sidecar is written, and the original v2
    ownership record is left completely untouched — reported the same way
    any other skip is, safe for an operator to investigate and rerun.

    Returns the number of sidecars actually rewritten. A failure partway
    through (an unresolvable/disallowed owner, a disagreeing existing
    catalog row, a revalidation mismatch, or a genuine write failure)
    leaves every already-migrated sidecar migrated (write_sidecar_atomic()
    never leaves a partial file), every already-reconciled catalog row
    exactly as reconciled, and every not-yet-reached candidate completely
    untouched — safe to rerun.

    Stage 5C corrective pass #4 (Blocker 6): the single revalidation above
    proves freshness as of the TOP of this iteration, but an independent
    audit reproduced a SECOND mutation window this alone didn't close:
    `resolve_owner_uuid()` can itself take arbitrary time (a real
    PostgreSQL round trip), and the file can mutate again during it —
    "revalidate once per apply() iteration" is not the same guarantee as
    "revalidate immediately before each durable write". Two further
    deterministic barriers close the remaining gap, each re-running the
    EXACT SAME `_validate_candidate()`:
      - immediately before `_reconcile_catalog_row()` (the first durable
        write) — catches a mutation during `resolve_owner_uuid()`;
      - immediately before `write_sidecar_atomic()` (THE irreversible
        step — this is what destroys the v2 ownership record) — catches a
        mutation during the catalog round trip itself. If the file
        changed again since the catalog write, the sidecar is NOT
        touched: the original v2 record is preserved untouched rather
        than being destroyed to reflect state that's already stale by the
        time it would be written. The catalog row, already reconciled
        against a genuinely validated snapshot, is simply left for a
        later rerun to pick up again — never claimed migrated when it
        isn't.
    Neither barrier eliminates every theoretical race (a mutation in the
    residual instant between the last check and the write itself remains
    possible, as with any check-then-act sequence not wrapped in a single
    filesystem-level transaction) — but each closes a concretely
    reproducible window, and downstream `scripts/rebuild_qdrant.py`
    independently re-validates against current source before ever
    activating anything in Qdrant, so this script does not need to be the
    sole line of defense.

    Stage 5C corrective pass #5 (Blocker 3): the THIRD barrier (immediately
    before `write_sidecar_atomic()`) used to compare ONLY
    `final_check.content_sha256 != candidate.content_sha256` — a WEAKER
    check than barriers 1 and 2, both of which already compare the
    complete `MigrationCandidate` (`candidate != planned_candidate`, full
    dataclass equality). An independent audit exploited exactly that gap:
    it changed the v2 sidecar's `owner_user_id` (or `display_name`) AFTER
    `_reconcile_catalog_row()` had already reconciled the catalog row
    against the OLD values, while leaving the physical file's bytes
    completely unchanged — the content-hash-only barrier saw no
    disagreement and let `write_sidecar_atomic()` overwrite the v2 sidecar
    (now genuinely recording the NEWER owner/display_name) with a v3
    sidecar built from the STALE `candidate` snapshot, permanently
    destroying the newer, still-legitimate legacy ownership metadata.
    Barrier 3 now compares the COMPLETE candidate too
    (`final_check != candidate`) — the exact same full-dataclass-equality
    check barriers 1/2 already use — so ANY field changing (owner,
    display_name, stored_name, document_id, content hash, or the sidecar
    no longer being schema_version 2 at all) aborts this candidate exactly
    like a content mutation always did, preserving the newer legacy
    ownership record untouched.
    """
    import db.documents as db_documents

    uploads_dir = Path(uploads_dir)
    uploads_root = uploads_dir.resolve()

    migrated = 0
    for planned_candidate in plan.candidates:
        candidate, skip_reason = _validate_candidate(planned_candidate.physical_path, uploads_dir, uploads_root)
        if candidate is None or candidate != planned_candidate:
            logger.warning(
                "Sidecar migration: skipping upload whose durable state no longer matches the "
                "plan (mutated, removed, or already migrated since planning) | skip_reason=%s",
                skip_reason,
            )
            continue

        owner_user_uuid = resolve_owner_uuid(candidate.telegram_owner_id)
        if owner_user_uuid is None:
            logger.warning(
                "Sidecar migration: skipping upload whose legacy Telegram owner has no existing "
                "mapping and is not currently allowlisted — leaving it unmigrated"
            )
            continue

        # Stage 5C corrective pass #4 (Blocker 6), barrier 2: re-validate
        # again, immediately before the first durable/irreversible write —
        # resolve_owner_uuid() above can itself take arbitrary time (a real
        # PostgreSQL round trip), leaving another window in which the
        # physical file could mutate before the catalog write.
        candidate, skip_reason = _validate_candidate(planned_candidate.physical_path, uploads_dir, uploads_root)
        if candidate is None or candidate != planned_candidate:
            logger.warning(
                "Sidecar migration: skipping upload — source mutated again after owner resolution, "
                "before the catalog write | skip_reason=%s",
                skip_reason,
            )
            continue

        document_id = uuid.UUID(candidate.physical_path.stem)
        try:
            reconciliation = _reconcile_catalog_row(
                document_id=document_id,
                owner_user_id=owner_user_uuid,
                stored_name=candidate.stored_name,
                display_name=candidate.display_name,
                content_sha256=candidate.content_sha256,
            )
        except db_documents.CatalogConsistencyError:
            logger.warning(
                "Sidecar migration: skipping upload whose existing PostgreSQL catalog row "
                "disagrees with its legacy sidecar — leaving both untouched"
            )
            continue

        if reconciliation.created_by_this_attempt is None:
            # Stage 5C corrective pass #7: authorship of whatever catalog
            # state now exists for this document_id could not be
            # conclusively established (an ambiguous database outcome —
            # see _reconcile_catalog_row()'s own docstring). Leave this
            # candidate's v2 sidecar completely untouched rather than
            # proceed on an unproven catalog state — safe to rerun.
            logger.warning(
                "Sidecar migration: skipping upload whose catalog reconciliation outcome could "
                "not be conclusively established — leaving the legacy sidecar untouched; safe "
                "to rerun"
            )
            continue

        if _test_pre_sidecar_write_hook is not None:
            # Test-only seam (mirrors rag.safe_files.read_regular_file_
            # secure()'s own `_test_pre_open_hook` convention) — called
            # ONLY so a test can deterministically mutate the physical file
            # in the exact window barrier 3 below exists to close. Every
            # real caller leaves this None, making it a complete no-op in
            # production.
            _test_pre_sidecar_write_hook(candidate)

        # Stage 5C corrective pass #4 (Blocker 6), barrier 3 — the most
        # important one: re-validate ONE MORE TIME, immediately before the
        # step that DESTROYS the v2 sidecar (write_sidecar_atomic()
        # overwrites it in place with v3). If the file changed again since
        # the catalog write just above, do NOT touch the sidecar — the
        # original v2 ownership record is preserved untouched, and the
        # (already-consistent-as-of-reconciliation) catalog row is simply
        # left for a later rerun to revisit.
        #
        # Stage 5C corrective pass #5 (Blocker 3): compares the COMPLETE
        # candidate (`final_check != candidate`, full dataclass equality —
        # document_id, display_name, stored_name, content_sha256, AND
        # telegram_owner_id all together) — never content_sha256 alone.
        # See this function's own docstring above for the exact
        # owner/display_name overwrite this closes.
        final_check, skip_reason = _validate_candidate(planned_candidate.physical_path, uploads_dir, uploads_root)
        if final_check is None or final_check != candidate:
            logger.warning(
                "Sidecar migration: skipping upload — legacy sidecar state changed again after the "
                "catalog row was reconciled; preserving the original v2 sidecar untouched | skip_reason=%s",
                skip_reason,
            )
            # Stage 5C corrective pass #5 (Blocker 3): the catalog write
            # just above (_reconcile_catalog_row()) may have INSERTED a
            # brand-new 'pending' row for this document_id (the common
            # case: no row existed yet). Left in place, that row would
            # block a later rerun — `_reconcile_catalog_row()` raises
            # CatalogConsistencyError on ANY disagreement with a rerun's
            # freshly-resolved (newer) owner/display_name, since a
            # mismatching existing row is never silently overwritten.
            #
            # Stage 5C corrective pass #7: cleanup is now gated on
            # `reconciliation.created_by_this_attempt is True` — the
            # release blocker this pass closes. The previous version of
            # this code called reconcile_ambiguous_create_pending_sync()
            # unconditionally whenever the row still matched what this
            # attempt would have written, reasoning "it matches, so it's
            # safe to delete." An independent real-PostgreSQL reproduction
            # showed that reasoning is false: an IDENTICAL pending row that
            # already existed BEFORE this attempt ever ran (this attempt
            # only reused it — see _reconcile_catalog_row()) also "still
            # matches", and was being deleted right along with a
            # genuinely attempt-created one. A matching row is not proof
            # that this attempt created it. Only when
            # `_reconcile_catalog_row()` itself proved — at the one moment
            # the database can answer this atomically, the INSERT itself —
            # that THIS call is the one that put the row there does
            # cleanup run at all; a reused pre-existing row, or one whose
            # authorship could not be conclusively established, is left
            # completely untouched here. `reconcile_ambiguous_create_
            # pending_sync()` (the same atomic-conditional-DELETE
            # primitive app.documents.py's own ambiguous-create-pending
            # window uses) is still the mechanism that performs the
            # deletion once authorship is confirmed — it deletes the row
            # ONLY if it still exactly matches what THIS attempt wrote AND
            # is still 'pending', never a row a concurrent process has
            # since activated/changed.
            if reconciliation.created_by_this_attempt:
                try:
                    db_documents.reconcile_ambiguous_create_pending_sync(
                        document_id=document_id, owner_user_id=owner_user_uuid,
                        stored_name=candidate.stored_name, display_name=candidate.display_name,
                        content_sha256=candidate.content_sha256,
                    )
                except Exception as e:
                    logger.warning(
                        "Sidecar migration: could not clean up a possibly attempt-created pending catalog "
                        "row after aborting this candidate — safe to rerun (a future reconciliation attempt "
                        "will re-evaluate it) | error_type=%s",
                        type(e).__name__,
                    )
            else:
                # reconciliation.created_by_this_attempt is False here —
                # the None case already `continue`d right after
                # _reconcile_catalog_row() was called, above.
                logger.info(
                    "Sidecar migration: aborting this candidate — the catalog row was reused from a "
                    "pre-existing row, not created by this attempt, so it is left untouched"
                )
            continue

        write_sidecar_atomic(
            candidate.sidecar_path,
            build_sidecar(
                document_id=candidate.document_id,
                display_name=candidate.display_name,
                stored_name=candidate.stored_name,
                content_sha256=candidate.content_sha256,
                owner_user_uuid=str(owner_user_uuid),
            ),
        )
        migrated += 1
        logger.info("Sidecar migration: rewrote sidecar as v3 | migrated_count=%s", migrated)
    return migrated


def _resolve_owner_uuid_fail_closed(telegram_id: int) -> Optional[uuid.UUID]:
    """
    Production `resolve_owner_uuid` for apply_plan() (Stage 5C corrective
    pass ownership rule): a legacy Telegram id found in a v2 sidecar must
    never be silently onboarded as a brand-new internal user merely by
    being migrated.

    1. If a Telegram->UUID mapping already exists, use it unconditionally
       — this is migrating already-legitimately-owned data, never a live
       authorization decision.
    2. Otherwise, only create one if `telegram_id` is CURRENTLY on the
       fail-closed Telegram allowlist (utils.access_control) — the exact
       same gate a live Telegram request from this id would have to pass.
    3. Otherwise, return None: the caller (apply_plan()) leaves this
       candidate's sidecar unmigrated rather than guessing.

    Never uses username/display name as identity — Telegram numeric id is
    the only input, same as every other identity resolution in this
    codebase.
    """
    import db.identity as db_identity
    import utils.access_control as access_control

    existing = db_identity.lookup_user_by_telegram_id_sync(telegram_id)
    if existing is not None:
        return existing
    if not access_control.is_authorized(telegram_id):
        return None
    return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.migrate_sidecars_v2_to_v3",
        description=(
            "One-time migration: rewrite v2 (Telegram-integer-owned) managed-"
            "upload sidecars to v3 (canonical-UUID-owned) in place. Never "
            "touches physical files, never touches Qdrant — run "
            "`python -m scripts.rebuild_qdrant --apply` afterward to reconcile "
            "the migrated sidecars into the current Qdrant collection."
        ),
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually reconcile the PostgreSQL documents catalog and rewrite "
             "sidecars in place (resolves/creates internal users in PostgreSQL "
             "for allowlisted legacy owners only). Without this flag, performs "
             "a safe dry run only: no PostgreSQL call, no file write.",
    )
    args = parser.parse_args(argv)

    from rag.constants import MANAGED_UPLOADS_DIR

    plan = build_plan(MANAGED_UPLOADS_DIR)

    print(f"v2 sidecars found and eligible for migration: {len(plan.candidates)}")
    if plan.skipped_reasons:
        print(f"Uploads skipped (v1/v3/invalid/mismatched): {len(plan.skipped_reasons)}")

    if not args.apply:
        print("\nDry run only: no PostgreSQL call was made, no sidecar was rewritten.")
        print("Pass --apply to actually migrate v2 sidecars to v3.")
        return 0

    print(
        "\nApplying migration: resolving legacy Telegram owners (existing mappings, or "
        "currently-allowlisted ids only) to internal UUIDs, reconciling the PostgreSQL "
        "documents catalog, and rewriting sidecars..."
    )
    migrated = apply_plan(plan, _resolve_owner_uuid_fail_closed, MANAGED_UPLOADS_DIR)
    skipped_count = len(plan.candidates) - migrated
    print(f"Migration complete: {migrated} sidecar(s) rewritten as v3.")
    if skipped_count:
        print(
            f"{skipped_count} eligible sidecar(s) were left unmigrated (unmapped/disallowed "
            "legacy owner, or a disagreeing existing catalog row — see logs)."
        )
    print("Run `python -m scripts.rebuild_qdrant --apply` next to reconcile them into Qdrant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
