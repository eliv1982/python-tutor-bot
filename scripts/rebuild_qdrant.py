#!/usr/bin/env python
"""
Operator-facing utility to rebuild the Qdrant index from source.

Stage 2B treats Qdrant as fully rebuildable DERIVED state — never itself
the source of truth. This script is the explicit, manual rebuild path. It
is NEVER invoked automatically at application startup (main.py only calls
VectorIndex.index_documents_directory() for reference documents; managed
uploads are indexed individually, once, at upload time).

Rebuilds ONLY from:
    1. the built-in Markdown reference documents named in
       config.BUILTIN_REFERENCE_FILES (data/documents/*.md) — enumerated
       EXACTLY, never an unconstrained extension-based scan (Stage 2B-C
       Blocker 5): a stray legacy `.txt` or an arbitrary extra `.md` sitting
       in data/documents/ is never silently promoted into product knowledge.
    2. managed uploads that have a VALID `.meta.json` sidecar
       (data/documents/uploads/)

This is deliberately NOT a Chroma migration utility: it never reads the
legacy Chroma store (data/chroma_db/), never depends on it, and never
compares against its point count. Legacy Chroma is out of scope by
product decision — see README.md.

Default behavior is a safe dry run: enumerate and validate every source
document, make ZERO provider calls, make ZERO Qdrant mutations. Pass
--apply to actually reconcile Qdrant to source (this DOES call the
embeddings provider for new/changed documents and DOES mutate Qdrant).

Stage 2B-C Blocker 3: --apply is a NON-DESTRUCTIVE reconciliation, never a
clear-then-rebuild. The existing collection is never cleared first; each
desired document is safely reconciled (embed-then-upsert-then-delete-stale,
or a zero-embedding convergence when already current — see
VectorIndex.reconcile_document()), and orphan logical documents (indexed in
Qdrant but no longer present in source truth) are removed ONLY after every
desired document has reconciled successfully. A failure partway through
leaves every already-reconciled document, and the untouched rest of the
index, completely intact — a deterministic rerun converges.

Stage 2B-C Blocker 4: the supported operator invocation is the MODULE form,
run from the repository root:

    python -m scripts.rebuild_qdrant            # dry run (default)
    python -m scripts.rebuild_qdrant --apply     # rebuild for real

Running this file directly (`python scripts/rebuild_qdrant.py`) does NOT
work — Python puts only this file's own directory on sys.path in that mode,
so top-level packages like `rag`/`config` are not importable. This is
deliberately NOT worked around with sys.path manipulation; the module form
above is the supported contract.
"""

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rag.constants import BUILTIN_REFERENCE_FILES
from rag.identity import reference_document_id, sha256_hex, upload_document_id
from rag.loader import SUPPORTED_EXTENSIONS, document_loader
from rag.safe_files import SecureReadError, read_regular_file_secure
from rag.sidecar import (
    PathContainmentError,
    SidecarError,
    parse_sidecar_bytes,
    resolve_managed_upload_path,
    resolve_sidecar_path,
    secure_read_sidecar_bytes,
    sidecar_path_for,
)
from utils.logging import logger

# Stage 2B-D Section D/E/H: everything imported above is pure/side-effect-
# free and requires ZERO credentials/environment — this is what makes
# `python -m scripts.rebuild_qdrant`'s dry-run path credential-independent.
# `config` (the full, credential-validating application module) is
# imported only lazily, inside main()'s `--apply` branch below.
#
# Stage 5C corrective pass #4 (Blocker 9): the one skip reason that means
# "we genuinely could not determine this candidate's state" rather than
# "we examined it and found it invalid" — see _validate_upload_candidate()
# and apply_plan()'s orphan-pruning gate below, which must never treat the
# former as grounds for destructive pruning (Principle 5: "could not
# establish state" must never be treated as "state absent").
CATALOG_UNREACHABLE_REASON = "catalog_unreachable"

# db.documents is imported LAZILY inside _validate_upload_candidate() below
# (Stage 5C corrective pass), never at module level: this module is copied
# into several tests' isolated, `db`-package-free subprocess trees (see
# tests/test_stage2d_hardening.py's _copy_importable_tree()) that prove a
# credential-free dry run with no managed uploads present needs nothing
# beyond rag/scripts/utils — a top-level `import db.documents` would break
# that contract outright. A real PostgreSQL round trip only happens once
# there is at least one local v3-sidecar upload candidate to validate
# ownership for (proving private-document ownership against the canonical
# catalog is no longer optional — Section 2/5: a private document must
# never be planned for indexing on the strength of a syntactically-valid
# sidecar alone) — and even then, a genuine failure to reach/import it
# fails that ONE document closed (skipped, logged) rather than crashing
# the whole rebuild run; see the try/except around the lazy import below.


@dataclass(frozen=True)
class SourceDocument:
    """One document the rebuild plan will reconcile, resolved from a
    single, trusted source: either a version-controlled reference file or
    a managed upload with a validated sidecar.

    Stage 2B-E Section F (plan-to-apply content integrity): `content_bytes`
    carries the EXACT bytes already securely read once, at plan-build
    time, for a managed upload (never set for a reference document — see
    Section G: version-controlled files are deliberately not
    overengineered). apply_plan() passes these bytes straight through to
    VectorIndex.reconcile_document(), which then never reopens
    `physical_path`'s pathname for the actual parsed/embedded content —
    closing the plan-to-apply gap completely (a swap of the on-disk source
    after the plan was built cannot influence what gets embedded)."""
    kind: str  # "reference" | "upload"
    document_id: str
    display_source: str  # recorded as the `source` / display_name metadata
    physical_path: Path
    content_sha256: str
    stored_name: Optional[str] = None
    content_bytes: Optional[bytes] = None
    # Stage 3A ownership, migrated to canonical UUID Stage 5C: the
    # sidecar's `owner_user_uuid` for an "upload" document, sourced from
    # the sidecar itself (never guessed) — always None for a "reference"
    # document (no owner). apply_plan() passes this straight to
    # VectorIndex.reconcile_document(), which derives scope="private" (with
    # this owner) vs. scope="reference" from it exactly the same way the
    # live upload path does.
    owner_user_uuid: Optional[str] = None
    # Stage 5C corrective pass: the PostgreSQL `documents.id` this "upload"
    # document's catalog row was already proven to match at plan-build time
    # (see _plan_upload_documents()'s catalog gate below) — always None for
    # a "reference" document (no catalog row). apply_plan() uses this to
    # re-affirm/activate the row via db.documents.mark_active_sync() only
    # AFTER Qdrant reconciliation for this exact document has genuinely
    # succeeded, never before.
    document_uuid: Optional[uuid.UUID] = None


@dataclass
class RebuildPlan:
    reference_documents: List[SourceDocument] = field(default_factory=list)
    upload_documents: List[SourceDocument] = field(default_factory=list)
    # Safe (no path, no filename) reasons uploads were excluded — e.g. a
    # missing or malformed sidecar. Never includes any user-controlled or
    # filesystem-path text.
    skipped_upload_reasons: List[str] = field(default_factory=list)

    @property
    def all_documents(self) -> List[SourceDocument]:
        return [*self.reference_documents, *self.upload_documents]


@dataclass
class RebuildReport:
    # Every desired document from the plan that reconciled successfully
    # (regardless of whether it actually needed re-embedding).
    documents_reconciled: int = 0
    # Of those, how many were actually (re)embedded — i.e. genuinely
    # new/changed, or missing/outdated expected points. Proves the
    # zero-cost-when-unchanged convergence property (Blocker 2/3).
    documents_reindexed: int = 0
    chunks_reindexed: int = 0
    # Orphan logical documents (indexed in Qdrant, absent from source
    # truth) removed at the end, only after every desired document
    # reconciled successfully.
    documents_removed: int = 0
    # Stage 5C corrective pass #4 (Blocker 7): "upload" documents whose
    # plan-time candidate no longer matches a FRESH revalidation performed
    # immediately before use — the physical file/sidecar/catalog state
    # mutated (or a previously-valid candidate is no longer valid at all)
    # since the plan was built. Never reconciled/activated from stale
    # plan-time data; its document_id is still protected from orphan
    # pruning this run (see apply_plan()) rather than treated as gone.
    uploads_skipped_stale_at_apply: int = 0
    # Stage 5C corrective pass #4 (Blocker 9): set True when planning could
    # not establish a COMPLETE, authoritative private-upload set (the
    # PostgreSQL catalog was unreachable for at least one candidate during
    # planning — see CATALOG_UNREACHABLE_REASON). Orphan pruning below is
    # skipped ENTIRELY whenever this is True: "could not establish state"
    # must never be treated as "state absent" (Principle 5) — an
    # incomplete plan can never safely prove any existing Qdrant document
    # is a genuine orphan, since the very documents the outage hid from
    # planning would look exactly like orphans otherwise.
    orphan_pruning_skipped_incomplete_plan: bool = False
    # Stage 5C corrective pass: "upload" documents whose catalog row was
    # successfully re-affirmed/activated (db.documents.mark_active_sync())
    # immediately after their Qdrant reconciliation succeeded — this is
    # what actually clears a stale 'pending' row (Section 3's deterministic
    # reconciliation path) once rebuild has proven real Qdrant content
    # exists for it.
    documents_catalog_activated: int = 0
    # A catalog activation failure AFTER successful Qdrant reconciliation
    # is logged and counted here rather than aborting the whole rebuild
    # run (unlike a reconciliation failure, which already propagates and
    # halts everything) — the Qdrant content is already correct at that
    # point, and rebuild is an idempotent, safely-rerunnable operator tool:
    # a rerun's catalog gate will simply see the still-non-'active' row
    # again (it already matched at plan time) and retry activation.
    documents_catalog_activation_failed: int = 0
    # Stage 5C corrective pass #5 (Blocker 2): "upload" documents whose
    # CURRENT reconciliation produced ZERO chunks (an empty or whitespace-
    # only source) — never newly activated. Live ingestion already refuses
    # to activate a document that parses into zero meaningful chunks
    # (app.documents.EmptyDocumentError, Principle 6: success must mean a
    # useful, internally consistent document actually exists) — an
    # independent audit reproduced rebuild NOT enforcing the identical
    # rule, calling mark_active_sync() unconditionally after
    # reconcile_document() regardless of chunk_count and activating a
    # pending catalog row with genuinely zero indexed Qdrant points. The
    # catalog row is left exactly as reconciliation found it — a
    # still-'pending' row stays 'pending' (fail-closed/recoverable, never
    # silently promoted), an already-'active' row is simply never
    # re-affirmed here either — and its document_id remains protected from
    # orphan pruning this run (already added to desired_document_ids
    # before reconciliation runs).
    #
    # Stage 5C corrective pass #6 (Blocker 2): this counter fires purely on
    # chunk_count == 0, so it also fires for an ALREADY-'active' document
    # whose current source reconciled to zero chunks — deliberately: the
    # catalog row stays 'active' (see this pass's own lifecycle choice —
    # module docstring / VectorIndex.reconcile_document()'s "emptied"
    # outcome), Qdrant just converges to zero points for it. Never confuse
    # this with "the row was deactivated" — it wasn't; only
    # documents_emptied below distinguishes "stale Qdrant points were
    # actually just removed" from "there was nothing to activate/remove in
    # the first place".
    documents_skipped_zero_chunks: int = 0
    # Stage 5C corrective pass #6 (Blocker 2): "upload"/reference documents
    # whose reconciliation status was "emptied" — the CURRENT source now
    # has zero chunks AND previously-existing Qdrant points for it were
    # just deleted (see VectorIndex.reconcile_document()). Distinct from
    # documents_skipped_zero_chunks above (which counts "catalog activation
    # was skipped because chunk_count == 0" and fires for BOTH a
    # still-'pending' document with nothing to remove and an already-
    # 'active' document whose stale points just got removed): this counter
    # specifically proves the derived-state convergence itself actually
    # happened — Qdrant genuinely had stale points removed this run, not
    # merely "no activation occurred". A document that was already at zero
    # Qdrant points (nothing to remove) reconciles to "unchanged" instead
    # and is never counted here — this counter is 0 for an idempotent rerun.
    documents_emptied: int = 0


def _plan_reference_documents(
    documents_dir: Path, reference_filenames: Optional[Sequence[str]]
) -> List[SourceDocument]:
    documents_dir = Path(documents_dir)
    resolved_root = documents_dir.resolve()
    if reference_filenames is not None:
        source_files = document_loader.list_builtin_reference_files(documents_dir, reference_filenames)
    else:
        source_files = document_loader.list_source_files(documents_dir)
    planned = []
    for file_path in source_files:
        relative = file_path.resolve().relative_to(resolved_root).as_posix()
        document_id = reference_document_id(relative)
        content_sha256 = sha256_hex(file_path.read_bytes())
        planned.append(SourceDocument(
            kind="reference",
            document_id=document_id,
            display_source=file_path.name,
            physical_path=file_path,
            content_sha256=content_sha256,
        ))
    return planned


def _validate_upload_candidate(
    physical_path: Path, uploads_dir: Path, uploads_root: Path
) -> Tuple[Optional[SourceDocument], Optional[str]]:
    """
    Validate ONE managed-upload candidate — containment, secure read,
    schema, sidecar/physical-file identity correspondence, and PostgreSQL
    catalog ownership agreement — and return either `(SourceDocument, None)`
    or `(None, skip_reason)`. Returns `(None, None)` for a directory entry
    that is silently not a candidate at all (not a regular file, a sidecar's
    own `.meta.json` name, or an unsupported extension) rather than a
    genuine skip reason.

    Stage 5C corrective pass #4 (Blocker 7): this is the SAME validation
    `_plan_upload_documents()` uses to build a plan AND `apply_plan()` uses
    to REVALIDATE a planned upload candidate immediately before it is ever
    reconciled into Qdrant / activated in the catalog — a single
    implementation, so a plan can never be trusted as still-accurate stale
    data at apply time (mirrors scripts/migrate_sidecars_v2_to_v3.py's own
    `_validate_candidate()` reuse between its build_plan()/apply_plan()).
    An independent audit reproduced rebuild indexing/activating OLD BYTES:
    a managed upload's plan-time `content_bytes` snapshot was passed
    through to apply_plan() unchanged even after the physical file had
    since mutated, because the hash check inside
    VectorIndex.reconcile_document() compared that SAME stale snapshot
    against itself — a tautology that could never detect a mutation
    occurring after planning. Planning must never be treated as
    authoritative for a destructive/activating apply-time operation
    (Principle 2); apply_plan() below re-runs this exact function fresh,
    right before using an upload candidate, closing that gap.
    """
    if not physical_path.is_file():
        return None, None
    if physical_path.name.endswith(".meta.json"):
        return None, None
    if physical_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None, None

    # Stage 2B-D Blocker 1: the SIDECAR PATH ITSELF must be proven safe
    # BEFORE it is ever opened/read — never read first and validate
    # afterward. A plain (non-symlink) candidate that simply doesn't
    # exist is the ordinary "missing_sidecar" case; a candidate that IS
    # a symlink is always a containment violation, regardless of
    # whether its target happens to exist, and regardless of whether
    # that target is inside or outside uploads_dir (sidecars may not be
    # symlinks, period — see resolve_sidecar_path()'s docstring).
    candidate_sidecar_path = sidecar_path_for(physical_path)
    if candidate_sidecar_path.is_symlink():
        logger.warning("Rebuild plan: skipping upload whose sidecar path is a symlink | document_extension=%s", physical_path.suffix.lower())
        return None, "path_containment_violation"
    if not candidate_sidecar_path.is_file():
        logger.warning("Rebuild plan: skipping upload with no sidecar | document_extension=%s", physical_path.suffix.lower())
        return None, "missing_sidecar"

    try:
        sidecar_path = resolve_sidecar_path(uploads_dir, physical_path)
    except PathContainmentError as e:
        logger.warning("Rebuild plan: skipping upload whose sidecar failed path containment | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    # Stage 2B-E Blocker 1: `resolve_sidecar_path()` above proves the
    # PATHNAME is safe, but a separate reopen of that same pathname to
    # actually read it (the previous `load_sidecar(sidecar_path)` call
    # here) leaves a TOCTOU window in which the filesystem object the
    # pathname refers to can be replaced (e.g. with a symlink to
    # external JSON) between validation and read. secure_read_sidecar_
    # bytes() closes that gap: it performs its own fresh pre-open
    # validation immediately before opening and proves the object
    # actually opened is the object just validated — see
    # rag/safe_files.py's read_regular_file_secure() docstring.
    try:
        sidecar_bytes = secure_read_sidecar_bytes(uploads_root, sidecar_path)
    except SidecarError as e:
        logger.warning("Rebuild plan: skipping upload whose sidecar failed secure read (possible race) | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    try:
        sidecar = parse_sidecar_bytes(sidecar_bytes)
    except SidecarError as e:
        logger.warning("Rebuild plan: skipping upload with invalid sidecar | error_type=%s", type(e).__name__)
        return None, "invalid_sidecar"

    # Stage 5C: direct rebuild planning requires a v3 (canonical UUID
    # owner) sidecar. A v1 (pre-Stage-3A, no owner recorded at all) or
    # v2 (Stage 3A-5B, legacy Telegram-integer owner) sidecar is fail-
    # closed here rather than guessed/silently promoted to
    # scope="reference": v1 has never been safe to reconcile as
    # anyone's private document, and v2's owner identity is no longer
    # canonical — it must go through scripts/migrate_sidecars_v2_to_v3.py
    # (which resolves its Telegram owner to a real internal UUID and
    # rewrites the sidecar in place) before this rebuild path will ever
    # plan it.
    if sidecar["schema_version"] == 1:
        logger.warning("Rebuild plan: skipping upload with no recorded owner (legacy v1 sidecar)")
        return None, "missing_owner"
    if sidecar["schema_version"] == 2:
        logger.warning("Rebuild plan: skipping upload with legacy v2 sidecar (run scripts/migrate_sidecars_v2_to_v3.py first)")
        return None, "legacy_schema_requires_migration"

    # Stage 2B-C Section H: re-resolve the sidecar's declared
    # stored_name through the containment-checked resolver — never
    # trust the directory-listing entry alone. This rejects a
    # symlink placed inside uploads_dir (same name, either the
    # directory entry itself or the name the sidecar declares) that
    # would otherwise be transparently followed by the read_bytes()
    # call below.
    try:
        resolve_managed_upload_path(uploads_root, sidecar["stored_name"])
        resolve_managed_upload_path(uploads_root, physical_path.name)
    except PathContainmentError as e:
        logger.warning("Rebuild plan: skipping upload that failed path containment | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    expected_document_id = upload_document_id(physical_path.stem)
    if sidecar["document_id"] != expected_document_id or sidecar["stored_name"] != physical_path.name:
        # Sidecar content doesn't match the physical file it's paired
        # with (e.g. hand-edited, or copied alongside the wrong file).
        # Fail closed: skip rather than guess which one is right.
        logger.warning("Rebuild plan: skipping upload with sidecar/physical-file identity mismatch")
        return None, "sidecar_identity_mismatch"

    # Stage 2B-E Blocker 1: same TOCTOU class as the sidecar above —
    # `resolve_managed_upload_path()` proved the PATHNAME is safe, but
    # the previous `physical_path.read_bytes()` here reopened it a
    # second time. Read through the secure single-open primitive
    # instead, and carry the exact bytes read forward into the plan
    # (Section F) so apply_plan() never has to reopen this pathname
    # either.
    try:
        source_bytes = read_regular_file_secure(physical_path, root=uploads_root)
    except SecureReadError as e:
        logger.warning("Rebuild plan: skipping upload whose source failed secure read (possible race) | error_type=%s", type(e).__name__)
        return None, "path_containment_violation"

    actual_content_sha256 = sha256_hex(source_bytes)
    if actual_content_sha256 != sidecar["content_sha256"]:
        logger.warning("Rebuild plan: skipping upload whose content no longer matches its sidecar hash")
        return None, "sidecar_content_hash_mismatch"

    # Stage 5C corrective pass (Section 2/5): a private v3 sidecar is
    # NOT sufficient ownership authority on its own — the PostgreSQL
    # `documents` catalog is the canonical durable ownership/lifecycle
    # record, and Qdrant/the sidecar must never become an independent
    # source of truth that disagrees with it. Before this managed
    # upload is ever planned for indexing into the active UUID Qdrant
    # collection, verify the catalog has a corresponding row whose
    # owner/stored_name/display_name/content_sha256 all agree with
    # what was just validated locally, and whose lifecycle status is
    # one this application ever treats as durably indexed. Any
    # disagreement (Stage 5C corrective pass #3, Blocker 4: this now
    # includes display_name, previously omitted here) —
    # including "no such row at all" (e.g. a legacy sidecar that was
    # never migrated through scripts/migrate_sidecars_v2_to_v3.py's
    # catalog-population step) — fails this document closed: it is
    # skipped, never guessed into the plan.
    document_uuid = uuid.UUID(physical_path.stem)
    try:
        import db.documents as db_documents
        catalog_row = db_documents.get_sync(document_id=document_uuid)
    except Exception as e:
        # A genuine failure to even reach/import the catalog layer
        # (PostgreSQL unreachable, or — in a deliberately minimal
        # subprocess environment with no `db` package at all — an
        # ImportError) fails THIS document closed, exactly like any
        # other unprovable ownership claim; it must never crash the
        # whole rebuild run over one connectivity blip, and must never
        # be silently treated as "no catalog row disagrees, so it's
        # fine". Stage 5C corrective pass #4 (Blocker 9): this is the
        # ONE skip reason apply_plan()'s orphan-pruning gate treats as
        # "the plan is incomplete", never as "these documents are
        # genuinely gone" — see CATALOG_UNREACHABLE_REASON.
        logger.warning("Rebuild plan: skipping upload — PostgreSQL catalog unreachable | error_type=%s", type(e).__name__)
        return None, CATALOG_UNREACHABLE_REASON
    if catalog_row is None:
        logger.warning("Rebuild plan: skipping upload with no PostgreSQL catalog row")
        return None, "catalog_row_missing"
    if catalog_row.status not in db_documents.ACTIVE_STATUSES and catalog_row.status != "pending":
        logger.warning("Rebuild plan: skipping upload with an unrecognized catalog lifecycle status")
        return None, "catalog_status_invalid"
    if str(catalog_row.owner_user_id) != sidecar["owner_user_uuid"]:
        logger.warning("Rebuild plan: skipping upload whose catalog owner disagrees with its sidecar")
        return None, "catalog_owner_mismatch"
    if catalog_row.content_sha256 != actual_content_sha256:
        logger.warning("Rebuild plan: skipping upload whose catalog content hash disagrees with its actual content")
        return None, "catalog_hash_mismatch"
    if catalog_row.stored_name != sidecar["stored_name"] or catalog_row.id != document_uuid:
        logger.warning("Rebuild plan: skipping upload whose catalog identity disagrees with its sidecar/physical file")
        return None, "catalog_identity_mismatch"
    if catalog_row.display_name != sidecar["display_name"]:
        # Stage 5C corrective pass #3 (Blocker 4): this check was
        # previously missing here — the exact same identity contract
        # normal ingestion's own final verification requires (Stage
        # 5C corrective pass #2) must also hold before rebuild ever
        # indexes a private document into the active Qdrant
        # collection. Codex reproduced a catalog row and sidecar
        # disagreeing on display_name (e.g. "DIFFERENT-CATALOG-NAME.txt"
        # vs "sidecar-name.txt") with rebuild continuing to accept it.
        logger.warning("Rebuild plan: skipping upload whose catalog display_name disagrees with its sidecar")
        return None, "catalog_display_name_mismatch"

    return SourceDocument(
        kind="upload",
        document_id=sidecar["document_id"],
        display_source=sidecar["display_name"],
        physical_path=physical_path,
        content_sha256=actual_content_sha256,
        stored_name=sidecar["stored_name"],
        content_bytes=source_bytes,
        owner_user_uuid=sidecar["owner_user_uuid"],
        document_uuid=document_uuid,
    ), None


def _plan_upload_documents(uploads_dir: Path, skipped_reasons: List[str]) -> List[SourceDocument]:
    uploads_dir = Path(uploads_dir)
    if not uploads_dir.exists():
        return []
    uploads_root = uploads_dir.resolve()

    planned = []
    for physical_path in sorted(uploads_dir.iterdir()):
        doc, skip_reason = _validate_upload_candidate(physical_path, uploads_dir, uploads_root)
        if doc is not None:
            planned.append(doc)
        elif skip_reason is not None:
            skipped_reasons.append(skip_reason)
    return planned


def build_plan(
    documents_dir: Path,
    uploads_dir: Path,
    reference_filenames: Optional[Sequence[str]] = BUILTIN_REFERENCE_FILES,
) -> RebuildPlan:
    """
    Enumerate and validate every source document. Purely local filesystem
    reads + hashing — makes NO provider call and NO Qdrant call, so it is
    always safe to run (this is exactly what dry-run mode calls, and apply
    mode calls it first too, before doing anything destructive).

    Args:
        reference_filenames: passed through to
            rag.loader.list_builtin_reference_files() as the manifest
            (Stage 2B-C Blocker 5). Defaults to config.BUILTIN_REFERENCE_FILES
            — mirrors VectorIndex.index_documents_directory()'s own default,
            so a caller who simply omits this argument still gets EXACTLY
            the built-in manifest, never an unconstrained scan (Stage 2B-C1
            correction: omitting it previously fell back to
            list_source_files(), silently restoring unconstrained scanning).
            Pass explicit `None` to opt into that unconstrained
            extension-based scan of `documents_dir` instead — this bypasses
            the manifest gate entirely and exists ONLY for low-level tests
            exercising generic directory-scan/reconciliation mechanics
            against a synthetic corpus; production code must never do this.
    """
    plan = RebuildPlan()
    plan.reference_documents = _plan_reference_documents(documents_dir, reference_filenames)
    plan.upload_documents = _plan_upload_documents(uploads_dir, plan.skipped_upload_reasons)
    return plan


def apply_plan(plan: RebuildPlan, vector_index) -> RebuildReport:
    """
    Reconcile Qdrant to exactly match `plan` — Stage 2B-C Blocker 3:
    NEVER clears the collection first. Each desired document is safely
    reconciled via VectorIndex.reconcile_document() (embed-then-upsert-
    then-delete-stale, or a zero-embedding convergence when the document
    is already exactly current — see rag/index.py). If any desired
    document's reconciliation raises, that exception propagates
    immediately: every already-reconciled document AND every untouched
    document remain exactly as they were, and orphan pruning below never
    runs — the existing index stays usable, and a deterministic rerun
    converges. This DOES call the embeddings provider (for genuinely
    new/changed documents) and DOES mutate Qdrant — never call this in
    dry-run mode.

    Orphan logical documents (indexed in Qdrant under a document_id no
    longer present in `plan`) are removed ONLY after every desired
    document above has reconciled successfully — never before, and never
    partially.

    Stage 5C corrective pass #4 (Blocker 7): PLANNING IS NEVER
    AUTHORITATIVE for this destructive/activating operation. Each "upload"
    document in `plan` is freshly revalidated — via the exact same
    `_validate_upload_candidate()` `build_plan()` itself used — immediately
    before it is reconciled, rather than trusting its plan-time
    `content_bytes`/`content_sha256` snapshot as still-accurate: an
    independent audit reproduced a physical file mutating between plan and
    apply (an operator can review a dry run for an arbitrary amount of time
    before applying it) with the STALE plan-time bytes still getting
    indexed/activated, because reconcile_document()'s own hash check
    compared that same stale snapshot against itself. A candidate that no
    longer revalidates identically is skipped this run (never reconciled/
    activated from stale data) but its document_id is still added to
    `desired_document_ids` below — fail closed means PRESERVING its
    existing Qdrant points, never treating "could not confirm it's still
    valid this run" as "it's gone, prune it".
    """
    report = RebuildReport()
    desired_document_ids = set()
    incomplete_plan = CATALOG_UNREACHABLE_REASON in plan.skipped_upload_reasons
    for doc in plan.all_documents:
        if doc.kind == "upload":
            uploads_root = doc.physical_path.parent.resolve()
            fresh_doc, skip_reason = _validate_upload_candidate(doc.physical_path, doc.physical_path.parent, uploads_root)
            if fresh_doc is None or fresh_doc != doc:
                logger.warning(
                    "Rebuild apply: skipping upload whose durable state no longer matches the "
                    "plan (mutated, removed, or no longer valid since planning) | skip_reason=%s",
                    skip_reason,
                )
                report.uploads_skipped_stale_at_apply += 1
                # Never orphan-prune a document we simply couldn't
                # reconfirm this run — preserve its existing Qdrant state.
                desired_document_ids.add(doc.document_id)
                continue
            doc = fresh_doc
        desired_document_ids.add(doc.document_id)
        status, chunk_count = vector_index.reconcile_document(
            doc.document_id,
            doc.physical_path,
            display_name=doc.display_source,
            stored_name=doc.stored_name,
            expected_content_sha256=doc.content_sha256,
            source_bytes=doc.content_bytes,
            owner_user_uuid=doc.owner_user_uuid,
        )
        report.documents_reconciled += 1
        if status == "reindexed":
            report.documents_reindexed += 1
            report.chunks_reindexed += chunk_count
        elif status == "emptied":
            report.documents_emptied += 1

        # Stage 5C corrective pass (Section 3/5): Qdrant reconciliation for
        # this exact document has just genuinely succeeded — re-affirm (or,
        # for a still-'pending' row, finally confirm) its catalog status.
        # Deliberately AFTER reconcile_document() succeeded, never before:
        # if reconciliation had raised instead, this line is never reached
        # and the catalog row is left exactly as the plan-time gate found
        # it, safe to retry on a rerun. mark_active_sync() is idempotent
        # against an already-'active' row (see its own docstring), so this
        # is also a harmless no-op for a document that was already active.
        #
        # Stage 5C corrective pass #5 (Blocker 2): chunk_count == 0 means
        # this reconciliation produced NO indexed content for this
        # document (an empty/whitespace-only source reconciles to
        # ("unchanged", 0) if nothing was ever indexed for it, or —
        # Stage 5C corrective pass #6, Blocker 2 — ("emptied", 0) if
        # PREVIOUSLY-indexed points for it just got removed; see
        # VectorIndex.reconcile_document()) — never newly activate a
        # catalog row on the strength of that. Checked BEFORE calling
        # mark_active_sync() at all, mirroring live ingestion's own
        # EmptyDocumentError gate in app.documents._load_and_index_document().
        # An already-'active' row reconciled to "emptied" is deliberately
        # left at 'active' here too (see this pass's lifecycle choice, this
        # module's own docstring) — mark_active_sync() is simply never
        # called either way when chunk_count == 0, so an active row's
        # status is untouched by this branch regardless.
        if doc.kind == "upload" and doc.document_uuid is not None:
            if chunk_count == 0:
                logger.warning(
                    "Rebuild apply: skipping catalog activation — document reconciled with zero chunks, "
                    "never newly activated with no indexed Qdrant content"
                )
                report.documents_skipped_zero_chunks += 1
            else:
                try:
                    import db.documents as db_documents
                    db_documents.mark_active_sync(document_id=doc.document_uuid)
                    report.documents_catalog_activated += 1
                except Exception as e:
                    logger.warning(
                        "Rebuild apply: Qdrant reconciliation succeeded but catalog activation failed "
                        "(row may remain 'pending' — safe to rerun) | error_type=%s",
                        type(e).__name__,
                    )
                    report.documents_catalog_activation_failed += 1

    # Stage 5C corrective pass #4 (Blocker 9): orphan pruning is a GLOBAL
    # judgment ("every Qdrant document_id not in desired_document_ids is
    # gone from source") — it can only ever be safe if desired_document_ids
    # is a COMPLETE, authoritative set. A PostgreSQL outage during planning
    # means AT LEAST ONE private upload could not be validated and was
    # therefore omitted from the plan for a reason that says nothing about
    # whether it still genuinely exists — treating that omission as
    # "confirmed gone" is exactly the destructive-on-uncertainty defect an
    # independent audit reproduced (existing private Qdrant points deleted
    # merely because the catalog was briefly unreachable while planning).
    # "Could not establish state" must never be treated as "state absent"
    # (Principle 5) — pruning is skipped ENTIRELY in that case; every
    # document that COULD be individually validated was still reconciled
    # normally above (an invalid individual document, examined and found
    # invalid, still excludes that ONE document — only the GLOBAL pruning
    # judgment is gated on the plan's completeness).
    if incomplete_plan:
        report.orphan_pruning_skipped_incomplete_plan = True
        logger.warning(
            "Rebuild apply: skipping orphan pruning — the PostgreSQL catalog was unreachable while "
            "planning at least one private upload, so the desired-document set cannot be trusted as "
            "complete/authoritative this run (existing Qdrant state is left untouched)"
        )
    else:
        indexed_document_ids = vector_index.list_document_ids()
        orphan_ids = indexed_document_ids - desired_document_ids
        for orphan_id in orphan_ids:
            vector_index.delete_document(orphan_id)
            report.documents_removed += 1

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.rebuild_qdrant",
        description=(
            "Rebuild the Qdrant knowledge base index from source documents "
            "(built-in Markdown reference documents + sidecar-backed managed "
            "uploads). Never reads legacy Chroma. Must be run as a module "
            "from the repository root — `python scripts/rebuild_qdrant.py` "
            "directly does not work; use `python -m scripts.rebuild_qdrant`."
        ),
    )
    parser.add_argument(
        "--apply", action="store_true",
        help=(
            "Actually reconcile Qdrant to source (non-destructive — see module "
            "docstring). Without this flag, performs a safe dry run only: no "
            "provider calls, no Qdrant mutation."
        ),
    )
    args = parser.parse_args(argv)

    # Stage 2B-D Blocker 3: DOCUMENTS_DIR/MANAGED_UPLOADS_DIR are pure Path
    # constants (no secret, no env-driven value) — imported from
    # rag.constants, never from the credential-validating `config` module,
    # so this line runs identically whether or not --apply was passed and
    # never requires TELEGRAM_BOT_TOKEN/OPENAI_API_KEY/ANTHROPIC_API_KEY.
    from rag.constants import DOCUMENTS_DIR, MANAGED_UPLOADS_DIR

    plan = build_plan(DOCUMENTS_DIR, MANAGED_UPLOADS_DIR, reference_filenames=BUILTIN_REFERENCE_FILES)

    print(f"Built-in reference documents found: {len(plan.reference_documents)}")
    print(f"Managed uploads with valid sidecars found: {len(plan.upload_documents)}")
    if plan.skipped_upload_reasons:
        print(f"Managed uploads skipped (missing/invalid sidecar/containment): {len(plan.skipped_upload_reasons)}")

    if not args.apply:
        print("\nDry run only: no embeddings were generated, no Qdrant mutation was performed.")
        print("Pass --apply to actually reconcile the Qdrant collection to source (non-destructive).")
        return 0

    print("\nApplying rebuild: reconciling Qdrant to source (this calls the embeddings provider for new/changed documents)...")
    from rag.index import VectorIndex
    vector_index = VectorIndex()
    try:
        report = apply_plan(plan, vector_index)
    finally:
        vector_index.close()
    print(
        f"Rebuild complete: {report.documents_reconciled} documents reconciled "
        f"({report.documents_reindexed} re-embedded, {report.chunks_reindexed} chunks), "
        f"{report.documents_removed} orphan document(s) removed, "
        f"{report.documents_catalog_activated} catalog row(s) activated/reaffirmed."
    )
    if report.documents_catalog_activation_failed:
        print(
            f"WARNING: {report.documents_catalog_activation_failed} document(s) reconciled into "
            "Qdrant successfully but their catalog row could not be activated — rerun this command "
            "to retry reconciliation (see logs for details)."
        )
    if report.documents_skipped_zero_chunks:
        print(
            f"NOTE: {report.documents_skipped_zero_chunks} document(s) reconciled with ZERO chunks and were "
            "NOT activated — an empty/whitespace-only document has nothing to index; fix its content and "
            "rerun this command."
        )
    if report.documents_emptied:
        print(
            f"NOTE: {report.documents_emptied} document(s) that previously had indexed content now reconciled "
            "with ZERO chunks — their stale Qdrant points were removed (any catalog row involved was left at "
            "its current status; fix the document's content and rerun this command to re-index it)."
        )
    if report.uploads_skipped_stale_at_apply:
        print(
            f"NOTE: {report.uploads_skipped_stale_at_apply} upload(s) skipped this run — their "
            "durable state no longer matched the plan by apply time (mutated, removed, or no "
            "longer valid since planning) — rerun this command to reconcile them against their "
            "current state."
        )
    if report.orphan_pruning_skipped_incomplete_plan:
        print(
            "NOTE: orphan pruning was SKIPPED this run — planning could not confirm the complete "
            "private-upload set (the PostgreSQL catalog was unreachable for at least one candidate). "
            "Existing Qdrant state was left untouched rather than risk deleting a document the "
            "outage merely hid from planning; rerun once the catalog is reachable to resume pruning."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
