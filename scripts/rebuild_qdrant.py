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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

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
    # Stage 3A: the sidecar's `owner_user_id` for an "upload" document,
    # sourced from the sidecar itself (never guessed) — always None for a
    # "reference" document (no owner). apply_plan() passes this straight to
    # VectorIndex.reconcile_document(), which derives scope="private" (with
    # this owner) vs. scope="reference" from it exactly the same way the
    # live upload path does.
    owner_user_id: Optional[int] = None


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


def _plan_upload_documents(uploads_dir: Path, skipped_reasons: List[str]) -> List[SourceDocument]:
    uploads_dir = Path(uploads_dir)
    if not uploads_dir.exists():
        return []
    uploads_root = uploads_dir.resolve()

    planned = []
    for physical_path in sorted(uploads_dir.iterdir()):
        if not physical_path.is_file():
            continue
        if physical_path.name.endswith(".meta.json"):
            continue
        if physical_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

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
            skipped_reasons.append("path_containment_violation")
            logger.warning("Rebuild plan: skipping upload whose sidecar path is a symlink | document_extension=%s", physical_path.suffix.lower())
            continue
        if not candidate_sidecar_path.is_file():
            skipped_reasons.append("missing_sidecar")
            logger.warning("Rebuild plan: skipping upload with no sidecar | document_extension=%s", physical_path.suffix.lower())
            continue

        try:
            sidecar_path = resolve_sidecar_path(uploads_dir, physical_path)
        except PathContainmentError as e:
            skipped_reasons.append("path_containment_violation")
            logger.warning("Rebuild plan: skipping upload whose sidecar failed path containment | error_type=%s", type(e).__name__)
            continue

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
            skipped_reasons.append("path_containment_violation")
            logger.warning("Rebuild plan: skipping upload whose sidecar failed secure read (possible race) | error_type=%s", type(e).__name__)
            continue

        try:
            sidecar = parse_sidecar_bytes(sidecar_bytes)
        except SidecarError as e:
            skipped_reasons.append("invalid_sidecar")
            logger.warning("Rebuild plan: skipping upload with invalid sidecar | error_type=%s", type(e).__name__)
            continue

        # Stage 3A: a legacy (pre-Stage-3A, schema_version=1) sidecar has
        # no recorded owner — parse_sidecar_bytes() represents that as an
        # explicit `owner_user_id: None` rather than raising. Fail closed
        # here rather than guessing: never reconcile it as a private
        # document owned by nobody-in-particular, and never silently
        # promote it to scope="reference" (which would make it visible to
        # every authorized user). Full operator-facing handling of legacy
        # uploads (e.g. an explicit reassignment/migration path) is Stage
        # 3B; this pass only needs to make sure one is never silently
        # exposed.
        if sidecar["owner_user_id"] is None:
            skipped_reasons.append("missing_owner")
            logger.warning("Rebuild plan: skipping upload with no recorded owner (legacy sidecar)")
            continue

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
            skipped_reasons.append("path_containment_violation")
            logger.warning("Rebuild plan: skipping upload that failed path containment | error_type=%s", type(e).__name__)
            continue

        expected_document_id = upload_document_id(physical_path.stem)
        if sidecar["document_id"] != expected_document_id or sidecar["stored_name"] != physical_path.name:
            # Sidecar content doesn't match the physical file it's paired
            # with (e.g. hand-edited, or copied alongside the wrong file).
            # Fail closed: skip rather than guess which one is right.
            skipped_reasons.append("sidecar_identity_mismatch")
            logger.warning("Rebuild plan: skipping upload with sidecar/physical-file identity mismatch")
            continue

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
            skipped_reasons.append("path_containment_violation")
            logger.warning("Rebuild plan: skipping upload whose source failed secure read (possible race) | error_type=%s", type(e).__name__)
            continue

        actual_content_sha256 = sha256_hex(source_bytes)
        if actual_content_sha256 != sidecar["content_sha256"]:
            skipped_reasons.append("sidecar_content_hash_mismatch")
            logger.warning("Rebuild plan: skipping upload whose content no longer matches its sidecar hash")
            continue

        planned.append(SourceDocument(
            kind="upload",
            document_id=sidecar["document_id"],
            display_source=sidecar["display_name"],
            physical_path=physical_path,
            content_sha256=actual_content_sha256,
            stored_name=sidecar["stored_name"],
            content_bytes=source_bytes,
            owner_user_id=sidecar["owner_user_id"],
        ))
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
    """
    report = RebuildReport()
    desired_document_ids = set()
    for doc in plan.all_documents:
        desired_document_ids.add(doc.document_id)
        status, chunk_count = vector_index.reconcile_document(
            doc.document_id,
            doc.physical_path,
            display_name=doc.display_source,
            stored_name=doc.stored_name,
            expected_content_sha256=doc.content_sha256,
            source_bytes=doc.content_bytes,
            owner_user_id=doc.owner_user_id,
        )
        report.documents_reconciled += 1
        if status == "reindexed":
            report.documents_reindexed += 1
            report.chunks_reindexed += chunk_count

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
        f"{report.documents_removed} orphan document(s) removed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
