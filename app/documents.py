"""
Document Ingestion Transaction (Stage 5B application boundary).

Extracted from handlers/document_upload.py: the hardened storage/index
transaction underneath the Telegram upload flow never depended on
telebot message/update objects or a Telegram bot instance — only
process_document_upload()'s progress/result messaging did. This module
owns the adapter-independent sequence (validate -> managed storage ->
secure read -> hash -> parse/index -> rollback/cleanup on failure ->
cancellation resolution) and returns a structured DocumentIngestResult;
handlers/document_upload.py now only translates Telegram input into the
call below and the result back into Telegram messages.

Every accepted Stage 1-3 guarantee is preserved unchanged here: upload
size limit, allowed-extension policy, exclusive writes, partial-file
cleanup, symlink/path safety, single-open/TOCTOU-resistant secure read,
content hashing, sidecar validation, Qdrant rollback/replacement,
cancellation, executor-thread ownership, and privacy-safe logging. See
each function's own docstring (carried over verbatim from
handlers/document_upload.py) for the specific threat/invariant it closes.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Tuple

import db.documents as db_documents
from config import MANAGED_UPLOADS_DIR, MAX_DOCUMENT_SIZE_BYTES
from rag.identity import sha256_hex, upload_document_id
# document_loader itself is no longer called directly from this module
# (Stage 2B-F: rag.index.reconcile_document() calls it internally, on a
# private secured snapshot — see _load_and_index_document()) — kept
# imported here anyway because it is the shared DocumentLoader SINGLETON
# also used by rag.index, and existing tests reach it for monkeypatching
# via this module's own `documents.document_loader` reference.
from rag.loader import document_loader, SUPPORTED_EXTENSIONS
from rag.safe_files import read_regular_file_secure
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
from rag.index import SourceMutatedError, get_vector_index, is_index_unavailable_error
from utils.logging import logger
from utils.helpers import cleanup_file, submit_worker, await_worker


class EmptyDocumentError(ValueError):
    """
    Raised by _load_and_index_document() when a document parses/chunks into
    ZERO meaningful chunks (Stage 5C corrective pass #4, Blocker 10). An
    independent audit reproduced an empty (or effectively empty) upload
    becoming a successful, ACTIVE catalog document with zero Qdrant points
    — success must mean a useful, internally consistent document actually
    exists (Principle 6). Treated exactly like any other indexing-region
    failure by ingest_document()'s caller: full compensating cleanup
    (Qdrant delete_document — a safe no-op here, since nothing was ever
    written — plus the physical file, sidecar, and the still-'pending'
    catalog row), reported as success=False. Deliberately a fixed, safe
    message; never embeds the document content or path.
    """


class SidecarConsistencyError(ValueError):
    """
    Raised by _load_and_index_document() when the durable v3 sidecar's
    CURRENT content — freshly re-read and validated immediately before
    catalog activation — no longer matches the exact identity/state
    contract this upload was stored with (Stage 5C corrective pass #5,
    Blocker 1).

    An independent audit reproduced live ingestion checking the physical
    file and the PostgreSQL catalog at the final success boundary but
    NEVER re-validating the durable sidecar itself: the sidecar was
    mutated (a different owner/display_name/content_sha256, or replaced
    with malformed content) after storage while the physical file was left
    completely unchanged, and ingestion still reported success — leaving
    an active catalog row + indexed Qdrant content + unchanged physical
    file that all agreed with each other, but disagreed with the durable
    sidecar. Treated exactly like any other indexing-region failure by
    ingest_document()'s caller: full compensating cleanup (Qdrant
    delete_document, physical file, sidecar, and the still-'pending'
    catalog row), reported as success=False. Deliberately a fixed, safe
    message; never embeds the sidecar's content or any path.
    """


@dataclass(frozen=True)
class StoredUpload:
    """Result of a successfully completed storage step: the physical file,
    its durable sidecar, and the identity/fingerprint values derived while
    creating them — everything the later load/index step and any cleanup
    path need, without re-deriving or re-reading anything.

    owner_user_id (Stage 5C): the canonical internal user UUID, resolved
    by the Telegram adapter via app/identity.py BEFORE this module is ever
    called — captured once, at storage time, by
    `_store_document_exclusively()` — already durably persisted in the
    sidecar (as `owner_user_uuid`) and the PostgreSQL documents catalog by
    that point. Carried here so `_load_and_index_document()` can pass it
    straight to `VectorIndex.reconcile_document()` without re-deriving or
    re-reading it from anywhere.

    document_uuid (Stage 5C): the upload's own storage UUID (the physical
    filename's stem, parsed as a uuid.UUID) — the SAME value used as the
    PostgreSQL `documents.id` primary key and embedded in the RAG
    document_id string (`upload_document_id()` just prefixes it with
    "upload:"). Carried here so DB catalog calls never need to re-parse
    `physical_path.stem` at each of their call sites."""
    physical_path: Path
    sidecar_path: Path
    document_id: str
    document_uuid: uuid.UUID
    content_sha256: str
    owner_user_id: uuid.UUID


@dataclass(frozen=True)
class DocumentIngestResult:
    """Adapter-independent outcome of ingest_document() — no Telegram
    types, safe for any adapter (Telegram today, a future FastAPI adapter
    later) to interpret directly.

    Exactly one of these shapes applies:
    - success=True: the upload is durably stored and indexed.
      chunk_count/stored/file_size_bytes are populated.
    - success=False, rejected_reason="unsupported_extension": rejected
      before any download/storage was attempted (no owner_user_id-scoped
      side effect occurred).
    - success=False, rejected_reason="oversized": rejected against the
      actual downloaded bytes, before any disk write; file_size_bytes is
      the actual size for a caller that wants to report it.
    - success=False, rejected_reason=None, error_type set: storage or
      indexing raised. Already rolled back via the same cleanup path as
      before this extraction — cleanup_complete reports whether every
      cleanup component (Qdrant delete, physical file, sidecar) actually
      completed (see _cleanup_new_upload()).
      failure_reason (Stage 7A-3 corrective pass) is
      INGEST_FAILURE_KNOWLEDGE_BASE_UNAVAILABLE ONLY when the underlying
      exception was a genuine Qdrant/index availability failure (see
      rag.index.is_index_unavailable_error() for the exact, narrow
      taxonomy) — None for every other storage/parse/embedding/catalog
      failure, which an adapter must treat as a generic processing failure.
    """
    success: bool
    chunk_count: Optional[int] = None
    stored: Optional[StoredUpload] = None
    file_size_bytes: int = 0
    rejected_reason: Optional[str] = None
    error_type: Optional[str] = None
    cleanup_complete: Optional[bool] = None
    failure_reason: Optional[str] = None


# DocumentIngestResult.failure_reason value for a genuine Qdrant/index
# availability failure during ingestion (Stage 7A-3 corrective pass) — a
# fixed, safe structured code, never exception text.
INGEST_FAILURE_KNOWLEDGE_BASE_UNAVAILABLE = "knowledge_base_unavailable"


def _store_document_exclusively(
    file_bytes: bytes, extension: str, display_name: str, owner_user_id: uuid.UUID, attempts: int = 5
) -> StoredUpload:
    """
    Atomically claim a fresh, opaque, application-generated storage path,
    write the document bytes into it in the same exclusive-create
    operation, then write its durable `.meta.json` sidecar (Stage 2B) and
    its PostgreSQL catalog row (Stage 5C, status='pending') — the physical
    file and its sidecar together remain the durable CONTENT source of
    truth for this upload, independent of whatever is or isn't currently
    in Qdrant; the catalog row is the durable identity/ownership/lifecycle
    record alongside them.

    `owner_user_id` (Stage 5C): the canonical internal user UUID (resolved
    by the Telegram adapter via app/identity.py before this module is ever
    called), persisted into both the sidecar (`owner_user_uuid`, via
    `build_sidecar()`) and the PostgreSQL `documents` row so ownership
    survives a process restart and is available independently of any
    Telegram session state — see rag/sidecar.py and db/documents.py.

    Each candidate is opened with 'xb' (O_CREAT | O_EXCL). Ownership
    boundary: a FileExistsError from that open call means nothing was
    created, so it's safe to just retry with a new UUID without touching
    whatever already occupies that path. Once the open call itself
    succeeds, this call is the sole owner of `candidate` — no other upload
    could have created it — so any later failure (write, close, or sidecar
    creation) triggers best-effort cleanup of exactly that file before the
    original exception is re-raised, never masked by a cleanup failure.
    Sidecar creation failing after the physical write succeeded must not
    leave a silently-unrebuildable upload behind, so it too triggers
    cleanup of the physical file.
    """
    MANAGED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    last_collision_error = None
    for _ in range(attempts):
        candidate = MANAGED_UPLOADS_DIR / f"{uuid.uuid4().hex}{extension}"
        try:
            handle = open(candidate, 'xb')
        except FileExistsError as e:
            last_collision_error = e
            continue

        try:
            with handle:
                handle.write(file_bytes)
        except Exception as e:
            # Ownership of `candidate` was established by the open() above,
            # so cleaning it up here can never remove another attempt's
            # file. cleanup_file() already swallows its own errors, so this
            # cannot mask the original write/close exception re-raised
            # below.
            #
            # Stage 5C corrective pass #2 (Section 4): the REAL outcome of
            # this cleanup — not just "an attempt was made" — is attached
            # to the exception itself. ingest_document()'s except-block has
            # no StoredUpload to hand to _cleanup_new_upload() at this
            # point (this function never got to construct/return one), so
            # without this it would have no way to know cleanup actually
            # failed and would default to reporting `cleanup_complete=True`
            # even though the partially-written file is still on disk. The
            # original exception type/message is preserved unchanged (a
            # plain attribute, not a wrapper) — existing callers that match
            # on the exact exception type/text are unaffected.
            e.partial_storage_cleanup_complete = cleanup_file(candidate)
            raise

        document_id = upload_document_id(candidate.stem)
        document_uuid = uuid.UUID(candidate.stem)
        content_sha256 = sha256_hex(file_bytes)
        sidecar_path = sidecar_path_for(candidate)
        try:
            write_sidecar_atomic(
                sidecar_path,
                build_sidecar(
                    document_id=document_id,
                    display_name=display_name,
                    stored_name=candidate.name,
                    content_sha256=content_sha256,
                    owner_user_uuid=str(owner_user_id),
                ),
            )
        except Exception as e:
            # Same rationale as the write-failure branch above: attach the
            # real cleanup outcome, never let the caller default to
            # cleanup_complete=True just because it has no StoredUpload.
            e.partial_storage_cleanup_complete = cleanup_file(candidate)
            raise

        try:
            db_documents.create_pending_sync(
                document_id=document_uuid,
                owner_user_id=owner_user_id,
                stored_name=candidate.name,
                display_name=display_name,
                content_sha256=content_sha256,
            )
        except Exception as e:
            # By this point the sidecar already exists durably (unlike the
            # write-failure branches above, which run before it does) — a
            # catalog-insert failure must clean up the physical file, the
            # sidecar, AND (Stage 5C corrective pass #3, Blocker 3) the
            # PostgreSQL catalog row itself: create_pending_sync()'s own
            # session.commit() can, in principle, durably commit its
            # INSERT and still raise back to this except-block (e.g. the
            # acknowledgement is lost) — this call has no StoredUpload to
            # hand to the normal cleanup path in that case, and must never
            # assume "the insert failed, so nothing to clean up in the DB".
            # reconcile_ambiguous_create_pending_sync() only ever removes a
            # row that exactly matches what THIS call was attempting to
            # insert (never a blind delete-by-id), so a genuinely
            # pre-existing, unrelated/mismatching row is reported as an
            # incomplete cleanup rather than silently destroyed. Complete
            # only if every one of the three artifacts is actually gone
            # afterward.
            physical_removed = cleanup_file(candidate)
            sidecar_removed = cleanup_file(sidecar_path)
            db_removed = db_documents.reconcile_ambiguous_create_pending_sync(
                document_id=document_uuid,
                owner_user_id=owner_user_id,
                stored_name=candidate.name,
                display_name=display_name,
                content_sha256=content_sha256,
            )
            e.partial_storage_cleanup_complete = physical_removed and sidecar_removed and db_removed
            raise

        return StoredUpload(
            physical_path=candidate,
            sidecar_path=sidecar_path,
            document_id=document_id,
            document_uuid=document_uuid,
            content_sha256=content_sha256,
            owner_user_id=owner_user_id,
        )

    raise RuntimeError("Could not allocate a unique document storage path") from last_collision_error


def _validate_current_sidecar_matches(stored: StoredUpload, display_name: str) -> None:
    """
    Freshly re-read and validate the durable v3 sidecar against the exact
    identity/state contract `stored` was written with (Stage 5C corrective
    pass #5, Blocker 1) — see SidecarConsistencyError's own docstring for
    the defect this closes.

    Reuses the SAME hardened containment/secure-read/schema-validation
    primitives rebuild (scripts/rebuild_qdrant.py's
    `_validate_upload_candidate()`) and migration
    (scripts/migrate_sidecars_v2_to_v3.py's `_validate_candidate()`) already
    use — `resolve_sidecar_path()` (containment, never a symlink),
    `secure_read_sidecar_bytes()` (single secure open, no TOCTOU reopen),
    and `parse_sidecar_bytes()` (full schema validation) — never a second,
    weaker sidecar-parsing implementation of its own.

    Checks the COMPLETE expected sidecar contract, not just the content
    hash: schema_version (must still be the current v3 shape),
    document_id, owner_user_uuid, stored_name, display_name, and
    content_sha256 must all still agree with what `stored` (captured once,
    at storage time — see `_store_document_exclusively()`) and this
    ingestion's own `display_name` argument say they should be. Any
    containment/secure-read/schema failure, or any single field
    disagreeing, raises SidecarConsistencyError — the caller
    (`_load_and_index_document()`) treats this exactly like any other
    indexing-region failure: no success is ever reported, and the caller's
    existing exception path rolls everything back (Qdrant points, the
    still-'pending' catalog row, the physical file, and the sidecar) via
    `_cleanup_new_upload()`.
    """
    try:
        sidecar_path = resolve_sidecar_path(MANAGED_UPLOADS_DIR, stored.physical_path)
        sidecar_bytes = secure_read_sidecar_bytes(MANAGED_UPLOADS_DIR, sidecar_path)
        sidecar = parse_sidecar_bytes(sidecar_bytes)
    except (PathContainmentError, SidecarError) as e:
        raise SidecarConsistencyError(stored.document_id) from e

    if (
        sidecar.get("schema_version") != 3
        or sidecar.get("document_id") != stored.document_id
        or sidecar.get("owner_user_uuid") != str(stored.owner_user_id)
        or sidecar.get("stored_name") != stored.physical_path.name
        or sidecar.get("display_name") != display_name
        or sidecar.get("content_sha256") != stored.content_sha256
    ):
        raise SidecarConsistencyError(stored.document_id)


def _load_and_index_document(
    stored: StoredUpload,
    display_name: str,
    *,
    _test_post_index_hook: Optional[Callable[[], None]] = None,
) -> int:
    """
    Runs document parsing (PyPDFLoader/TextLoader/Docx2txtLoader, CPU/disk
    bound) and Qdrant indexing (OpenAIEmbeddings network call + vector-store
    write) as a single blocking unit, so the whole pipeline can be offloaded
    to a worker thread in one `asyncio.to_thread()` call rather than
    thread-hopping per step. No async/Telegram/UserSession work happens
    between these two calls in the original code, so combining them changes
    no ordering or behavior.

    Stage 2B-F (Codex-disclosed residual gap from Stage 2B-E's report):
    the previous implementation here re-hashed `stored.physical_path` both
    BEFORE and immediately AFTER `document_loader.load_document()`
    reopened that same pathname a second time to actually parse it — the
    exact "hash one object / parse another" TOCTOU shape Stage 2B-E closed
    for rebuild, still present here for immediate upload-time indexing.
    `stored.physical_path` is now read exactly ONCE, through the same
    secure same-object primitive rebuild uses
    (`rag.safe_files.read_regular_file_secure()`, rooted at
    `MANAGED_UPLOADS_DIR`) — proving the bytes obtained come from the
    identical filesystem object that was validated immediately
    beforehand, never a pathname reopen. Those exact bytes are then handed
    to `VectorIndex.reconcile_document(..., source_bytes=...)` (the same
    Stage 2B-E plan-to-apply snapshot mechanism rebuild already uses,
    reused here rather than duplicated): it re-derives the hash from
    those bytes and requires it to match `stored.content_sha256` (the hash
    recorded in the durable sidecar at storage time) BEFORE any Qdrant
    mutation, and the document loader parses a private temporary snapshot
    written from those exact bytes — never `stored.physical_path` a second
    time. A hash mismatch or a detected same-object-identity race both
    raise (SourceMutatedError / rag.safe_files.SecureReadError
    respectively) with NO Qdrant mutation, handled exactly like any other
    indexing failure by the caller (cleanup via `_cleanup_new_upload()`).
    For a brand-new upload (no prior points under this document_id),
    reconcile_document() always classifies as "reindexed" — functionally
    identical to the previous direct `add_documents()` call.
    """
    secure_bytes = read_regular_file_secure(stored.physical_path, root=MANAGED_UPLOADS_DIR)

    _status, chunk_count = get_vector_index().reconcile_document(
        stored.document_id,
        stored.physical_path,
        display_name=display_name,
        stored_name=stored.physical_path.name,
        expected_content_sha256=stored.content_sha256,
        source_bytes=secure_bytes,
        owner_user_uuid=str(stored.owner_user_id),
    )

    # Stage 5C corrective pass #4 (Blocker 10): a document that parses/
    # chunks into ZERO meaningful chunks must never become an active
    # catalog document with nothing actually indexed for it. Checked BEFORE
    # mark_active_sync() — the still-'pending' row and the (never-written)
    # Qdrant points are then cleaned up exactly like any other indexing
    # failure by ingest_document()'s caller.
    if chunk_count == 0:
        raise EmptyDocumentError(stored.document_id)

    if _test_post_index_hook is not None:
        # Test-only seam (mirrors rag.safe_files.read_regular_file_secure()'s
        # own `_test_pre_open_hook` convention) — called ONLY so a test can
        # deterministically mutate the physical file in the exact window
        # this function's own final revalidation below exists to close.
        # Every real caller leaves this None, making it a complete no-op in
        # production.
        _test_post_index_hook()

    # Stage 5C corrective pass #4 (Blocker 5): `secure_bytes` above is a
    # SINGLE secure snapshot of the physical file, taken once at the top of
    # this function — reconcile_document() indexed exactly those bytes
    # (never reopening the file), so Qdrant is now provably consistent with
    # THAT snapshot. But a snapshot only proves the file's content as of
    # the moment it was read; a concurrent rewrite of the physical file
    # landing AFTER that read (and before this ingestion actually commits)
    # would leave Qdrant/sidecar/catalog all self-consistently describing
    # the OLD snapshot while the DURABLE FILE ON DISK silently disagrees
    # with all three. Success must never be reported in that state
    # (Principle 1: durable source state must be bound to the exact bytes
    # that were indexed and committed). One more secure read, immediately
    # before activation, re-verifies the physical file's CURRENT hash still
    # matches the exact snapshot that was actually indexed; any
    # disagreement raises SourceMutatedError here, BEFORE mark_active_sync()
    # ever runs, so the catalog row is never activated on stale/
    # inconsistent state — the caller's existing exception path then rolls
    # everything back (Qdrant points, the still-'pending' row, the file,
    # and the sidecar) via _cleanup_new_upload().
    revalidation_bytes = read_regular_file_secure(stored.physical_path, root=MANAGED_UPLOADS_DIR)
    if sha256_hex(revalidation_bytes) != stored.content_sha256:
        raise SourceMutatedError(stored.document_id)

    # Stage 5C corrective pass #5 (Blocker 1): the physical-file
    # revalidation immediately above proves the FILE still matches what was
    # indexed, but says nothing about the durable v3 SIDECAR — a separate
    # durable artifact this application also treats as source-of-truth
    # metadata (see rag/sidecar.py). An independent audit mutated the
    # sidecar's owner_user_uuid/display_name/content_sha256 (or replaced it
    # with malformed content) AFTER storage while leaving the physical file
    # byte-for-byte unchanged — the physical-file check above saw no
    # disagreement, and the catalog check further below never reads the
    # sidecar at all, so ingestion still reported success with an active
    # catalog row + indexed Qdrant content + unchanged file that all agreed
    # with each other but disagreed with the durable sidecar. Freshly
    # re-read and validated here, immediately before activation, reusing
    # the exact same hardened secure-read/schema-validation primitives
    # rebuild/migration already use (rag.sidecar.resolve_sidecar_path() /
    # secure_read_sidecar_bytes() / parse_sidecar_bytes()) rather than a
    # second, weaker check.
    _validate_current_sidecar_matches(stored, display_name)

    # Stage 5C corrective pass (Section 1): flip the catalog row to
    # 'active' now that indexing has genuinely committed to Qdrant, then
    # PROVE the resulting row actually matches what was just ingested.
    # Deliberately NOT best-effort any more: an ingestion operation must
    # never report success while its durable catalog state is missing,
    # still 'pending', or inconsistent with the file/sidecar/Qdrant state
    # it claims to describe. Any failure here (mark_active_sync() raising,
    # or the verification below disagreeing) propagates straight out of
    # this function — it runs inside the same executor-thread worker
    # ingest_document() already treats as one indexing-region unit, so an
    # exception here is handled exactly like a Qdrant/embedding failure by
    # the caller: full compensating cleanup via _cleanup_new_upload()
    # (Qdrant points, physical file, sidecar, and the catalog row itself),
    # then reported as success=False. No open database transaction spans
    # this call and the Qdrant mutation above — mark_active_sync() commits
    # its own short transaction; a failure here triggers explicit
    # compensating cleanup afterward, never a distributed transaction.
    db_documents.mark_active_sync(document_id=stored.document_uuid)

    # Stage 5C corrective pass #2 (Section 3): the verification below must
    # cover EVERY catalog field that identifies the just-ingested durable
    # document — id, owner, stored_name, display_name, content_sha256, and
    # active status. A prior version of this check omitted display_name
    # entirely, so a document could be reported as successfully ingested
    # even with a mismatched PostgreSQL display name.
    record = db_documents.get_sync(document_id=stored.document_uuid)
    if (
        record is None
        or record.id != stored.document_uuid
        or record.status not in db_documents.ACTIVE_STATUSES
        or record.owner_user_id != stored.owner_user_id
        or record.stored_name != stored.physical_path.name
        or record.display_name != display_name
        or record.content_sha256 != stored.content_sha256
    ):
        raise db_documents.CatalogConsistencyError(
            "catalog row does not correspond to the durable document just ingested"
        )

    return chunk_count


def _cleanup_new_upload(stored: StoredUpload) -> bool:
    """
    Full rollback for a BRAND NEW upload that never successfully completed
    indexing: removes the physical file and its sidecar, and best-effort
    removes any Qdrant points that might already have been written for its
    document_id — every one of the three components is attempted
    unconditionally, even if an earlier one failed, so nothing
    silently-unrebuildable or orphaned is left behind. Safe to call even if
    some/all of these were never created.

    Stage 2B-C Section J / Stage 2B-D Section C: returns True ONLY if every
    cleanup component — Qdrant delete, physical-file unlink, sidecar
    unlink, AND (Stage 5C) DB catalog row delete — actually completed.
    Before Stage 2B-D, a filesystem unlink failure was invisible here:
    `utils.helpers.cleanup_file()` swallowed unlink errors and returned
    nothing, so this function could return True even though a durable
    artifact physically remained on disk. It now aggregates
    `cleanup_file()`'s own real per-file outcome (Stage 2B-D Blocker 2)
    alongside the Qdrant and DB outcomes. This function never raises, so
    callers already resolving a failure/cancellation can still safely call
    it unconditionally.
    """
    qdrant_removed = True
    try:
        get_vector_index().delete_document(stored.document_id)
    except Exception as e:
        logger.warning("Document upload cleanup: Qdrant delete_document failed (filesystem cleanup still attempted) | error_type=%s", type(e).__name__)
        qdrant_removed = False

    db_row_removed = True
    try:
        db_documents.delete_sync(document_id=stored.document_uuid)
    except Exception as e:
        logger.warning("Document upload cleanup: catalog row delete failed | error_type=%s", type(e).__name__)
        db_row_removed = False

    physical_removed = cleanup_file(stored.physical_path)
    sidecar_removed = cleanup_file(stored.sidecar_path)

    return qdrant_removed and db_row_removed and physical_removed and sidecar_removed


def _resolve_cancelled_storage(storage_future: "asyncio.Future", user_id: uuid.UUID) -> None:
    """
    Called from `ingest_document()`'s `except asyncio.CancelledError:`
    handler around the storage step, after `await_worker(storage_future)`
    has already blocked until `storage_future` reached a terminal state —
    genuine terminal state of the executor thread itself (see
    `utils.helpers.submit_worker()`), not merely of an asyncio Task wrapping
    it — so it is always safe here to inspect/act on its outcome without
    racing the worker thread.

    Cleans up ONLY files the worker itself newly created (never a
    bystander); sends no adapter-facing notification (the caller is
    already being cancelled); logs only safe, sanitized metadata (Stage 1D).
    """
    if storage_future.cancelled():
        return
    exc = storage_future.exception()
    if exc is not None:
        # _store_document_exclusively() already cleans up its own partial
        # write/sidecar before raising — nothing new to delete. Retrieving
        # the exception here just keeps asyncio from ever reporting it as
        # "exception was never retrieved".
        logger.warning(
            "Document upload: cancelled during storage, worker failed | user_id=%s, error_type=%s",
            user_id, type(exc).__name__
        )
        return
    stored = storage_future.result()
    logger.warning(
        "Document upload: cancelled during storage, removing orphaned file | user_id=%s",
        user_id
    )
    # By the time storage_future reached this terminal state, the DB
    # catalog row was already created (create_pending_sync() runs inside
    # the same worker call, before _store_document_exclusively() returns)
    # — clean it up alongside the physical file and sidecar.
    try:
        db_documents.delete_sync(document_id=stored.document_uuid)
    except Exception as e:
        logger.warning("Document upload: cancelled-storage cleanup, catalog row delete failed | user_id=%s, error_type=%s", user_id, type(e).__name__)
    cleanup_file(stored.physical_path)
    cleanup_file(stored.sidecar_path)


def _resolve_cancelled_after_storage(
    index_future: "Optional[asyncio.Future]", stored: StoredUpload, user_id: uuid.UUID
) -> None:
    """
    Called from `ingest_document()`'s `except asyncio.CancelledError:`
    handler wrapping the ENTIRE protected region between durable storage
    success and successful indexing (Stage 2B-C Blocker 1) — this covers
    BOTH the caller-supplied `before_indexing` hook's await AND the
    load/index worker submission + `await_worker()`, so a cancellation
    landing at that hook (e.g. a Telegram "indexing..." status message,
    sent by the adapter) can never orphan the already-durable source file
    + sidecar the way it did before this fix (there was previously no
    `except asyncio.CancelledError:` covering that specific await at all —
    the exception propagated straight out of `process_document_upload()`,
    uncaught, since `CancelledError` is a `BaseException` and the
    function's own `except Exception:` never saw it).

    `index_future` is `None` when cancellation landed BEFORE the load/index
    worker was ever submitted (during/before the `before_indexing` hook) —
    nothing was started, so nothing to wait for; the storage step's file +
    sidecar are the only durable artifacts and are cleaned up directly.

    Otherwise `index_future` is guaranteed already terminal: this is only
    ever called after `await_worker(index_future)` has itself returned/
    raised, which does not happen until the underlying executor thread
    genuinely finished (see utils/helpers.py) — so it is always safe here
    to inspect `.cancelled()`/`.exception()`/`.result()` directly, and safe
    to touch `stored.physical_path` (the worker is guaranteed done with
    it), without racing the worker thread. This also correctly handles the
    race where the worker completed SUCCESSFULLY between when the caller
    was cancelled and when it observed that: `index_future.exception()`
    is `None` and `.cancelled()` is `False` in exactly that case, so the
    "already committed" branch below is taken and nothing is deleted.

    Sends no adapter-facing notification; logs only safe, sanitized
    metadata (Stage 1D).
    """
    if index_future is None:
        logger.warning(
            "Document upload: cancelled before indexing started, removing orphaned file | user_id=%s",
            user_id
        )
        if not _cleanup_new_upload(stored):
            logger.warning("Document upload: cleanup incomplete after cancellation before indexing | user_id=%s", user_id)
        return

    if index_future.cancelled():
        if not _cleanup_new_upload(stored):
            logger.warning("Document upload: cleanup incomplete after cancelled indexing worker | user_id=%s", user_id)
        return
    exc = index_future.exception()
    if exc is not None:
        # Ingestion did not commit — remove the newly-owned upload (file +
        # sidecar + any partial Qdrant points), same as the non-cancelled
        # failure path.
        logger.warning(
            "Document upload: cancelled during indexing, ingestion did not commit | user_id=%s, error_type=%s",
            user_id, type(exc).__name__
        )
        if not _cleanup_new_upload(stored):
            logger.warning("Document upload: cleanup incomplete after indexing failure | user_id=%s", user_id)
        return
    # Indexing succeeded despite cancellation: chunks are already committed
    # into Qdrant. Never delete successfully ingested data, and skip the
    # normal success notification — the request itself was cancelled.
    logger.warning(
        "Document upload: cancelled after indexing already committed, retaining file | user_id=%s",
        user_id
    )


async def ingest_document(
    file_bytes: bytes,
    extension: str,
    display_name: str,
    owner_user_id: uuid.UUID,
    *,
    before_indexing: Optional[Callable[[], Awaitable[None]]] = None,
    _test_post_index_hook: Optional[Callable[[], None]] = None,
) -> DocumentIngestResult:
    """
    Adapter-independent document ingestion transaction (Stage 5B,
    identity migrated to canonical UUID ownership Stage 5C).

    Validates the extension and (against the actual bytes) the size limit,
    then runs storage + indexing as a single unit and returns a structured
    result — no Telegram types, no exception propagation for expected
    failure modes (unsupported extension, oversized, storage/indexing
    failure all come back as `DocumentIngestResult(success=False, ...)`).
    `asyncio.CancelledError` is the one exception that still propagates:
    a caller (any adapter) that gets cancelled while awaiting this
    coroutine relies on that to know its own task was cancelled, same as
    every other awaited call in this codebase.

    `before_indexing`, if given, is awaited BETWEEN the storage and
    indexing steps, INSIDE the same protected region that guards the
    load/index worker (Stage 2B-C Blocker 1) — this is deliberately where
    handlers/document_upload.py's original inline
    `await bot.send_message(..., "Индексирую документ...")` used to sit.
    Moving that Telegram-specific send into an adapter-supplied hook
    (rather than dropping it, or moving it outside this function into the
    adapter) is what lets this function stay Telegram-independent while
    the cancellation-safety window around it stays EXACTLY as wide as
    before: a cancellation landing on the hook's own await is resolved via
    `_resolve_cancelled_after_storage(index_future=None, ...)` exactly
    like a cancellation on the original inline status-message send was.
    A plain (non-cancellation) exception from the hook is likewise treated
    exactly like any other indexing-region failure: rolled back via
    `_cleanup_new_upload()` and reported as `success=False`.

    `_test_post_index_hook`, if given, is passed straight through to
    `_load_and_index_document()` — see its own docstring (Stage 5C
    corrective pass #4, Blocker 5). Test-only; every real caller leaves
    this None.
    """
    if extension not in SUPPORTED_EXTENSIONS:
        return DocumentIngestResult(success=False, rejected_reason="unsupported_extension")

    if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        return DocumentIngestResult(
            success=False, rejected_reason="oversized", file_size_bytes=len(file_bytes)
        )

    stored: "Optional[StoredUpload]" = None
    try:
        # Disk write of the downloaded bytes (up to MAX_DOCUMENT_SIZE_BYTES)
        # plus the durable sidecar write are blocking I/O — run them off
        # the event loop via submit_worker(), awaited through
        # await_worker(): if this coroutine is cancelled (even repeatedly)
        # while they're in flight, the worker thread is never abandoned
        # (see _resolve_cancelled_storage()).
        storage_future = submit_worker(_store_document_exclusively, file_bytes, extension, display_name, owner_user_id)
        try:
            stored = await await_worker(storage_future)
        except asyncio.CancelledError:
            _resolve_cancelled_storage(storage_future, owner_user_id)
            raise

        logger.info(
            "Document upload: file saved | user_id=%s, storage_name=%s, size_bytes=%s",
            owner_user_id, stored.physical_path.name, len(file_bytes)
        )
        # Stage 2B-C Blocker 1: EVERYTHING from here through a successfully
        # observed indexing result is one protected region against
        # cancellation — the source file + sidecar are already durable at
        # this point, so a cancellation landing ANYWHERE in here (including
        # at the `before_indexing` hook, not just the load/index worker's
        # own await_worker() call) must be resolved via
        # _resolve_cancelled_after_storage(), never allowed to propagate
        # uncaught. `index_future` starts as None and is only assigned once
        # the load/index worker is actually submitted, so the resolver can
        # tell "cancelled before the worker even started" apart from
        # "cancelled while/after the worker ran".
        index_future: "Optional[asyncio.Future]" = None
        try:
            if before_indexing is not None:
                await before_indexing()
            # Parsing (PDF/DOCX/TXT) + Qdrant/embeddings indexing is a
            # blocking pipeline — same submit_worker()/await_worker()
            # pattern: repeated cancellation here must not race the worker
            # for ownership of `stored.physical_path`.
            # _test_post_index_hook is passed as an extra kwarg ONLY when
            # actually given — never unconditionally — so a test that
            # monkeypatches `_load_and_index_document` with a double
            # matching the ordinary (stored, display_name) signature (the
            # overwhelming majority of this codebase's existing tests)
            # keeps working unchanged; only a test that deliberately opts
            # into this hook needs to accept the extra keyword.
            index_kwargs = {}
            if _test_post_index_hook is not None:
                index_kwargs["_test_post_index_hook"] = _test_post_index_hook
            index_future = submit_worker(_load_and_index_document, stored, display_name, **index_kwargs)
            chunk_count = await await_worker(index_future)
        except asyncio.CancelledError:
            _resolve_cancelled_after_storage(index_future, stored, owner_user_id)
            raise
        logger.info("Document indexed | user_id=%s, chunks=%s", owner_user_id, chunk_count)
    except Exception as e:
        # rag.safe_files.read_regular_file_secure() can raise on a detected
        # race/containment violation, document_loader.load_document_bytes()
        # can fail on local PDF/TXT/DOCX parsing, and
        # vector_index.reconcile_document() reaches Qdrant + OpenAIEmbeddings
        # (a network call to OpenAI) — any of these can surface a
        # provider/HTTP exception, so only the exception's class name is
        # logged here, never its text, a traceback, or the user-controlled
        # display filename.
        logger.error("Document upload failed | user_id=%s, extension=%s, error_type=%s", owner_user_id, extension, type(e).__name__)
        cleanup_complete = True
        if stored is not None:
            cleanup_complete = _cleanup_new_upload(stored)
            if not cleanup_complete:
                logger.warning("Document upload: cleanup incomplete after upload failure | user_id=%s", owner_user_id)
        else:
            # Stage 5C corrective pass #2 (Section 4): `stored` is None
            # whenever storage itself failed (_store_document_exclusively()
            # raised before ever returning a StoredUpload) — there is no
            # owned StoredUpload to hand to _cleanup_new_upload() here, but
            # that function ALREADY performed its own best-effort cleanup
            # internally and knows whether it actually succeeded. Reading
            # that real outcome off the exception (when present) is what
            # stops this branch from defaulting to a blind
            # cleanup_complete=True that could misreport a physical file
            # genuinely left behind on disk. Absent for any other
            # exception (e.g. the storage-path-exhaustion RuntimeError,
            # which never created an artifact to begin with) — True stays
            # correct there.
            partial_cleanup_complete = getattr(e, "partial_storage_cleanup_complete", None)
            if partial_cleanup_complete is not None:
                cleanup_complete = partial_cleanup_complete
                if not cleanup_complete:
                    logger.warning("Document upload: cleanup incomplete after partial storage failure | user_id=%s", owner_user_id)
        # Stage 7A-3 corrective pass: classified from the ORIGINAL exception
        # by TYPE only (never its text), after rollback has already run above
        # — the compensating cleanup is identical for every failure kind.
        failure_reason = (
            INGEST_FAILURE_KNOWLEDGE_BASE_UNAVAILABLE if is_index_unavailable_error(e) else None
        )
        return DocumentIngestResult(
            success=False,
            error_type=type(e).__name__,
            cleanup_complete=cleanup_complete,
            failure_reason=failure_reason,
        )

    # Ingestion has already committed (file + sidecar stored, chunks
    # indexed into Qdrant) — this is the ingestion success boundary.
    # Nothing after this point rolls back or reports the upload as failed.
    return DocumentIngestResult(
        success=True, chunk_count=chunk_count, stored=stored, file_size_bytes=len(file_bytes)
    )


# =============================================================================
# Stage 7A-3: catalog-only list/detail, and owner-initiated delete.
# =============================================================================


@dataclass(frozen=True)
class DocumentSummary:
    """The only document fields ever exposed to an HTTP caller (Stage
    7A-3) — never `stored_name`, `content_sha256`, or `status` (an active
    document's status is implied by it being visible at all; see
    db.documents.ACTIVE_STATUSES)."""
    id: uuid.UUID
    display_name: str
    created_at: datetime


class KnowledgeBaseUnavailableError(RuntimeError):
    """Raised by delete_document() when the Qdrant deletion step itself
    fails (Stage 7A-3) — the catalog row is left at 'deleting' (never
    rolled back to 'active': the delete was genuinely authorized and must
    still complete) for a caller to map to a fixed, safe 503 and retry."""


class DocumentDeletionError(RuntimeError):
    """Raised by delete_document() when physical/sidecar cleanup or the
    final catalog-row removal fails (Stage 7A-3), after Qdrant cleanup has
    already succeeded — the row is left at 'deleting' for a caller to map
    to a fixed, safe 500 and retry. Every step this wraps is idempotent, so
    a retry may safely re-run the whole sequence."""


async def list_documents(owner_user_id: uuid.UUID, *, limit: int, offset: int) -> List[DocumentSummary]:
    """Catalog-only, paginated list of `owner_user_id`'s own ACTIVE
    documents (Stage 7A-3) — never touches Qdrant or the filesystem."""
    records = await await_worker(
        submit_worker(db_documents.list_active_by_owner_sync, owner_user_id=owner_user_id, limit=limit, offset=offset)
    )
    return [DocumentSummary(id=r.id, display_name=r.display_name, created_at=r.created_at) for r in records]


async def get_document(owner_user_id: uuid.UUID, document_id: uuid.UUID) -> Optional[DocumentSummary]:
    """Catalog-only single-document lookup (Stage 7A-3). Returns None —
    never a distinguishable error — for a missing id, a foreign owner, a
    'pending' row, or a 'deleting' row alike: only an ACTIVE document owned
    by `owner_user_id` is ever returned, and every other case must produce
    an identical public 404 from the caller."""
    record = await await_worker(submit_worker(db_documents.get_sync, document_id=document_id))
    if record is None or record.status not in db_documents.ACTIVE_STATUSES or record.owner_user_id != owner_user_id:
        return None
    return DocumentSummary(id=record.id, display_name=record.display_name, created_at=record.created_at)


def _resolve_delete_targets(document_id: uuid.UUID, stored_name: str) -> Tuple[Path, Path]:
    """
    Validate a catalog row's `stored_name` as this document's own managed
    upload identity and return (physical_path, sidecar_path) — the ONLY two
    paths a delete may ever unlink (Stage 7A-3 corrective pass: an
    independent audit found deletion concatenating `MANAGED_UPLOADS_DIR /
    stored_name` directly, so a corrupted/hand-edited catalog value such as
    `..\\..\\outside.pdf` could point cleanup outside managed storage).

    `stored_name` is internal catalog data, never client input — but the
    catalog is still only a database column, so it is treated exactly like
    the sidecar-declared stored_name every other lifecycle path (rebuild,
    migration) already re-validates: the existing authoritative resolver
    rag.sidecar.resolve_managed_upload_path() enforces a bare basename (no
    separator, absolute path, or `.`/`..` segment), a supported managed-
    upload extension, and that the fully resolved path — following any
    symlink — remains inside MANAGED_UPLOADS_DIR. On top of that, this
    document's own identity is bound to the name: the resolved leaf must be
    exactly the declared `stored_name` (a symlinked leaf resolving to some
    OTHER in-directory file is rejected, never followed to and unlinked)
    and its stem must be exactly `document_id.hex` (the storage UUID this
    row's primary key and the RAG document_id are both derived from), so a
    catalog value naming any other managed file — even one that is
    perfectly contained — can never delete someone else's upload.

    The sidecar path is derived only from the successfully validated
    physical path via the one and only naming rule (rag.sidecar.
    sidecar_path_for()) — never built from unvalidated catalog text.

    Raises PathContainmentError (fixed, safe message; never the offending
    value) on any violation; performs no unlink and no Qdrant/DB call.
    """
    physical_path = resolve_managed_upload_path(MANAGED_UPLOADS_DIR, stored_name)
    if physical_path.name != stored_name or physical_path.stem != document_id.hex:
        raise PathContainmentError("stored_name does not correspond to this document's managed upload")
    return physical_path, sidecar_path_for(physical_path)


def _perform_delete_cleanup_sync(document_id: uuid.UUID, stored_name: str) -> None:
    """
    Idempotent cleanup for a catalog row already durably transitioned to
    'deleting' under the caller's ownership (Stage 7A-3) — mirrors
    _cleanup_new_upload()'s own ordering and idempotency contract (every
    step is safe to repeat: Qdrant delete_document() is itself a best-
    effort points-by-filter delete, cleanup_file() no-ops if the file is
    already gone, and db_documents.delete_sync() no-ops if the row is
    already gone), so a retry after ANY partial failure below may safely
    re-run the entire sequence rather than needing its own resume logic.

    Order (Section 6 of the Stage 7A-3 spec): Qdrant points, then the
    physical upload, then its sidecar, then the catalog row itself, last —
    never derived from anything client-supplied (`stored_name` comes from
    the catalog row, never a request body/path parameter), and validated
    against managed-storage containment by _resolve_delete_targets() BEFORE
    any side effect at all: a catalog storage identity that fails validation
    performs no Qdrant call, no unlink, and no catalog delete — the row is
    left at 'deleting' and DocumentDeletionError (fixed message, never the
    offending value or reason) is raised, so a retry remains possible after
    administrative/data correction.
    """
    try:
        physical_path, sidecar_path = _resolve_delete_targets(document_id, stored_name)
    except (ValueError, OSError) as e:
        # PathContainmentError is a ValueError; a NUL byte in the value makes
        # Path.resolve() raise a plain ValueError, an unresolvable one OSError.
        logger.warning(
            "Document delete: catalog storage identity failed validation | document_id=%s, error_type=%s",
            document_id, type(e).__name__,
        )
        raise DocumentDeletionError("Document deletion failed") from e

    rag_document_id = upload_document_id(document_id.hex)
    try:
        get_vector_index().delete_document(rag_document_id)
    except Exception as e:
        logger.warning("Document delete: Qdrant delete_document failed | error_type=%s", type(e).__name__)
        raise KnowledgeBaseUnavailableError("Knowledge base unavailable") from e

    physical_removed = cleanup_file(physical_path)
    sidecar_removed = cleanup_file(sidecar_path)
    if not (physical_removed and sidecar_removed):
        logger.warning("Document delete: physical/sidecar cleanup incomplete | document_id=%s", document_id)
        raise DocumentDeletionError("Document deletion failed")

    try:
        db_documents.delete_sync(document_id=document_id)
    except Exception as e:
        logger.warning("Document delete: catalog row delete failed | error_type=%s", type(e).__name__)
        raise DocumentDeletionError("Document deletion failed") from e


async def delete_document(owner_user_id: uuid.UUID, document_id: uuid.UUID) -> bool:
    """
    Owner-initiated document delete (Stage 7A-3): atomically establish or
    resume authorized deletion, then run idempotent cleanup.

    Returns False — never a distinguishable error — for a missing id, a
    foreign owner, or an owned 'pending' row alike (see
    db_documents.begin_or_resume_delete_sync()'s own docstring for why
    these three must stay indistinguishable).

    Authorization and the row's `stored_name` come from ONE atomic
    `UPDATE ... RETURNING` (begin_or_resume_delete_sync()): it authorizes
    both an own 'active' row (flipped to 'deleting' by that same statement)
    and an own 'deleting' row (resumed), and hands back the `stored_name`
    cleanup needs. There is deliberately NO later catalog read here — a
    follow-up SELECT (to recover stored_name, or to recognize an own
    'deleting' row) is exactly the disappearance window an audit found:
    a concurrent request could complete the whole cleanup and remove the
    row between the two, turning a request that WAS authorized into a
    spurious 404. If this request's own authorization statement ran while
    the row still existed, it is authorized to completion: the cleanup
    steps are all idempotent, so a concurrent DELETE that finishes first
    (including the final catalog-row removal, a zero-row DELETE here) just
    means both requests converge to success (Section 8 of the Stage 7A-3
    spec). A request whose authorization statement only runs after the row
    is already fully gone gets None -> False -> 404.

    Raises KnowledgeBaseUnavailableError / DocumentDeletionError for the
    two genuine failure modes below — the row is left at 'deleting' either
    way, so a caller mapping either to a 503/500 and retrying later will
    resume cleanup from wherever it left off.
    """
    stored_name = await await_worker(
        submit_worker(db_documents.begin_or_resume_delete_sync, document_id=document_id, owner_user_id=owner_user_id)
    )
    if stored_name is None:
        return False

    await await_worker(submit_worker(_perform_delete_cleanup_sync, document_id, stored_name))
    return True
