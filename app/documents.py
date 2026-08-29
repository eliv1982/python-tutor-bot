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
from pathlib import Path
from typing import Awaitable, Callable, Optional

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
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag.index import get_vector_index
from utils.logging import logger
from utils.helpers import cleanup_file, submit_worker, await_worker


@dataclass(frozen=True)
class StoredUpload:
    """Result of a successfully completed storage step: the physical file,
    its durable sidecar, and the identity/fingerprint values derived while
    creating them — everything the later load/index step and any cleanup
    path need, without re-deriving or re-reading anything.

    owner_user_id (Stage 3A): the immutable Telegram numeric id
    (`from_user.id`) captured once, at storage time, by
    `_store_document_exclusively()` — already durably persisted in the
    sidecar by that point. Carried here so `_load_and_index_document()`
    can pass it straight to `VectorIndex.reconcile_document()` without
    re-deriving or re-reading it from anywhere."""
    physical_path: Path
    sidecar_path: Path
    document_id: str
    content_sha256: str
    owner_user_id: int


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
    """
    success: bool
    chunk_count: Optional[int] = None
    stored: Optional[StoredUpload] = None
    file_size_bytes: int = 0
    rejected_reason: Optional[str] = None
    error_type: Optional[str] = None
    cleanup_complete: Optional[bool] = None


def _store_document_exclusively(
    file_bytes: bytes, extension: str, display_name: str, owner_user_id: int, attempts: int = 5
) -> StoredUpload:
    """
    Atomically claim a fresh, opaque, application-generated storage path,
    write the document bytes into it in the same exclusive-create
    operation, then write its durable `.meta.json` sidecar (Stage 2B) —
    the physical file and its sidecar together are the durable source of
    truth for this upload, independent of whatever is or isn't currently
    in Qdrant.

    `owner_user_id` (Stage 3A): the uploader's immutable Telegram numeric
    id, persisted into the sidecar via `build_sidecar()` so ownership
    survives a process restart and is available independently of any
    Telegram session state — see rag/sidecar.py.

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
        except Exception:
            # Ownership of `candidate` was established by the open() above,
            # so cleaning it up here can never remove another attempt's
            # file. cleanup_file() already swallows its own errors, so this
            # cannot mask the original write/close exception re-raised
            # below.
            cleanup_file(candidate)
            raise

        document_id = upload_document_id(candidate.stem)
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
                    owner_user_id=owner_user_id,
                ),
            )
        except Exception:
            cleanup_file(candidate)
            raise

        return StoredUpload(
            physical_path=candidate,
            sidecar_path=sidecar_path,
            document_id=document_id,
            content_sha256=content_sha256,
            owner_user_id=owner_user_id,
        )

    raise RuntimeError("Could not allocate a unique document storage path") from last_collision_error


def _load_and_index_document(stored: StoredUpload, display_name: str) -> int:
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
        owner_user_id=stored.owner_user_id,
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
    cleanup component — Qdrant delete, physical-file unlink, AND sidecar
    unlink — actually completed. Before Stage 2B-D, a filesystem unlink
    failure was invisible here: `utils.helpers.cleanup_file()` swallowed
    unlink errors and returned nothing, so this function could return True
    even though a durable artifact physically remained on disk. It now
    aggregates `cleanup_file()`'s own real per-file outcome (Stage 2B-D
    Blocker 2) alongside the Qdrant outcome. This function never raises, so
    callers already resolving a failure/cancellation can still safely call
    it unconditionally.
    """
    qdrant_removed = True
    try:
        get_vector_index().delete_document(stored.document_id)
    except Exception as e:
        logger.warning("Document upload cleanup: Qdrant delete_document failed (filesystem cleanup still attempted) | error_type=%s", type(e).__name__)
        qdrant_removed = False

    physical_removed = cleanup_file(stored.physical_path)
    sidecar_removed = cleanup_file(stored.sidecar_path)

    return qdrant_removed and physical_removed and sidecar_removed


def _resolve_cancelled_storage(storage_future: "asyncio.Future", user_id: int) -> None:
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
    cleanup_file(stored.physical_path)
    cleanup_file(stored.sidecar_path)


def _resolve_cancelled_after_storage(
    index_future: "Optional[asyncio.Future]", stored: StoredUpload, user_id: int
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
    owner_user_id: int,
    *,
    before_indexing: Optional[Callable[[], Awaitable[None]]] = None,
) -> DocumentIngestResult:
    """
    Adapter-independent document ingestion transaction (Stage 5B).

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
            index_future = submit_worker(_load_and_index_document, stored, display_name)
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
        return DocumentIngestResult(
            success=False, error_type=type(e).__name__, cleanup_complete=cleanup_complete
        )

    # Ingestion has already committed (file + sidecar stored, chunks
    # indexed into Qdrant) — this is the ingestion success boundary.
    # Nothing after this point rolls back or reports the upload as failed.
    return DocumentIngestResult(
        success=True, chunk_count=chunk_count, stored=stored, file_size_bytes=len(file_bytes)
    )
