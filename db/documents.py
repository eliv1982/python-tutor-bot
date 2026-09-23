"""
Document ownership/catalog operations (Stage 5C) — SYNC functions,
deliberately.

Called from INSIDE the existing executor-thread functions in
app/documents.py (via utils.helpers.submit_worker()/await_worker()) so
they share the exact cancellation-safety guarantee already established
for physical-file/sidecar writes and Qdrant reconciliation: a real OS
thread survives asyncio cancellation of the awaiting Task, so a caller's
existing cancellation-resolution logic can safely run afterward knowing
these calls already reached a genuine terminal state (committed or
raised) — see db/engine.py's module docstring for the full rationale.

Each function calls db.engine.get_sync_engine() fresh internally (never a
threaded-through `engine` parameter) — mirrors rag.index.get_vector_index()
being called fresh at each use site rather than passed around.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from db.engine import get_sync_engine
from db.models import Document

# Lifecycle states a caller may treat as "durably, genuinely indexed" for
# rebuild/retrieval purposes (Stage 5C corrective pass, Section 3). 'pending'
# is deliberately excluded: a row stuck at 'pending' means indexing was
# never confirmed to complete, and must never be silently treated as a
# valid private document for rebuild or retrieval — see
# scripts/rebuild_qdrant.py's catalog gate and rag/query.py's
# _validated_similarity_search().
ACTIVE_STATUSES = frozenset({"active"})


class CatalogConsistencyError(RuntimeError):
    """
    Raised when a document's catalog row cannot be proven to match the
    durable content that was just ingested/reconciled (Stage 5C corrective
    pass) — missing row, non-active status, or owner/stored_name/
    content_sha256 disagreement. Deliberately a fixed, safe message; the
    caller (app/documents.py) treats this exactly like any other
    indexing-region failure: full compensating cleanup, ingestion reported
    as failed, never a successful result with inconsistent catalog state.
    """


@dataclass(frozen=True)
class DocumentRecord:
    """Minimal, concrete snapshot of one `documents` catalog row — not a
    generic repository/DTO layer, just the exact fields Stage 5C's
    consistency checks need (ingestion verification, rebuild's ownership
    gate, retrieval's batched validation)."""
    id: uuid.UUID
    owner_user_id: uuid.UUID
    stored_name: str
    display_name: str
    content_sha256: str
    status: str
    created_at: datetime


def _record_from_row(row: Optional[Document]) -> Optional[DocumentRecord]:
    if row is None:
        return None
    return DocumentRecord(
        id=row.id,
        owner_user_id=row.owner_user_id,
        stored_name=row.stored_name,
        display_name=row.display_name,
        content_sha256=row.content_sha256,
        status=row.status,
        created_at=row.created_at,
    )


def get_sync(*, document_id: uuid.UUID) -> Optional[DocumentRecord]:
    """Return the catalog row for `document_id` regardless of its lifecycle
    status (callers must inspect `.status` themselves — this is a plain
    read, never itself an authorization/consistency decision), or None if
    no such row exists. Used to verify a just-activated row corresponds to
    the durable document that was actually ingested (app/documents.py) and
    by scripts/rebuild_qdrant.py's per-document ownership gate."""
    with Session(get_sync_engine()) as session:
        return _record_from_row(session.get(Document, document_id))


def get_active_owners_sync(document_ids: Sequence[uuid.UUID]) -> Dict[uuid.UUID, uuid.UUID]:
    """Batch lookup: {document_id: owner_user_id} for exactly those ids in
    `document_ids` that have a catalog row currently in an ACTIVE_STATUSES
    status. An id that is missing from the catalog, or whose row exists but
    is still 'pending' (or any other non-active status), is simply absent
    from the returned mapping — callers must treat absence as fail-closed
    ("do not trust this point"), never guess or fall back to the Qdrant
    payload alone. One batched query rather than one per candidate id (Stage
    5C corrective pass, Section 2) — used by rag/query.py's
    _validated_similarity_search() to cross-check every private-scope
    Qdrant result it is about to return in a single round trip."""
    if not document_ids:
        return {}
    with Session(get_sync_engine()) as session:
        rows = session.execute(
            select(Document.id, Document.owner_user_id).where(
                Document.id.in_(list(document_ids)), Document.status.in_(ACTIVE_STATUSES)
            )
        ).all()
    return {row.id: row.owner_user_id for row in rows}


def create_pending_sync(
    *, document_id: uuid.UUID, owner_user_id: uuid.UUID, stored_name: str, display_name: str, content_sha256: str
) -> None:
    """Insert the catalog row for a brand-new upload, status='pending'.
    Session.commit() is atomic — there is no partially-committed state a
    caller needs to defend against; a failure here means nothing was
    persisted at all, and the caller (app/documents.py) treats it exactly
    like any other storage-step failure (full rollback of the physical
    file + sidecar already written)."""
    with Session(get_sync_engine()) as session:
        session.add(
            Document(
                id=document_id,
                owner_user_id=owner_user_id,
                stored_name=stored_name,
                display_name=display_name,
                content_sha256=content_sha256,
            )
        )
        session.commit()


def reconcile_ambiguous_create_pending_sync(
    *, document_id: uuid.UUID, owner_user_id: uuid.UUID, stored_name: str, display_name: str, content_sha256: str
) -> bool:
    """
    Cleanup/reconciliation for create_pending_sync()'s AMBIGUOUS failure
    window (Stage 5C corrective pass #3, Blocker 3): `session.commit()`
    can, in principle, durably commit its INSERT at the database level
    and still raise back to the caller (e.g. the server commits but the
    client never receives the acknowledgement) — the caller then believes
    storage failed entirely and has no StoredUpload to clean up through
    the normal path, yet a 'pending' row may already durably exist. Codex
    reproduced exactly this: pending row committed, operation raised,
    physical/sidecar cleanup ran, the DB row was never touched, and
    ingestion still reported `cleanup_complete=True`.

    Called by app/documents.py's `_store_document_exclusively()` from the
    SAME exception handler that cleans up the physical file/sidecar after
    a `create_pending_sync()` failure — every field this function compares
    against is already known there (the caller was ABOUT to insert exactly
    these values), so this is a pure identity check, never a guess.

    Never blindly `DELETE ... WHERE id = document_id`: a pre-existing,
    UNRELATED row already occupying this exact id must never be silently
    destroyed merely because a new upload also picked this id (in
    practice `document_id` is a freshly-generated uuid4(), so a genuine
    collision is not the realistic threat — a caller bug or a stale
    document_id reused by mistake is). The row is read first and deleted
    ONLY if every field (owner, stored_name, display_name, content_sha256)
    matches exactly what THIS create_pending_sync() call was attempting to
    insert AND its status is still 'pending' (anything else means some
    other process already progressed/touched it — never ours to delete).

    Returns:
      - True — the DB side of cleanup is now provably complete: either no
        row exists at all (create_pending_sync() genuinely never
        committed — the common case), or a row existed, matched exactly,
        and was deleted just now.
      - False — cleanup is INCOMPLETE, never silently treated as done:
        either a row exists but disagrees with the expected identity/
        ownership/status (a genuine inconsistency the caller must report,
        never silently delete), or reading/deleting the row itself raised
        (the DB itself is unreachable — the same ambiguous-failure class
        this function exists to handle, now affecting the cleanup attempt
        too).

    Stage 5C corrective pass #4 (Blocker 3): the previous implementation
    here was SELECT (session.get) -> compare in Python -> DELETE, three
    separate steps — a plain SELECT takes no row lock, so a CONCURRENT
    transaction could activate (or otherwise change) this exact row after
    the comparison judged it a safe-to-delete 'pending' match but before
    the DELETE actually ran, turning what was judged a safe compensating
    action into a destructive one against since-changed state. The
    comparison and the deletion are now ONE atomic conditional DELETE whose
    WHERE clause carries the COMPLETE expected pending-row contract (id,
    owner, stored_name, display_name, content_sha256, status='pending') —
    a single DML statement PostgreSQL evaluates and applies atomically per
    row (under READ COMMITTED, a concurrent UPDATE that commits first
    simply makes this statement's predicate no longer match the now-current
    row, so it deletes nothing; a concurrent UPDATE that is still in-flight
    blocks this statement until it commits or rolls back, and the predicate
    is then re-evaluated against whatever the row actually ended up as).
    There is no in-between "we judged it safe, but haven't deleted yet"
    window at all — the database itself is the sole arbiter of whether the
    row still matches at the instant of deletion. The follow-up SELECT
    below (when the DELETE affected zero rows) is purely INFORMATIONAL —
    it exists only to distinguish "never existed" (True) from "exists but
    disagreed/already progressed" (False) for the caller's return value,
    and never itself drives any destructive action, so it introduces no
    new race.
    """
    try:
        with Session(get_sync_engine()) as session:
            result = session.execute(
                delete(Document).where(
                    Document.id == document_id,
                    Document.owner_user_id == owner_user_id,
                    Document.stored_name == stored_name,
                    Document.display_name == display_name,
                    Document.content_sha256 == content_sha256,
                    Document.status == "pending",
                )
            )
            if result.rowcount == 1:
                session.commit()
                return True
            session.rollback()
            row = session.get(Document, document_id)
            return row is None
    except Exception:
        return False


def mark_active_sync(*, document_id: uuid.UUID) -> None:
    """Flip a document's catalog row to 'active' after indexing has
    genuinely committed to Qdrant. Raises if NO row at all exists for
    `document_id` — reaching indexing success without a prior committed
    create_pending_sync() call is a genuine invariant violation worth
    surfacing loudly. Idempotent against an already-'active' row (the
    UPDATE matches by id alone, not `WHERE status='pending'`): calling this
    again for an already-active document — e.g. scripts/rebuild_qdrant.py
    re-affirming a document it just successfully reconciled — is a
    harmless no-op, not an error.

    Stage 5C corrective pass (Section 1): this is now the OPPOSITE of
    best-effort from the caller's point of view. A failure here means
    ingestion cannot prove its catalog state matches the durable content it
    just committed, and app/documents.py's _load_and_index_document() lets
    that failure propagate instead of swallowing it — the caller's normal
    indexing-failure path (full compensating cleanup: Qdrant points,
    physical file, sidecar, and the — possibly still-'pending' — catalog
    row) then runs, and ingestion is reported as failed. Correctness takes
    priority over avoiding repeat embedding/provider cost; see the module
    docstring's cross-reference from app/documents.py for the rationale."""
    with Session(get_sync_engine()) as session:
        result = session.execute(
            update(Document).where(Document.id == document_id).values(status="active")
        )
        if result.rowcount != 1:
            session.rollback()
            raise RuntimeError("mark_active_sync: no pending document row found to update")
        session.commit()


def delete_sync(*, document_id: uuid.UUID) -> None:
    """Best-effort delete of a document's catalog row — a no-op if it was
    never created (same "safe to call unconditionally" contract as
    utils.helpers.cleanup_file()). Used by app/documents.py's
    _cleanup_new_upload() alongside its existing Qdrant/physical-file/
    sidecar cleanup, and (Stage 7A-3) as the final step of authenticated
    document deletion — idempotent either way, so a retry after a partial
    failure, or a concurrent request that already removed the row, is
    never itself an error."""
    with Session(get_sync_engine()) as session:
        session.execute(delete(Document).where(Document.id == document_id))
        session.commit()


def list_active_by_owner_sync(*, owner_user_id: uuid.UUID, limit: int, offset: int) -> List[DocumentRecord]:
    """Page through `owner_user_id`'s own ACTIVE documents only (Stage
    7A-3) — 'pending'/'deleting' rows are never catalog-listable. Stable
    ordering (created_at DESC, id DESC as a deterministic tie-breaker for
    same-instant rows) so pagination never skips/repeats a row across
    calls."""
    with Session(get_sync_engine()) as session:
        rows = session.execute(
            select(Document)
            .where(Document.owner_user_id == owner_user_id, Document.status == "active")
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        ).scalars().all()
    return [_record_from_row(row) for row in rows]


def begin_or_resume_delete_sync(*, document_id: uuid.UUID, owner_user_id: uuid.UUID) -> Optional[str]:
    """
    Race-safe entry point into document deletion (Stage 7A-3) — ownership/
    status authorization, the 'active' -> 'deleting' mutation, AND capture
    of the `stored_name` the caller's cleanup needs are ONE atomic
    `UPDATE ... RETURNING` statement, never a separate SELECT-then-UPDATE
    and never an UPDATE followed by any later authorization read, for the
    identical reason reconcile_ambiguous_create_pending_sync() above gives:
    a plain SELECT takes no row lock, so a concurrent transaction could
    change (or fully delete) the row between a Python-level check and a
    later read/write.

    Stage 7A-3 corrective pass: an independent audit reproduced the exact
    gap the previous shape had — `UPDATE ... WHERE status='active'` (zero
    rows, because a concurrent request A had already flipped this row to
    'deleting') -> A finishes its cleanup and removes the catalog row ->
    this call's follow-up SELECT (which was the only thing that could
    recognize an own 'deleting' row) finds nothing -> spurious 404 for a
    caller that reached the row while it still existed as own 'deleting'.
    The WHERE clause below authorizes BOTH own states — 'active' (this
    call becomes the one to flip it) and 'deleting' (a previous/concurrent
    delete of the SAME owner's row; the SET is then an idempotent no-op on
    `status`) — so there is no zero-rows-then-look-again step left to have
    a window in at all. The row's own `stored_name`, captured by the same
    statement's RETURNING, is what the caller uses for cleanup: no later
    read of the catalog row is ever needed to recover it (which would
    recreate the same disappearance window).

    Returns:
      - the row's `stored_name` (a str) if, at the instant this single
        statement ran, a row with this exact id, owned by `owner_user_id`,
        existed in status 'active' or 'deleting' — the caller is now an
        authorized party proceeding with (or resuming) idempotent cleanup.
        Callers do not need to distinguish "started" from "resumed".
      - None otherwise — covers a nonexistent id (including one whose
        deletion had already completed before this statement began), a
        foreign owner, and an owned 'pending' row alike. These cases are
        DELIBERATELY indistinguishable to the caller (and therefore to the
        HTTP layer, which maps None to a single generic 404): a foreign
        document's mere existence must never be disclosed by a different
        response for "belongs to someone else" versus "does not exist".

    Concurrency note (PostgreSQL READ COMMITTED): if a concurrent request's
    own final catalog DELETE is still in flight when this statement reaches
    the row, this statement waits for that DELETE and then re-evaluates
    against the now-removed row (returning None). That is a correct
    linearization — this request's authorization statement is ordered after
    the row's complete removal — not a lost authorization.
    """
    with Session(get_sync_engine()) as session:
        row = session.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.owner_user_id == owner_user_id,
                Document.status.in_(("active", "deleting")),
            )
            .values(status="deleting")
            .returning(Document.stored_name)
        ).first()
        session.commit()
        return row.stored_name if row is not None else None
