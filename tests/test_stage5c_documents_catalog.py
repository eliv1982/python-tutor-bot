"""
Stage 5C regression tests: the PostgreSQL document ownership/catalog
(db.documents) stays consistent with the sidecar/Qdrant state across the
real app.documents.ingest_document() pipeline, against a REAL disposable
PostgreSQL container — proving genuine constraint/transaction behavior,
never a mocked stand-in.

Covers:
- a successful ingest leaves file + sidecar + DB row (status='active') +
  Qdrant points all agreeing on ownership;
- an indexing failure rolls back file + sidecar + DB row + any partial
  Qdrant points together (no partial/inconsistent state across stores);
- mark_active_sync() failure (Stage 5C corrective pass): a catalog
  activation failure is now treated exactly like any other indexing-region
  failure — full compensating cleanup across file/sidecar/DB/Qdrant, and
  ingestion reported as failed, never a successful result with a missing
  or still-'pending' catalog row;
- delete_sync()/create_pending_sync()/mark_active_sync() constraint
  behavior directly (FK to users, CHECK on status, idempotent delete).

See tests/conftest.py's postgres_container()/postgres_db() fixtures for
the disposable-container mechanics; tests/rag_fakes.py's
DeterministicFakeEmbeddings for the local Qdrant double (still real local-
persistent Qdrant, never a real OpenAI/Qdrant network call).
"""

import contextlib
import json
import threading
import time
import uuid

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import app.documents as app_documents
import db.documents as db_documents
import db.identity as db_identity
from db.engine import get_sync_engine
from db.models import Document, User
from rag.index import VectorIndex
from rag.sidecar import write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's same-named autouse fixture — this module
    exercises the REAL db.documents functions against postgres_db."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Identity resolution used by this module's owner-UUID setup also
    needs the real resolver (a real users row is required for the
    documents.owner_user_id foreign key to be satisfiable)."""
    yield


@pytest.fixture
def owner_uuid(postgres_db):
    """A real, committed users row — app.documents.ingest_document()'s
    owner_user_id is a foreign key into users.id, so tests need a genuine
    row there, not an arbitrary uuid4()."""
    return db_identity.resolve_or_create_user_by_telegram_id_sync(770000001)


@pytest.fixture
def real_vector_index(tmp_path, monkeypatch):
    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage5c_catalog_test",
    )
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)
    yield uploads_dir
    vi.close()


def _catalog_row(document_id: uuid.UUID):
    with Session(get_sync_engine()) as session:
        return session.get(Document, document_id)


# ---------------------------------------------------------------------------
# A. Successful ingest: file + sidecar + DB row (active) + Qdrant agree
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_ingest_leaves_consistent_active_catalog_row(postgres_db, owner_uuid, real_vector_index):
    result = await app_documents.ingest_document(
        file_bytes=b"Consistent catalog content for a successful ingest.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is True
    row = _catalog_row(result.stored.document_uuid)
    assert row is not None
    assert row.status == "active"
    assert row.owner_user_id == owner_uuid
    assert row.content_sha256 == result.stored.content_sha256
    assert row.stored_name == result.stored.physical_path.name
    assert row.display_name == "notes.txt"


# ---------------------------------------------------------------------------
# B. Indexing failure -> full rollback across file/sidecar/DB/Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_indexing_failure_removes_the_catalog_row_too(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    monkeypatch.setattr(
        app_documents.get_vector_index(), "reconcile_document",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated indexing failure")),
    )

    result = await app_documents.ingest_document(
        file_bytes=b"content that will fail to index",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.cleanup_complete is True
    # DocumentIngestResult.stored is None on the failure path (only
    # populated on success) — check by owner instead: the catalog row
    # created during storage must be gone too, never left behind as an
    # orphan 'pending' row for content that was fully rolled back on disk
    # and in Qdrant.
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []
    assert list(real_vector_index.iterdir()) == []


# ---------------------------------------------------------------------------
# C. mark_active_sync() failure now triggers full compensating rollback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_active_failure_rolls_back_and_reports_failure(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Stage 5C corrective pass: a catalog activation failure, proved
    against a REAL constraint failure (not a mock) by sabotaging
    mark_active_sync() so it genuinely fails against Postgres (deleting
    the row out from under it right before it runs, reproducing the exact
    'no pending row to update' RuntimeError db/documents.py raises), must
    now be treated exactly like any other indexing-region failure: the
    already-committed Qdrant points, physical file, and sidecar are rolled
    back via the existing _cleanup_new_upload() path, and ingestion
    reports failure — an ingestion operation must never report success
    while its catalog row is missing or stuck at 'pending'."""
    real_mark_active = db_documents.mark_active_sync

    def sabotaged_mark_active(*, document_id):
        with Session(get_sync_engine()) as session:
            session.execute(Document.__table__.delete().where(Document.id == document_id))
            session.commit()
        return real_mark_active(document_id=document_id)

    monkeypatch.setattr(app_documents.db_documents, "mark_active_sync", sabotaged_mark_active)

    result = await app_documents.ingest_document(
        file_bytes=b"content that must not survive a catalog activation failure",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.cleanup_complete is True
    # File + sidecar rolled back (never retained over a catalog activation
    # failure — correctness/consistency takes priority over the sunk
    # embedding/provider cost).
    physical_files = [p for p in real_vector_index.iterdir() if not p.name.endswith(".meta.json")]
    assert physical_files == []
    # Qdrant content rolled back too.
    assert app_documents.get_vector_index().get_stats(requesting_user_uuid=str(owner_uuid))["total_documents"] == 0
    # No orphan catalog row of any status remains for this owner.
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []


@pytest.mark.asyncio
async def test_catalog_owner_mismatch_after_activation_rolls_back_and_reports_failure(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Stage 5C corrective pass, Section 2: ingestion verifies the
    resulting ACTIVE catalog row actually corresponds to the durable
    document just ingested — not merely that SOME row with the right id
    exists. Sabotage mark_active_sync() to activate a row that (by the
    time verification runs) has a different owner recorded than the one
    this ingest call is using — reproducing a cross-store ownership
    disagreement — and prove ingestion fails closed and fully rolls back
    rather than reporting success with a mismatched catalog row."""
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(770000099)
    real_mark_active = db_documents.mark_active_sync

    def sabotaged_mark_active(*, document_id):
        real_mark_active(document_id=document_id)
        with Session(get_sync_engine()) as session:
            session.execute(
                Document.__table__.update().where(Document.id == document_id).values(owner_user_id=other_owner)
            )
            session.commit()

    monkeypatch.setattr(app_documents.db_documents, "mark_active_sync", sabotaged_mark_active)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose catalog owner is corrupted right after activation",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.cleanup_complete is True
    physical_files = [p for p in real_vector_index.iterdir() if not p.name.endswith(".meta.json")]
    assert physical_files == []
    assert app_documents.get_vector_index().get_stats(requesting_user_uuid=str(owner_uuid))["total_documents"] == 0


# ---------------------------------------------------------------------------
# D. db.documents constraint behavior, direct
# ---------------------------------------------------------------------------

def test_create_pending_requires_a_real_owner_foreign_key(postgres_db):
    with pytest.raises(IntegrityError):
        db_documents.create_pending_sync(
            document_id=uuid.uuid4(),
            owner_user_id=uuid.uuid4(),  # never resolved — no such users row
            stored_name="deadbeef00000000000000000000000000.txt",
            display_name="notes.txt",
            content_sha256="a" * 64,
        )


def test_mark_active_raises_if_no_pending_row_exists(postgres_db):
    with pytest.raises(RuntimeError):
        db_documents.mark_active_sync(document_id=uuid.uuid4())


def test_delete_sync_is_a_no_op_if_the_row_never_existed(postgres_db):
    db_documents.delete_sync(document_id=uuid.uuid4())  # must not raise


def test_status_check_constraint_rejects_an_invalid_status(postgres_db, owner_uuid):
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="b" * 64,
    )
    with Session(get_sync_engine()) as session:
        # The CHECK constraint fires immediately on execute() against a
        # real Postgres, not deferred to commit() — psycopg validates
        # server-side as soon as the statement runs.
        with pytest.raises(IntegrityError):
            session.execute(Document.__table__.update().where(Document.id == doc_id).values(status="deleted"))


def test_create_pending_then_mark_active_round_trip(postgres_db, owner_uuid):
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="c" * 64,
    )
    assert _catalog_row(doc_id).status == "pending"

    db_documents.mark_active_sync(document_id=doc_id)
    assert _catalog_row(doc_id).status == "active"

    db_documents.delete_sync(document_id=doc_id)
    assert _catalog_row(doc_id) is None


# ---------------------------------------------------------------------------
# E. Stage 5C corrective pass #3, Blocker 3: create_pending_sync()'s
# AMBIGUOUS failure window — session.commit() can, in principle, durably
# commit its INSERT and still raise back to the caller. Every scenario here
# runs against REAL PostgreSQL (never only the in-memory fake catalog).
# ---------------------------------------------------------------------------

def test_reconcile_ambiguous_create_pending_removes_a_genuinely_matching_row(postgres_db, owner_uuid):
    """Direct, unit-level proof of reconcile_ambiguous_create_pending_sync()'s
    conditional-delete contract against a REAL committed row."""
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="e" * 64,
    )

    reconciled = db_documents.reconcile_ambiguous_create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="e" * 64,
    )

    assert reconciled is True
    assert _catalog_row(doc_id) is None


def test_reconcile_ambiguous_create_pending_is_a_true_no_op_when_no_row_exists(postgres_db, owner_uuid):
    """The common case: create_pending_sync() genuinely never committed —
    reconciliation reports the DB side already complete, with zero writes."""
    doc_id = uuid.uuid4()

    reconciled = db_documents.reconcile_ambiguous_create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="e" * 64,
    )

    assert reconciled is True
    assert _catalog_row(doc_id) is None


def test_reconcile_ambiguous_create_pending_does_not_delete_a_mismatching_row(postgres_db, owner_uuid):
    """A pre-existing row at the same document_id but disagreeing on
    owner/stored_name (a genuine inconsistency — e.g. a UUID reused by
    mistake, or a leftover from an unrelated bug) must never be silently
    deleted — reported as an incomplete reconciliation instead, and the
    existing row left completely untouched."""
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(770000090)
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=other_owner,
        stored_name="preexisting.txt", display_name="preexisting-name.txt", content_sha256="f" * 64,
    )

    reconciled = db_documents.reconcile_ambiguous_create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,  # different owner — mismatch
        stored_name="attempted-new.txt", display_name="attempted-new-name.txt", content_sha256="a" * 64,
    )

    assert reconciled is False
    row = _catalog_row(doc_id)
    assert row is not None
    assert row.owner_user_id == other_owner
    assert row.stored_name == "preexisting.txt"
    assert row.display_name == "preexisting-name.txt"


def test_reconcile_ambiguous_create_pending_does_not_delete_an_already_active_row(postgres_db, owner_uuid):
    """Even a row that otherwise matches exactly, but has already been
    progressed to 'active' by the time reconciliation runs (some other
    process got there first), must not be deleted — only a still-'pending'
    row created by THIS exact attempt is ever ours to remove."""
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="b" * 64,
    )
    db_documents.mark_active_sync(document_id=doc_id)

    reconciled = db_documents.reconcile_ambiguous_create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="b" * 64,
    )

    assert reconciled is False
    row = _catalog_row(doc_id)
    assert row is not None
    assert row.status == "active"


def test_reconcile_ambiguous_create_pending_is_race_safe_against_concurrent_activation(postgres_db, owner_uuid):
    """
    Stage 5C corrective pass #4 (Blocker 3): REAL PostgreSQL concurrency
    proof that the comparison-and-delete is genuinely atomic, using two
    real threads/connections and an actual row-level lock — never a
    timing sleep pretending to model the race.

    Thread A opens its own real connection, executes an UPDATE (status
    'pending' -> 'active') INSIDE an open transaction, and deliberately
    does NOT commit yet — this holds a genuine PostgreSQL row lock. Thread
    B then calls the real reconcile_ambiguous_create_pending_sync() for
    that exact row with the ORIGINAL 'pending' contract; its own
    conditional DELETE statement must BLOCK on Thread A's lock (proven via
    a deterministic condition-poll against pg_stat_activity — not slept
    for) rather than proceeding past it. Only once Thread A commits does
    Thread B's DELETE resume, re-evaluate its WHERE clause against the
    NOW-CURRENT (committed 'active') row, and correctly find nothing left
    to delete — proving a concurrent activation can never be raced into a
    destructive delete no matter how the two operations interleave.
    """
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="d" * 64,
    )

    engine = get_sync_engine()
    # Pre-warm a couple of pooled connections before the timed critical
    # section below — a COLD connection's first-ever checkout (TCP
    # connect + auth) can occasionally take long enough on this host to
    # approach thread A's own safety-net wait timeout, which would let it
    # auto-commit before thread B's DELETE is even issued: a test-harness
    # false negative on the blocking guard, not a real production race.
    with engine.connect() as _warm1, engine.connect() as _warm2:
        _warm1.execute(text("SELECT 1"))
        _warm2.execute(text("SELECT 1"))

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_uncommitted_lock():
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(
                    Document.__table__.update().where(Document.id == doc_id).values(status="active")
                )
                lock_acquired.set()
                # A generous safety-net timeout, deliberately far longer
                # than the poll deadline below — this must never be what
                # actually releases the lock in a healthy run; it only
                # guards against the test hanging forever if something
                # else goes wrong.
                release_lock.wait(timeout=60)
            # `with conn.begin():` commits here, releasing the row lock.

    thread_a = threading.Thread(target=hold_uncommitted_lock)
    thread_a.start()
    assert lock_acquired.wait(timeout=5), "thread A never reported holding its uncommitted lock"

    result_holder = {}

    def run_reconcile():
        result_holder["reconciled"] = db_documents.reconcile_ambiguous_create_pending_sync(
            document_id=doc_id, owner_user_id=owner_uuid,
            stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="d" * 64,
        )

    thread_b = threading.Thread(target=run_reconcile)
    thread_b.start()

    # Deterministic condition-poll for genuine blocking (never a blind
    # sleep): only release thread A once PostgreSQL itself confirms SOME
    # other backend is actually waiting on a lock. This disposable
    # container runs nothing but this test's own two connections at this
    # point, so "any other backend waiting on a Lock" unambiguously means
    # thread B's DELETE — no need to match on query text (whose exact
    # driver-level formatting isn't a stable contract to assert on).
    deadline = time.monotonic() + 10
    observed_blocked = False
    with engine.connect() as poll_conn:
        while time.monotonic() < deadline:
            waiting = poll_conn.execute(text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE wait_event_type = 'Lock' AND pid != pg_backend_pid()"
            )).scalar()
            if waiting and waiting > 0:
                observed_blocked = True
                break
            time.sleep(0.05)
    assert observed_blocked, "thread B's DELETE never observably blocked on thread A's uncommitted lock"

    release_lock.set()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)
    assert not thread_a.is_alive() and not thread_b.is_alive()

    assert result_holder["reconciled"] is False, (
        "a row concurrently activated mid-reconciliation must be preserved, never destroyed"
    )
    row = _catalog_row(doc_id)
    assert row is not None
    assert row.status == "active"  # thread A's activation survives untouched


@pytest.mark.asyncio
async def test_create_pending_ordinary_failure_before_commit_reports_complete_cleanup(postgres_db, real_vector_index):
    """Requirement 1: an ORDINARY (non-ambiguous) create_pending_sync()
    failure — the INSERT genuinely never committed at all (a foreign-key
    violation against a nonexistent owner) — leaves no DB row to
    reconcile at all; cleanup_complete reflects only the physical file/
    sidecar cleanup outcome, exactly as before this corrective pass."""
    result = await app_documents.ingest_document(
        file_bytes=b"content whose owner foreign key does not exist",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=uuid.uuid4(),  # never resolved — no such users row
    )

    assert result.success is False
    assert result.cleanup_complete is True
    assert list(real_vector_index.iterdir()) == []


@pytest.mark.asyncio
async def test_ambiguous_create_pending_commit_then_raise_removes_the_real_row(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Requirement 2: the exact scenario Codex reproduced by wrapping the
    operation — create_pending_sync()'s commit genuinely succeeds against
    REAL PostgreSQL, then the call still raises (simulating a lost
    acknowledgement). ingest_document()'s exception handler must find and
    remove the REAL, matching row — not merely believe it does."""
    real_create_pending = db_documents.create_pending_sync

    def ambiguous_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        real_create_pending(
            document_id=document_id, owner_user_id=owner_user_id,
            stored_name=stored_name, display_name=display_name, content_sha256=content_sha256,
        )
        raise RuntimeError("simulated: commit succeeded but the acknowledgement was lost")

    monkeypatch.setattr(app_documents.db_documents, "create_pending_sync", ambiguous_create_pending)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose create_pending_sync commit is ambiguous",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.cleanup_complete is True
    physical_files = [p for p in real_vector_index.iterdir() if not p.name.endswith(".meta.json")]
    assert physical_files == []
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == [], "the ambiguously-committed pending row must be reconciled away"


@pytest.mark.asyncio
async def test_ambiguous_create_pending_with_db_reconcile_failure_reports_incomplete(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Requirement 3: the same ambiguous-commit scenario, but the DB-side
    reconciliation attempt ITSELF also fails — cleanup_complete must be
    False, and the row genuinely remains in real PostgreSQL (proving this
    isn't merely a mocked claim)."""
    real_create_pending = db_documents.create_pending_sync

    def ambiguous_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        real_create_pending(
            document_id=document_id, owner_user_id=owner_user_id,
            stored_name=stored_name, display_name=display_name, content_sha256=content_sha256,
        )
        raise RuntimeError("simulated: commit succeeded but the acknowledgement was lost")

    monkeypatch.setattr(app_documents.db_documents, "create_pending_sync", ambiguous_create_pending)
    monkeypatch.setattr(app_documents.db_documents, "reconcile_ambiguous_create_pending_sync", lambda **kwargs: False)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose DB-side cleanup attempt will itself fail",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.cleanup_complete is False
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert len(remaining) == 1, "the row genuinely remains — cleanup_complete=False reflects real state"


@pytest.mark.asyncio
async def test_ambiguous_create_pending_with_physical_cleanup_failure_reports_incomplete(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Requirement 4: the ambiguous commit is reconciled away successfully
    (real Postgres), but the PHYSICAL file cleanup fails — cleanup_complete
    must still be False overall: no known artifact may remain while it
    reports True."""
    real_create_pending = db_documents.create_pending_sync

    def ambiguous_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        real_create_pending(
            document_id=document_id, owner_user_id=owner_user_id,
            stored_name=stored_name, display_name=display_name, content_sha256=content_sha256,
        )
        raise RuntimeError("simulated: commit succeeded but the acknowledgement was lost")

    monkeypatch.setattr(app_documents.db_documents, "create_pending_sync", ambiguous_create_pending)
    monkeypatch.setattr(app_documents, "cleanup_file", lambda filepath: False)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose physical cleanup will itself fail",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.cleanup_complete is False
    # The DB row WAS genuinely reconciled away despite the physical
    # cleanup failure — proving the aggregate honestly reflects BOTH
    # artifacts' real state, not just one.
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []


@pytest.mark.asyncio
async def test_full_pipeline_existing_mismatching_row_blocks_new_ingest_and_reports_incomplete(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Requirement 5, exercised through the FULL ingest_document() pipeline:
    a pre-existing row genuinely occupies the storage UUID
    _store_document_exclusively() is about to generate (forced
    deterministic here via a fixed uuid.uuid4()), owned by someone else —
    create_pending_sync() genuinely fails (a real primary-key conflict),
    and the pre-existing row must be left completely untouched, with
    cleanup reported as incomplete."""
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(770000091)
    fixed_uuid = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=fixed_uuid, owner_user_id=other_owner,
        stored_name="preexisting.txt", display_name="preexisting-name.txt", content_sha256="c" * 64,
    )

    monkeypatch.setattr(app_documents.uuid, "uuid4", lambda: fixed_uuid)

    result = await app_documents.ingest_document(
        file_bytes=b"content whose freshly-generated storage uuid collides with a pre-existing row",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.cleanup_complete is False
    row = db_documents.get_sync(document_id=fixed_uuid)
    assert row is not None
    assert row.owner_user_id == other_owner  # untouched
    assert row.stored_name == "preexisting.txt"
    assert row.display_name == "preexisting-name.txt"
    physical_files = [p for p in real_vector_index.iterdir() if not p.name.endswith(".meta.json")]
    assert physical_files == []


# ---------------------------------------------------------------------------
# F. Stage 5C corrective pass #4 (Blocker 10): a document that parses/
# chunks into ZERO meaningful chunks must never become an active catalog
# document with nothing actually indexed for it — success must mean a
# useful, internally consistent document actually exists.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_txt_document_fails_ingestion_and_leaves_no_active_document(postgres_db, owner_uuid, real_vector_index):
    result = await app_documents.ingest_document(
        file_bytes=b"",
        extension=".txt",
        display_name="empty.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.error_type == "EmptyDocumentError"
    assert result.cleanup_complete is True
    physical_files = [p for p in real_vector_index.iterdir() if not p.name.endswith(".meta.json")]
    assert physical_files == []
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []
    assert app_documents.get_vector_index().get_stats(requesting_user_uuid=str(owner_uuid))["total_documents"] == 0


@pytest.mark.asyncio
async def test_empty_md_document_fails_ingestion_and_leaves_no_active_document(postgres_db, owner_uuid, real_vector_index):
    result = await app_documents.ingest_document(
        file_bytes=b"",
        extension=".md",
        display_name="empty.md",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.error_type == "EmptyDocumentError"
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []


@pytest.mark.asyncio
async def test_whitespace_only_document_that_chunks_to_nothing_fails_ingestion(postgres_db, owner_uuid, real_vector_index):
    """Whitespace-only content that the REAL RecursiveCharacterTextSplitter
    genuinely reduces to zero chunks (verified directly: split_documents()
    on whitespace-only page_content returns []) must be rejected the same
    way a literally-empty file is — never treated as a successful ingest
    with nothing indexed."""
    result = await app_documents.ingest_document(
        file_bytes=b"   \n\n\t  \n   ",
        extension=".txt",
        display_name="whitespace.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is False
    assert result.error_type == "EmptyDocumentError"
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []


@pytest.mark.asyncio
async def test_ordinary_tiny_but_nonempty_document_still_succeeds(postgres_db, owner_uuid, real_vector_index):
    """Regression guard: Blocker 10's fix must reject ONLY genuinely
    zero-chunk documents — an ordinary tiny (but real) document must still
    ingest and activate successfully, exactly as before."""
    result = await app_documents.ingest_document(
        file_bytes=b"Hi.",
        extension=".txt",
        display_name="tiny.txt",
        owner_user_id=owner_uuid,
    )

    assert result.success is True
    assert result.chunk_count >= 1
    row = _catalog_row(result.stored.document_uuid)
    assert row is not None
    assert row.status == "active"
    assert app_documents.get_vector_index().get_stats(requesting_user_uuid=str(owner_uuid))["total_documents"] >= 1


# ---------------------------------------------------------------------------
# G. Stage 5C corrective pass #4 (Blocker 5): live-ingestion source binding.
# A deterministic barrier proof using ingest_document()'s own
# `_test_post_index_hook` seam (mirrors rag.safe_files.
# read_regular_file_secure()'s `_test_pre_open_hook` convention) — the
# mutation happens at an EXACT, named point in the pipeline, never a timing
# race against a background thread.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_source_mutated_between_indexing_and_activation_fails_closed(postgres_db, owner_uuid, real_vector_index):
    """The file is mutated at the EXACT point between Qdrant reconciliation
    (already committed, using the original snapshot) and catalog
    activation — `_load_and_index_document()`'s final pre-activation
    revalidation must detect this and fail the whole ingestion closed:
    no active document, no orphaned Qdrant content, physical/sidecar/DB
    state fully rolled back."""
    mutated_bytes = b"MUTATED BETWEEN INDEXING AND ACTIVATION"
    physical_path_holder = {}

    def mutate_after_indexing():
        physical_path_holder["path"].write_bytes(mutated_bytes)

    # Capture the physical path the storage step just created, so the hook
    # can mutate EXACTLY that file — wraps the real _store_document_exclusively
    # only to observe its result; behavior is otherwise untouched.
    real_store = app_documents._store_document_exclusively

    def spying_store(*args, **kwargs):
        stored = real_store(*args, **kwargs)
        physical_path_holder["path"] = stored.physical_path
        return stored

    import unittest.mock as _mock
    with _mock.patch.object(app_documents, "_store_document_exclusively", spying_store):
        result = await app_documents.ingest_document(
            file_bytes=b"Original content bound to indexing and activation.",
            extension=".txt",
            display_name="notes.txt",
            owner_user_id=owner_uuid,
            _test_post_index_hook=mutate_after_indexing,
        )

    assert result.success is False
    assert result.error_type == "SourceMutatedError"
    assert result.cleanup_complete is True
    physical_files = [p for p in real_vector_index.iterdir() if not p.name.endswith(".meta.json")]
    assert physical_files == []
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []
    assert app_documents.get_vector_index().get_stats(requesting_user_uuid=str(owner_uuid))["total_documents"] == 0


@pytest.mark.asyncio
async def test_source_unmutated_between_indexing_and_activation_still_succeeds(postgres_db, owner_uuid, real_vector_index):
    """Regression guard: the hook firing with NO mutation (the ordinary
    case) must not itself cause any failure — Blocker 5's extra
    revalidation read must be a genuine no-op when the source is stable."""
    hook_calls = {"n": 0}

    def observe_only():
        hook_calls["n"] += 1

    result = await app_documents.ingest_document(
        file_bytes=b"Stable content, never mutated during ingestion.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
        _test_post_index_hook=observe_only,
    )

    assert result.success is True
    assert hook_calls["n"] == 1
    row = _catalog_row(result.stored.document_uuid)
    assert row is not None
    assert row.status == "active"


# ---------------------------------------------------------------------------
# H. Stage 5C corrective pass #5 (Blocker 1): live ingestion must freshly
# revalidate the durable v3 sidecar itself — not just the physical file and
# the PostgreSQL catalog — before ever reporting success. An independent
# audit mutated the sidecar (owner/display_name/content_sha256, or replaced
# it with malformed content) AFTER storage while leaving the physical file
# byte-for-byte unchanged, and ingestion still reported success with an
# active catalog row + indexed Qdrant content that all agreed with each
# other but disagreed with the durable sidecar. Uses the SAME
# `_test_post_index_hook` deterministic seam as Blocker 5's tests above —
# the mutation happens at an EXACT, named point in the pipeline (after
# Qdrant indexing, before the physical-file/sidecar revalidation), never a
# timing race.
# ---------------------------------------------------------------------------

def _spy_on_store(monkeypatch):
    """Wrap _store_document_exclusively so a test's `_test_post_index_hook`
    can reach the exact StoredUpload the storage step produced (physical_
    path, sidecar_path, content_sha256, owner_user_id) — same pattern as
    test_source_mutated_between_indexing_and_activation_fails_closed above,
    factored out for reuse across every sidecar-mutation test below."""
    holder = {}
    real_store = app_documents._store_document_exclusively

    def spying_store(*args, **kwargs):
        stored = real_store(*args, **kwargs)
        holder["stored"] = stored
        return stored

    monkeypatch.setattr(app_documents, "_store_document_exclusively", spying_store)
    return holder


def _assert_fully_rolled_back(real_vector_index, owner_uuid):
    physical_files = [p for p in real_vector_index.iterdir() if not p.name.endswith(".meta.json")]
    assert physical_files == [], "physical file must be removed on a sidecar-consistency failure"
    sidecar_files = [p for p in real_vector_index.iterdir() if p.name.endswith(".meta.json")]
    assert sidecar_files == [], "sidecar (even the mutated one) must be removed on a sidecar-consistency failure"
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == [], "no active OR pending catalog row may survive a sidecar-consistency failure"
    assert app_documents.get_vector_index().get_stats(requesting_user_uuid=str(owner_uuid))["total_documents"] == 0


@pytest.mark.asyncio
async def test_sidecar_owner_changed_after_storage_fails_closed(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """The exact defect an independent audit reproduced: the durable
    sidecar's owner_user_uuid is mutated AFTER storage (the physical file
    is never touched) — the final success boundary must freshly validate
    the sidecar's CURRENT owner, never just the physical file/catalog."""
    holder = _spy_on_store(monkeypatch)
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(770000200)
    original_bytes_holder = {}

    def mutate_owner():
        stored = holder["stored"]
        original_bytes_holder["physical"] = stored.physical_path.read_bytes()
        data = json.loads(stored.sidecar_path.read_text(encoding="utf-8"))
        data["owner_user_uuid"] = str(other_owner)
        write_sidecar_atomic(stored.sidecar_path, data)

    result = await app_documents.ingest_document(
        file_bytes=b"Content whose sidecar owner is mutated after storage.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
        _test_post_index_hook=mutate_owner,
    )

    assert result.success is False
    assert result.error_type == "SidecarConsistencyError"
    assert result.cleanup_complete is True
    # The physical file itself was never touched by the mutation (item 6:
    # "physical file unchanged while sidecar alone changes") — captured
    # inside the hook, before cleanup removed it.
    assert original_bytes_holder["physical"] == b"Content whose sidecar owner is mutated after storage."
    _assert_fully_rolled_back(real_vector_index, owner_uuid)


@pytest.mark.asyncio
async def test_sidecar_display_name_changed_after_storage_fails_closed(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Same class of defect, for display_name specifically."""
    holder = _spy_on_store(monkeypatch)

    def mutate_display_name():
        stored = holder["stored"]
        data = json.loads(stored.sidecar_path.read_text(encoding="utf-8"))
        data["display_name"] = "renamed-after-storage.txt"
        write_sidecar_atomic(stored.sidecar_path, data)

    result = await app_documents.ingest_document(
        file_bytes=b"Content whose sidecar display_name is mutated after storage.",
        extension=".txt",
        display_name="original-name.txt",
        owner_user_id=owner_uuid,
        _test_post_index_hook=mutate_display_name,
    )

    assert result.success is False
    assert result.error_type == "SidecarConsistencyError"
    assert result.cleanup_complete is True
    _assert_fully_rolled_back(real_vector_index, owner_uuid)


@pytest.mark.asyncio
async def test_sidecar_hash_changed_after_storage_fails_closed(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Same class of defect, for content_sha256 specifically — the
    physical file's ACTUAL bytes are unchanged (the physical-file
    revalidation immediately above this check would not catch this on its
    own), but the sidecar's RECORDED hash no longer matches them."""
    holder = _spy_on_store(monkeypatch)

    def mutate_hash():
        stored = holder["stored"]
        data = json.loads(stored.sidecar_path.read_text(encoding="utf-8"))
        data["content_sha256"] = "b" * 64
        write_sidecar_atomic(stored.sidecar_path, data)

    result = await app_documents.ingest_document(
        file_bytes=b"Content whose sidecar content_sha256 is mutated after storage.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
        _test_post_index_hook=mutate_hash,
    )

    assert result.success is False
    assert result.error_type == "SidecarConsistencyError"
    assert result.cleanup_complete is True
    _assert_fully_rolled_back(real_vector_index, owner_uuid)


@pytest.mark.asyncio
async def test_sidecar_malformed_replaced_after_storage_fails_closed(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """The sidecar is replaced with malformed (non-JSON) content after
    storage — parse_sidecar_bytes() itself rejects it; the fresh
    revalidation must surface that as the same SidecarConsistencyError
    fail-closed outcome, never let a malformed sidecar slip through."""
    holder = _spy_on_store(monkeypatch)

    def replace_with_malformed():
        stored = holder["stored"]
        stored.sidecar_path.write_text("not valid json{", encoding="utf-8")

    result = await app_documents.ingest_document(
        file_bytes=b"Content whose sidecar is replaced with malformed data after storage.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
        _test_post_index_hook=replace_with_malformed,
    )

    assert result.success is False
    assert result.error_type == "SidecarConsistencyError"
    assert result.cleanup_complete is True
    _assert_fully_rolled_back(real_vector_index, owner_uuid)


@pytest.mark.asyncio
async def test_sidecar_rewritten_with_identical_content_still_succeeds(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """Regression guard: a legitimate rewrite of the sidecar that changes
    NOTHING (byte-identical field values, just re-serialized) must not
    itself cause a failure — the fresh validation compares field VALUES,
    never object/file identity."""
    holder = _spy_on_store(monkeypatch)

    def rewrite_identical():
        stored = holder["stored"]
        data = json.loads(stored.sidecar_path.read_text(encoding="utf-8"))
        write_sidecar_atomic(stored.sidecar_path, data)

    result = await app_documents.ingest_document(
        file_bytes=b"Content whose sidecar is rewritten with identical values after storage.",
        extension=".txt",
        display_name="notes.txt",
        owner_user_id=owner_uuid,
        _test_post_index_hook=rewrite_identical,
    )

    assert result.success is True
    row = _catalog_row(result.stored.document_uuid)
    assert row is not None
    assert row.status == "active"


# ---------------------------------------------------------------------------
# H. Stage 7A-3: 'deleting' as a third legal status, list_active_by_owner_sync(),
# and the atomic active -> deleting transition primitive.
# ---------------------------------------------------------------------------

def _make_active_document(owner_user_id: uuid.UUID, *, display_name: str = "notes.txt") -> uuid.UUID:
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_user_id,
        stored_name=f"{doc_id.hex}.txt", display_name=display_name, content_sha256="d" * 64,
    )
    db_documents.mark_active_sync(document_id=doc_id)
    return doc_id


def test_deleting_is_now_a_legal_status(postgres_db, owner_uuid):
    """Companion to test_status_check_constraint_rejects_an_invalid_status()
    above (which proves 'deleted' is still rejected) — 'deleting' must now
    be accepted by the live CHECK constraint."""
    doc_id = _make_active_document(owner_uuid)
    with Session(get_sync_engine()) as session:
        session.execute(Document.__table__.update().where(Document.id == doc_id).values(status="deleting"))
        session.commit()
    assert _catalog_row(doc_id).status == "deleting"


def test_pending_and_active_remain_legal_statuses(postgres_db, owner_uuid):
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="f" * 64,
    )
    assert _catalog_row(doc_id).status == "pending"
    db_documents.mark_active_sync(document_id=doc_id)
    assert _catalog_row(doc_id).status == "active"


def test_unknown_status_still_rejected_after_adding_deleting(postgres_db, owner_uuid):
    doc_id = _make_active_document(owner_uuid)
    with Session(get_sync_engine()) as session:
        with pytest.raises(IntegrityError):
            session.execute(Document.__table__.update().where(Document.id == doc_id).values(status="deleted"))


def test_list_active_by_owner_sync_returns_only_active_rows_for_the_owner(postgres_db, owner_uuid):
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(770000201)
    active_id = _make_active_document(owner_uuid, display_name="active.txt")

    pending_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=pending_id, owner_user_id=owner_uuid,
        stored_name=f"{pending_id.hex}.txt", display_name="pending.txt", content_sha256="1" * 64,
    )
    deleting_id = _make_active_document(owner_uuid, display_name="deleting.txt")
    with Session(get_sync_engine()) as session:
        session.execute(Document.__table__.update().where(Document.id == deleting_id).values(status="deleting"))
        session.commit()
    _make_active_document(other_owner, display_name="someone-elses.txt")

    results = db_documents.list_active_by_owner_sync(owner_user_id=owner_uuid, limit=20, offset=0)
    assert [r.id for r in results] == [active_id]
    assert results[0].display_name == "active.txt"
    assert results[0].created_at == _catalog_row(active_id).created_at


def test_list_active_by_owner_sync_orders_newest_first_with_deterministic_pagination(postgres_db, owner_uuid):
    ids = [_make_active_document(owner_uuid, display_name=f"doc-{i}.txt") for i in range(5)]

    # Force an identical created_at across every row (one UPDATE, one
    # transaction -> one now()) so ordering is driven entirely by the
    # documented deterministic tie-breaker (id DESC), never by incidental
    # timing between the inserts above.
    with Session(get_sync_engine()) as session:
        session.execute(
            Document.__table__.update().where(Document.owner_user_id == owner_uuid).values(created_at=text("now()"))
        )
        session.commit()

    expected_order = sorted(ids, reverse=True)

    page1 = db_documents.list_active_by_owner_sync(owner_user_id=owner_uuid, limit=2, offset=0)
    page2 = db_documents.list_active_by_owner_sync(owner_user_id=owner_uuid, limit=2, offset=2)
    page3 = db_documents.list_active_by_owner_sync(owner_user_id=owner_uuid, limit=2, offset=4)

    assert [r.id for r in page1] == expected_order[0:2]
    assert [r.id for r in page2] == expected_order[2:4]
    assert [r.id for r in page3] == expected_order[4:5]


def test_begin_or_resume_delete_sync_transitions_active_to_deleting_and_returns_stored_name(postgres_db, owner_uuid):
    doc_id = _make_active_document(owner_uuid)
    stored_name = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)
    assert stored_name == f"{doc_id.hex}.txt"  # the catalog row's own stored_name, from the same statement
    assert _catalog_row(doc_id).status == "deleting"


def test_begin_or_resume_delete_sync_resumes_an_existing_deleting_row(postgres_db, owner_uuid):
    doc_id = _make_active_document(owner_uuid)
    first = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)
    assert first == f"{doc_id.hex}.txt"

    # Own 'deleting' row: still authorized, still returns stored_name — the
    # caller never needs to distinguish "started" from "resumed".
    second = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)
    assert second == first
    assert _catalog_row(doc_id).status == "deleting"


def test_begin_or_resume_delete_sync_foreign_owner_cannot_transition(postgres_db, owner_uuid):
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(770000202)
    doc_id = _make_active_document(owner_uuid)

    result = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=other_owner)
    assert result is None
    assert _catalog_row(doc_id).status == "active"  # untouched


def test_begin_or_resume_delete_sync_foreign_already_deleting_row_cannot_be_resumed(postgres_db, owner_uuid):
    """A foreign owner must not be able to resume (or even observe) someone
    else's in-flight deletion: None, and the row is not written at all
    (updated_at unchanged)."""
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(770000203)
    doc_id = _make_active_document(owner_uuid)
    assert db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid) is not None
    before = _catalog_row(doc_id)

    result = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=other_owner)

    assert result is None
    after = _catalog_row(doc_id)
    assert after.status == "deleting"
    assert after.updated_at == before.updated_at


def test_begin_or_resume_delete_sync_pending_row_cannot_transition(postgres_db, owner_uuid):
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="notes.txt", content_sha256="2" * 64,
    )
    result = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)
    assert result is None
    assert _catalog_row(doc_id).status == "pending"


def test_begin_or_resume_delete_sync_nonexistent_row_is_none(postgres_db, owner_uuid):
    assert db_documents.begin_or_resume_delete_sync(document_id=uuid.uuid4(), owner_user_id=owner_uuid) is None


def test_begin_or_resume_delete_sync_is_race_safe_against_concurrent_delete_attempts(postgres_db, owner_uuid):
    """Real-thread proof: two concurrent delete attempts by the SAME owner
    against the same active row are BOTH authorized (both receive the row's
    stored_name — one flips active->deleting, the other resumes the
    now-deleting row or blocks on the first's row lock and then sees it) —
    never one authorized and one spuriously refused. The status ends
    'deleting' either way."""
    doc_id = _make_active_document(owner_uuid)
    results = []
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait(timeout=5)
        results.append(db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert results == [f"{doc_id.hex}.txt", f"{doc_id.hex}.txt"]
    assert _catalog_row(doc_id).status == "deleting"


# ---------------------------------------------------------------------------
# Stage 7A-3 corrective pass, Finding 1: the exact UPDATE-zero-rows ->
# row-disappears -> follow-up-SELECT gap.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _row_vanishes_right_after_first_transaction_ends(document_id: uuid.UUID):
    """Deterministically stands in for a concurrent request A finishing its
    ENTIRE deletion (final catalog-row removal included) at the exact
    instant the request under test's FIRST database transaction has ended
    and before it does anything else — no threads, no sleeps.

    Fires exactly once, on the first Session commit-or-rollback observed
    after arming: SQLAlchemy Session `after_commit`/`after_rollback` events
    run AFTER the real DBAPI commit/rollback (so the request under test no
    longer holds any row lock and the removal below cannot deadlock against
    it), and the removal itself uses a separate Core connection (never a
    Session, so it cannot re-trigger these events)."""
    engine = get_sync_engine()
    fired: list = []

    def remove_row(_session):
        if fired:
            return
        fired.append(True)
        with engine.begin() as conn:
            conn.execute(Document.__table__.delete().where(Document.id == document_id))

    event.listen(Session, "after_commit", remove_row)
    event.listen(Session, "after_rollback", remove_row)
    try:
        yield fired
    finally:
        event.remove(Session, "after_commit", remove_row)
        event.remove(Session, "after_rollback", remove_row)


def _legacy_two_step_begin_or_resume_delete(document_id: uuid.UUID, owner_user_id: uuid.UUID) -> str:
    """The PRE-corrective-pass implementation, reproduced (test-only
    control): conditional UPDATE active->deleting, then — only when it
    matched zero rows — a SEPARATE SELECT to recognize an own 'deleting'
    row. Exists solely to prove the harness above actually reproduces the
    audited gap (a harness that cannot fail the old design proves nothing
    about the new one)."""
    with Session(get_sync_engine()) as session:
        result = session.execute(
            Document.__table__.update()
            .where(Document.id == document_id, Document.owner_user_id == owner_user_id, Document.status == "active")
            .values(status="deleting")
        )
        if result.rowcount == 1:
            session.commit()
            return "started"
        session.rollback()
        row = session.execute(
            select(Document.id).where(
                Document.id == document_id, Document.owner_user_id == owner_user_id, Document.status == "deleting"
            )
        ).first()
        return "resumed" if row is not None else "not_found"


def test_harness_reproduces_the_audited_gap_against_the_legacy_two_step_shape(postgres_db, owner_uuid):
    """CONTROL: against the old UPDATE-then-SELECT shape, an own 'deleting'
    row that vanishes between the two statements is reported not_found (the
    spurious 404 the audit reproduced). Proves the regression below is
    sensitive to exactly this window."""
    doc_id = _make_active_document(owner_uuid)
    assert db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid) is not None  # request A

    with _row_vanishes_right_after_first_transaction_ends(doc_id) as fired:
        legacy_outcome = _legacy_two_step_begin_or_resume_delete(doc_id, owner_uuid)  # request B, old shape

    assert fired
    assert legacy_outcome == "not_found"  # the spurious-404 window, reproduced
    assert _catalog_row(doc_id) is None


def test_begin_or_resume_delete_sync_has_no_zero_rows_then_reselect_window(postgres_db, owner_uuid):
    """Finding 1 regression: request B reaches an own 'deleting' row (A already
    flipped it); A then completes the ENTIRE deletion, removing the catalog
    row, immediately after B's first transaction ends. B's authorization
    was established by that ONE atomic statement while the row existed, so B
    still holds stored_name — there is no follow-up authorization SELECT
    left to come back empty."""
    doc_id = _make_active_document(owner_uuid)
    assert db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid) is not None  # request A

    with _row_vanishes_right_after_first_transaction_ends(doc_id) as fired:
        result = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)  # request B

    assert fired
    assert result == f"{doc_id.hex}.txt"
    assert _catalog_row(doc_id) is None


def test_begin_or_resume_delete_sync_after_the_row_is_fully_gone_is_none(postgres_db, owner_uuid):
    """The other side of the same contract: a request whose FIRST statement
    only runs after the row is already fully removed is not authorized."""
    doc_id = _make_active_document(owner_uuid)
    assert db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid) is not None
    db_documents.delete_sync(document_id=doc_id)

    assert db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid) is None
