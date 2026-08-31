"""
Stage 5C corrective pass #7 regression tests (the one remaining release
blocker from the prior FULL acceptance review):

HIGH — a late-aborted legacy migration may delete an identical PostgreSQL
pending catalog row that existed before the current migration attempt.

`scripts/migrate_sidecars_v2_to_v3.py`'s `_reconcile_catalog_row()` used to
return nothing — its caller (`apply_plan()`) could only ever ask "does the
row's CURRENT state still match what I would have written?" when deciding
whether an aborting migration attempt may clean up a pending catalog row it
may have created. An independent real-PostgreSQL reproduction showed that
question is not sufficient: an IDENTICAL pending row that already existed
BEFORE this migration attempt ever ran also "still matches" — and was being
destroyed right alongside a genuinely attempt-created one, the moment the
v2 source mutated before the final pre-sidecar-write barrier and this
attempt correctly aborted.

`_reconcile_catalog_row()` now returns a `CatalogReconciliation` recording
PROVENANCE — created by this attempt, reused from a pre-existing row, or
(on a genuinely ambiguous database outcome) unknown — established at the
one place the database can answer it atomically: the INSERT itself either
succeeds (nothing was there before) or fails on the primary-key constraint
(something already was). `apply_plan()`'s abort-path cleanup now runs ONLY
when `created_by_this_attempt is True`.

All tests here run against a REAL disposable PostgreSQL container
(tests/conftest.py's postgres_container()/postgres_db()) — this defect is
inherently about real database semantics (a real PRIMARY KEY constraint,
real transactional commit ambiguity), never provable against the offline
in-memory fake catalog.
"""

import json
import uuid

import pytest

import db.documents as db_documents
import db.identity as db_identity
import scripts.migrate_sidecars_v2_to_v3 as migrate
from rag.identity import sha256_hex, upload_document_id
from rag.sidecar import load_sidecar, sidecar_path_for, write_sidecar_atomic


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture for this whole
    module (same mechanism test_stage5c_migration.py already uses) — every
    test here needs real identity resolution against postgres_db, never
    the offline fake."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's same-named autouse fixture — every test here
    exercises the REAL db.documents functions against postgres_db; the
    whole point of this corrective pass is proving real catalog-row
    provenance, never the offline in-memory fake's behavior."""
    yield


def _write_v2_upload(uploads_dir, uuid_hex, content, display_name, owner_telegram_id):
    uploads_dir.mkdir(parents=True, exist_ok=True)
    physical = uploads_dir / f"{uuid_hex}.txt"
    physical.write_bytes(content)
    document_id = upload_document_id(uuid_hex)
    v2_sidecar = {
        "schema_version": 2,
        "document_id": document_id,
        "display_name": display_name,
        "stored_name": physical.name,
        "content_sha256": sha256_hex(content),
        "owner_user_id": owner_telegram_id,
    }
    sidecar_path_for(physical).write_text(json.dumps(v2_sidecar), encoding="utf-8")
    return physical, document_id


def _mutate_display_name(physical, new_display_name):
    """Deterministic barrier-3 trigger: mutate the still-v2 sidecar's
    display_name in place — the exact class of post-reconciliation
    mutation Stage 5C corrective pass #5 already proved forces the final
    barrier to abort, reused here as this pass's trigger since it never
    touches the physical file's bytes at all."""
    def _hook(candidate):
        sidecar_path = sidecar_path_for(physical)
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["display_name"] = new_display_name
        write_sidecar_atomic(sidecar_path, data)
    return _hook


# ---------------------------------------------------------------------------
# Required test 1: pre-existing identical pending row survives an abort.
# ---------------------------------------------------------------------------

def test_pre_existing_identical_pending_row_survives_migration_abort(postgres_db, tmp_path):
    """The exact scenario the FULL acceptance review reproduced: a pending
    catalog row identical to what migration would itself write already
    exists BEFORE migration ever runs. Migration reuses it, reaches final
    validation, the v2 source mutates, migration aborts correctly — and
    the PRE-EXISTING row must survive untouched, because this attempt
    never created it."""
    uploads_dir = tmp_path / "uploads"
    content = b"legacy content whose catalog row pre-exists identically before migration"
    telegram_id = 991100001
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "original-name.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)

    owner_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=owner_uuid,
        stored_name=physical.name, display_name="original-name.txt", content_sha256=sha256_hex(content),
    )
    pre_existing_row = db_documents.get_sync(document_id=document_uuid)
    assert pre_existing_row is not None and pre_existing_row.status == "pending"

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=_mutate_display_name(physical, "renamed-after-reconciliation.txt"),
    )

    assert migrated == 0, "a source mutated after catalog reconciliation must never be migrated"
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # v2 record preserved
    assert loaded["display_name"] == "renamed-after-reconciliation.txt"

    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None, (
        "the pre-existing row must SURVIVE this attempt's abort — it predates this attempt "
        "and this attempt never created it"
    )
    assert row.display_name == "original-name.txt"
    assert row.owner_user_id == owner_uuid
    assert row.stored_name == physical.name
    assert row.content_sha256 == sha256_hex(content)
    assert row.status == "pending"


# ---------------------------------------------------------------------------
# Required test 2: a row genuinely created by this attempt is still cleaned
# up on abort (regression preservation of the existing, correct behavior —
# also already proven for owner/display_name/stored_name mutations in
# tests/test_stage5c_migration.py; reproduced once more here, explicitly
# alongside test 1's opposite outcome, for direct side-by-side contrast).
# ---------------------------------------------------------------------------

def test_attempt_created_pending_row_is_removed_on_migration_abort(postgres_db, tmp_path):
    """No PRE-EXISTING row: this attempt's own _reconcile_catalog_row()
    call is the one that inserts the pending row. When migration later
    aborts (source mutated after reconciliation), the row it itself
    created is cleaned up — the existing, already-accepted behavior,
    proven again here to contrast directly against test 1 above."""
    uploads_dir = tmp_path / "uploads"
    content = b"legacy content with no pre-existing catalog row at all"
    telegram_id = 991100002
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "original-name.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)

    assert db_documents.get_sync(document_id=document_uuid) is None, "sanity check: no pre-existing row"

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=_mutate_display_name(physical, "renamed-after-reconciliation.txt"),
    )

    assert migrated == 0
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2
    assert loaded["display_name"] == "renamed-after-reconciliation.txt"

    row = db_documents.get_sync(document_id=document_uuid)
    assert row is None, "the row this attempt itself created must be cleaned up rather than left dangling"


# ---------------------------------------------------------------------------
# Required tests 3/4: unchanged candidates still migrate normally, whether
# the catalog row is reused (pre-existing) or created fresh by this attempt.
# ---------------------------------------------------------------------------

def test_pre_existing_identical_pending_row_with_unchanged_candidate_migrates_normally(postgres_db, tmp_path):
    uploads_dir = tmp_path / "uploads"
    content = b"legacy content whose catalog row pre-exists identically and nothing mutates"
    telegram_id = 991100003
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "steady-name.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)

    owner_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=owner_uuid,
        stored_name=physical.name, display_name="steady-name.txt", content_sha256=sha256_hex(content),
    )

    plan = migrate.build_plan(uploads_dir)
    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 1
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 3
    assert loaded["owner_user_uuid"] == str(owner_uuid)
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.status == "pending"
    assert row.display_name == "steady-name.txt"


def test_attempt_created_row_with_unchanged_candidate_migrates_normally(postgres_db, tmp_path):
    uploads_dir = tmp_path / "uploads"
    content = b"legacy content with no pre-existing row and nothing mutates"
    telegram_id = 991100004
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "steady-name.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)

    plan = migrate.build_plan(uploads_dir)
    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 1
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 3
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.status == "pending"


# ---------------------------------------------------------------------------
# Required test 5: a pre-existing MISMATCHING row keeps migration fail-
# closed, and is never touched by any reconciliation-driven cleanup.
# ---------------------------------------------------------------------------

def test_pre_existing_mismatching_row_remains_fail_closed_and_untouched(postgres_db, tmp_path):
    """A disagreeing existing row is never a candidate for cleanup at all
    — _reconcile_catalog_row() raises CatalogConsistencyError before
    apply_plan() ever reaches the abort-path cleanup code, so this row
    survives regardless of provenance tracking. Proven again here
    (alongside tests/test_stage5c_migration.py's own equivalent) as this
    pass's required regression-preservation proof."""
    uploads_dir = tmp_path / "uploads"
    content = b"content whose pre-existing catalog row disagrees on owner"
    telegram_id = 991100005
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "notes.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)

    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(991100006)
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=other_owner,
        stored_name=physical.name, display_name="notes.txt", content_sha256=sha256_hex(content),
    )

    plan = migrate.build_plan(uploads_dir)
    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 0
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.owner_user_id == other_owner, "the disagreeing row must never be overwritten or deleted"


# ---------------------------------------------------------------------------
# Required test 6: a matching row that progresses to 'active' before abort
# cleanup runs must survive — defense-in-depth on top of provenance
# tracking (the existing exact conditional DELETE's own status='pending'
# predicate).
# ---------------------------------------------------------------------------

def test_row_progressed_to_active_before_abort_cleanup_survives(postgres_db, tmp_path):
    """This attempt DOES create the pending row (no pre-existing row), so
    created_by_this_attempt is True — but between the catalog write and
    the final pre-sidecar-write barrier, some other process (simulated
    deterministically via the test hook) activates it. The final barrier
    then aborts (a source mutation, applied in the same hook). Cleanup
    must still refuse to delete it: the exact conditional DELETE's own
    status='pending' predicate no longer matches an 'active' row."""
    uploads_dir = tmp_path / "uploads"
    content = b"legacy content whose freshly created row progresses to active before abort"
    telegram_id = 991100007
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "notes.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)

    assert db_documents.get_sync(document_id=document_uuid) is None, "sanity check: no pre-existing row"

    plan = migrate.build_plan(uploads_dir)

    def progress_to_active_then_mutate(candidate):
        db_documents.mark_active_sync(document_id=document_uuid)
        sidecar_path = sidecar_path_for(physical)
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["display_name"] = "renamed-while-active.txt"
        write_sidecar_atomic(sidecar_path, data)

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=progress_to_active_then_mutate,
    )

    assert migrated == 0
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # v2 record preserved

    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None, "an active row must never be deleted by migration-attempt cleanup"
    assert row.status == "active"


# ---------------------------------------------------------------------------
# Required test 7: concurrent identical-row creation — this attempt must
# never later claim ownership of another transaction's row and delete it.
# Deterministic (no sleep): a monkeypatched create_pending_sync wrapper
# makes a SEPARATE, genuinely committed insert win the race before this
# attempt's own insert runs, so this attempt's own INSERT hits the real
# PostgreSQL primary-key constraint exactly as true concurrency would.
# ---------------------------------------------------------------------------

def test_concurrent_identical_row_creation_is_never_later_claimed_by_this_attempt(postgres_db, tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    content = b"legacy content whose catalog row is created by a genuinely concurrent transaction"
    telegram_id = 991100008
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "notes.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)
    owner_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    real_create_pending = db_documents.create_pending_sync
    race_state = {"other_transaction_inserted": False}

    def racing_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        if not race_state["other_transaction_inserted"]:
            race_state["other_transaction_inserted"] = True
            # A genuinely separate transaction wins the race and commits
            # the IDENTICAL row first — deterministic stand-in for real
            # concurrency landing in the window between this attempt's own
            # get_sync() lookup and its own INSERT.
            real_create_pending(
                document_id=document_id, owner_user_id=owner_user_id, stored_name=stored_name,
                display_name=display_name, content_sha256=content_sha256,
            )
        # This attempt's own insert now hits the real PRIMARY KEY
        # constraint, exactly as true concurrency would.
        real_create_pending(
            document_id=document_id, owner_user_id=owner_user_id, stored_name=stored_name,
            display_name=display_name, content_sha256=content_sha256,
        )

    monkeypatch.setattr(db_documents, "create_pending_sync", racing_create_pending)

    plan = migrate.build_plan(uploads_dir)

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=_mutate_display_name(physical, "renamed-after-race.txt"),
    )

    assert migrated == 0, "a source mutated after catalog reconciliation must never be migrated"
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2
    assert loaded["display_name"] == "renamed-after-race.txt"

    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None, (
        "the OTHER transaction's row must survive — this attempt lost the race and must never "
        "claim ownership of it merely because it still matches"
    )
    assert row.owner_user_id == owner_uuid
    assert row.display_name == "notes.txt"
    assert row.status == "pending"


# ---------------------------------------------------------------------------
# Required test 8: an uncertain insert outcome (ambiguous commit-then-raise,
# the same class of event db.documents.reconcile_ambiguous_create_pending_
# sync()'s own docstring documents for the normal ingestion path) must
# never be treated as created_by_this_attempt=True, and must trigger no
# destructive cleanup at all.
# ---------------------------------------------------------------------------

def test_ambiguous_insert_outcome_triggers_no_destructive_cleanup(postgres_db, tmp_path, monkeypatch):
    """create_pending_sync()'s commit genuinely succeeds against REAL
    PostgreSQL, then the call still raises (simulating a lost
    acknowledgement) — the same ambiguous-commit scenario Codex originally
    reproduced for app/documents.py's normal ingestion path. Migration must
    treat authorship as UNKNOWN (never True), leave this candidate's v2
    sidecar completely untouched, and — critically — never attempt to
    delete the row that, in fact, WAS genuinely committed."""
    uploads_dir = tmp_path / "uploads"
    content = b"legacy content whose create_pending_sync commit is ambiguous"
    telegram_id = 991100009
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "notes.txt", telegram_id)
    document_uuid = uuid.UUID(uuid_hex)

    real_create_pending = db_documents.create_pending_sync

    def ambiguous_create_pending(*, document_id, owner_user_id, stored_name, display_name, content_sha256):
        real_create_pending(
            document_id=document_id, owner_user_id=owner_user_id, stored_name=stored_name,
            display_name=display_name, content_sha256=content_sha256,
        )
        raise RuntimeError("simulated: commit succeeded but the acknowledgement was lost")

    monkeypatch.setattr(db_documents, "create_pending_sync", ambiguous_create_pending)

    plan = migrate.build_plan(uploads_dir)
    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 0, "an unconfirmed catalog reconciliation must never be reported as migrated"
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # v2 sidecar never rewritten from an unproven catalog state
    assert loaded["owner_user_id"] == telegram_id

    # The row DID genuinely commit (real PostgreSQL) — it must survive:
    # migration never attempted any cleanup because it could not prove it
    # created it.
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None, "the ambiguously-but-genuinely-committed row must not be destructively cleaned up"
    assert row.status == "pending"
