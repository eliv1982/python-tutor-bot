"""
Stage 5C regression tests: scripts/migrate_sidecars_v2_to_v3.py — the
one-time bridge from legacy Telegram-integer-owned (v2) sidecars to
canonical-UUID-owned (v3) sidecars.

Section A (offline, no PostgreSQL): build_plan()'s pure filesystem
validation logic — v2 candidates found, v1/v3/invalid/mismatched sources
skipped with safe reasons, apply_plan()'s pure rewrite logic against a
deterministic fake resolver (never touches PostgreSQL itself).

Section B (against a REAL disposable PostgreSQL container): the full
migrate -> rebuild pipeline end to end — legacy v2 sidecars on disk,
migrated via the real db.identity resolver, picked up by a real
scripts.rebuild_qdrant --apply run into the current (UUID-owned)
collection, with isolation preserved.
"""

import uuid

import pytest

import scripts.migrate_sidecars_v2_to_v3 as migrate
import scripts.rebuild_qdrant as rebuild
from rag.identity import sha256_hex, upload_document_id
from rag.index import VectorIndex
from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic
from rag_fakes import DeterministicFakeEmbeddings


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
    import json
    sidecar_path_for(physical).write_text(json.dumps(v2_sidecar), encoding="utf-8")
    return physical, document_id


# ---------------------------------------------------------------------------
# A. Offline: build_plan()/apply_plan() pure logic
# ---------------------------------------------------------------------------

def test_build_plan_finds_v2_candidates_and_skips_v1_and_v3(tmp_path):
    uploads_dir = tmp_path / "uploads"

    # A genuine v2 candidate.
    _write_v2_upload(uploads_dir, "1" * 32, b"legacy content one", "one.txt", 111111)

    # A v1 (unowned) sidecar — never eligible for migration (nothing to
    # resolve).
    physical_v1 = uploads_dir / ("2" * 32 + ".txt")
    physical_v1.write_bytes(b"legacy v1 content")
    import json
    sidecar_path_for(physical_v1).write_text(json.dumps({
        "schema_version": 1,
        "document_id": upload_document_id("2" * 32),
        "display_name": "two.txt",
        "stored_name": physical_v1.name,
        "content_sha256": sha256_hex(b"legacy v1 content"),
    }), encoding="utf-8")

    # An already-migrated v3 sidecar — nothing to do.
    physical_v3 = uploads_dir / ("3" * 32 + ".txt")
    physical_v3.write_bytes(b"already migrated content")
    write_sidecar_atomic(
        sidecar_path_for(physical_v3),
        build_sidecar(upload_document_id("3" * 32), "three.txt", physical_v3.name, sha256_hex(b"already migrated content"), owner_user_uuid=str(uuid.uuid4())),
    )

    plan = migrate.build_plan(uploads_dir)

    assert len(plan.candidates) == 1
    assert plan.candidates[0].telegram_owner_id == 111111
    assert "v1_no_owner_to_migrate" in plan.skipped_reasons
    assert "already_v3" in plan.skipped_reasons


def test_build_plan_skips_hash_mismatched_source(tmp_path):
    uploads_dir = tmp_path / "uploads"
    physical, _doc_id = _write_v2_upload(uploads_dir, "4" * 32, b"original content", "tampered.txt", 222222)
    physical.write_bytes(b"tampered content, different from the recorded hash")

    plan = migrate.build_plan(uploads_dir)

    assert len(plan.candidates) == 0
    assert "sidecar_content_hash_mismatch" in plan.skipped_reasons


def test_apply_plan_rewrites_sidecar_as_v3_using_the_given_resolver(tmp_path):
    uploads_dir = tmp_path / "uploads"
    physical, doc_id = _write_v2_upload(uploads_dir, "5" * 32, b"content to migrate", "mig.txt", 333333)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    resolved_uuid = uuid.uuid4()

    def fake_resolver(telegram_id):
        assert telegram_id == 333333
        return resolved_uuid

    migrated_count = migrate.apply_plan(plan, fake_resolver, uploads_dir)

    assert migrated_count == 1
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 3
    assert loaded["owner_user_uuid"] == str(resolved_uuid)
    assert loaded["document_id"] == doc_id
    # The physical file itself is never touched.
    assert physical.read_bytes() == b"content to migrate"


def test_apply_plan_is_safe_to_rerun_after_partial_migration(tmp_path):
    """A rerun against already-migrated (now v3) sidecars finds nothing
    left to migrate — build_plan() naturally excludes them (schema_version
    != 2), so a second `--apply` is a safe no-op, never a double-rewrite."""
    uploads_dir = tmp_path / "uploads"
    _write_v2_upload(uploads_dir, "6" * 32, b"content", "notes.txt", 444444)

    plan1 = migrate.build_plan(uploads_dir)
    migrate.apply_plan(plan1, lambda tid: uuid.uuid4(), uploads_dir)

    plan2 = migrate.build_plan(uploads_dir)
    assert len(plan2.candidates) == 0
    assert "already_v3" in plan2.skipped_reasons


def test_apply_plan_leaves_unmapped_or_disallowed_owner_completely_unmigrated(tmp_path):
    """Stage 5C corrective pass ownership rule, pure-logic proof:
    `resolve_owner_uuid` returning None (the fail-closed contract every
    production caller — _resolve_owner_uuid_fail_closed — honors for an
    unmapped/disallowed legacy Telegram id) must leave that candidate's
    sidecar completely untouched and must not count toward `migrated`."""
    uploads_dir = tmp_path / "uploads"
    physical, doc_id = _write_v2_upload(uploads_dir, "7" * 32, b"content", "notes.txt", 555555)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    migrated_count = migrate.apply_plan(plan, lambda telegram_id: None, uploads_dir)

    assert migrated_count == 0
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # never rewritten
    assert loaded["owner_user_id"] == 555555


# ---------------------------------------------------------------------------
# B. Against a real disposable PostgreSQL container: full migrate -> rebuild
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's same-named autouse fixture for this WHOLE
    module (same mechanism tests/test_stage1c_access_control.py already
    uses) — safe to do unconditionally here: Section A's tests never call
    db.identity.resolve_or_create_user_by_telegram_id_sync at all (they
    pass their own local resolver callables straight into
    migrate.apply_plan()), and Section B's one test needs the REAL
    resolver against postgres_db."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog(request, monkeypatch):
    """
    Stage 5C corrective pass #2, Section 5 (previous-audit finding): this
    module used to shadow ONLY `_default_fake_preferences` above, which
    left conftest.py's autouse in-memory `db.documents` fake silently
    active even for Section B's tests below — despite those tests taking
    `postgres_db` and being titled "against a REAL disposable PostgreSQL
    container", every `db_documents.get_sync()`/`create_pending_sync()`
    call they made was secretly hitting a plain dict, never PostgreSQL.

    Section A's tests are genuinely offline (no `postgres_db`, and
    `DATABASE_URL` stays deliberately poisoned — see conftest.py) and
    still need the fake: `migrate.apply_plan()`'s internal
    `_reconcile_catalog_row()` calls `db.documents.get_sync()`/
    `create_pending_sync()` regardless of which resolver callable a test
    passes in. Section B's tests need the OPPOSITE: real `db.documents`
    against the real `postgres_db` container they already set up. Decided
    per-test by whether `postgres_db` is among that test's own requested
    fixtures — never by which section a test happens to sit in physically,
    so a future test in either section is correct by construction.
    """
    if "postgres_db" in request.fixturenames:
        yield  # Real db.documents — no monkeypatch applied.
        return
    from conftest import _install_fake_documents_catalog
    _install_fake_documents_catalog(monkeypatch)
    yield


def test_full_migration_and_rebuild_pipeline_preserves_isolation(postgres_db, tmp_path, monkeypatch):
    """End-to-end Stage 5C transition strategy proof: legacy v2 sidecars on
    disk -> scripts.migrate_sidecars_v2_to_v3 (resolving REAL internal
    UUIDs via db.identity against postgres_db) -> v3 sidecars ->
    scripts.rebuild_qdrant --apply into a fresh collection, with per-user
    isolation intact end to end."""
    import db.identity as db_identity

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    import config as app_config
    for filename in app_config.BUILTIN_REFERENCE_FILES:
        (documents_dir / filename).write_text(f"Reference content for {filename}.", encoding="utf-8")
    uploads_dir = documents_dir / "uploads"

    import rag.loader as rag_loader
    monkeypatch.setattr(rag_loader, "MANAGED_UPLOADS_DIR", uploads_dir)

    telegram_a, telegram_b = 991000001, 991000002
    content_a = b"User A's legacy private notes, still Telegram-owned pre-migration."
    content_b = b"User B's legacy private notes, still Telegram-owned pre-migration."
    _write_v2_upload(uploads_dir, "a" * 32, content_a, "a.txt", telegram_a)
    _write_v2_upload(uploads_dir, "b" * 32, content_b, "b.txt", telegram_b)

    # Step 1: migrate (real PostgreSQL identity resolution).
    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 2
    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)
    assert migrated == 2

    owner_a = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_a)
    owner_b = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_b)
    assert owner_a != owner_b

    # Stage 5C corrective pass, Section 5: explicitly assert the
    # PostgreSQL `documents` catalog row itself — migration is incomplete
    # without it, not just a rewritten sidecar.
    import db.documents as db_documents
    row_a = db_documents.get_sync(document_id=uuid.UUID("a" * 32))
    assert row_a is not None
    assert row_a.owner_user_id == owner_a
    assert row_a.stored_name == "a" * 32 + ".txt"
    assert row_a.content_sha256 == sha256_hex(content_a)
    assert row_a.status == "pending"  # not yet reconciled into Qdrant

    # Step 2: normal rebuild now picks up the migrated v3 sidecars.
    rebuild_plan = rebuild.build_plan(documents_dir, uploads_dir)
    assert not rebuild_plan.skipped_upload_reasons
    assert len(rebuild_plan.upload_documents) == 2

    fake = DeterministicFakeEmbeddings()
    vi = VectorIndex(persist_directory=tmp_path / "qdrant", embeddings=fake, collection_name="migration_e2e_test")
    try:
        report = rebuild.apply_plan(rebuild_plan, vi)
        assert report.documents_removed == 0

        results_a = vi.similarity_search_with_score(content_a.decode(), requesting_user_uuid=str(owner_a), k=5)
        assert any(d.metadata.get("source") == "a.txt" for d, _ in results_a)
        assert all(d.metadata.get("source") != "b.txt" for d, _ in results_a)

        results_b_query_a = vi.similarity_search_with_score(content_a.decode(), requesting_user_uuid=str(owner_b), k=5)
        assert all(d.metadata.get("source") != "a.txt" for d, _ in results_b_query_a)

        # Rebuild is the deterministic reconciliation path (Section 3):
        # the catalog row, still 'pending' right after migration, is
        # confirmed 'active' only once rebuild has genuinely reconciled
        # real Qdrant content for it.
        assert db_documents.get_sync(document_id=uuid.UUID("a" * 32)).status == "active"
    finally:
        vi.close()


# ---------------------------------------------------------------------------
# B (continued). Ownership rule + restartability, against real PostgreSQL
# ---------------------------------------------------------------------------

def test_resolve_owner_uuid_fail_closed_uses_existing_mapping_even_if_not_currently_allowlisted(postgres_db, monkeypatch):
    """Rule 1 of the Stage 5C ownership contract: an EXISTING Telegram
    mapping is used unconditionally — migrating already-legitimately-owned
    data is never itself an authorization decision, so a since-revoked
    allowlist entry must not block migrating data that was already
    genuinely owned.

    tests/conftest.py's autouse `_default_test_access_allowed` fixture
    replaces `access_control.is_authorized` itself (not just the
    underlying TELEGRAM_ALLOWED_USER_IDS frozenset) with an always-True
    stub for every test by default — these three tests are specifically
    ABOUT that real gate, so they override it back to the genuine
    frozenset-backed function for the duration of each test."""
    import db.identity as db_identity
    import utils.access_control as access_control

    monkeypatch.setattr(access_control, "is_authorized", lambda uid: uid in access_control.TELEGRAM_ALLOWED_USER_IDS)

    telegram_id = 991000010
    existing_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())  # nobody currently allowlisted

    resolved = migrate._resolve_owner_uuid_fail_closed(telegram_id)

    assert resolved == existing_uuid


def test_resolve_owner_uuid_fail_closed_never_creates_a_mapping_for_an_unmapped_disallowed_id(postgres_db, monkeypatch):
    """Rule 2/3: no existing mapping AND not currently allowlisted -> None,
    and — critically — the lookup itself must never have side effects: no
    user/mapping is silently created merely by being considered."""
    import db.identity as db_identity
    import utils.access_control as access_control

    monkeypatch.setattr(access_control, "is_authorized", lambda uid: uid in access_control.TELEGRAM_ALLOWED_USER_IDS)

    telegram_id = 991000011
    assert db_identity.lookup_user_by_telegram_id_sync(telegram_id) is None
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset())

    resolved = migrate._resolve_owner_uuid_fail_closed(telegram_id)

    assert resolved is None
    assert db_identity.lookup_user_by_telegram_id_sync(telegram_id) is None


def test_resolve_owner_uuid_fail_closed_creates_a_mapping_only_when_allowlisted(postgres_db, monkeypatch):
    import db.identity as db_identity
    import utils.access_control as access_control

    monkeypatch.setattr(access_control, "is_authorized", lambda uid: uid in access_control.TELEGRAM_ALLOWED_USER_IDS)

    telegram_id = 991000012
    monkeypatch.setattr(access_control, "TELEGRAM_ALLOWED_USER_IDS", frozenset({telegram_id}))

    resolved = migrate._resolve_owner_uuid_fail_closed(telegram_id)

    assert resolved is not None
    assert db_identity.lookup_user_by_telegram_id_sync(telegram_id) == resolved


def test_apply_plan_fails_closed_on_disagreeing_existing_catalog_row(postgres_db, tmp_path):
    """Restartability (Section 4): a rerun must never silently overwrite a
    catalog row that already disagrees with the sidecar being migrated —
    fail closed, leave both the sidecar and the row exactly as they were."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    content = b"content whose catalog row already disagrees with this migration run"
    physical, doc_id = _write_v2_upload(uploads_dir, "8" * 32, content, "notes.txt", 991000020)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    # Pre-create a documents row for this exact document UUID, owned by a
    # DIFFERENT user than the one this migration run will resolve.
    document_uuid = uuid.UUID("8" * 32)
    other_owner = db_identity.resolve_or_create_user_by_telegram_id_sync(991000021)
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=other_owner,
        stored_name=physical.name, display_name="notes.txt", content_sha256=sha256_hex(content),
    )

    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 0
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # sidecar never rewritten
    row = db_documents.get_sync(document_id=document_uuid)
    assert row.owner_user_id == other_owner  # catalog row never overwritten


def test_apply_plan_rerun_against_already_reconciled_catalog_row_is_a_safe_no_op(postgres_db, tmp_path):
    """Restartability: a rerun where the catalog row ALREADY agrees with
    the (still-v2, from this run's point of view — the first run already
    migrated it) sidecar must not raise and must not double-count."""
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    content = b"content migrated once, then apply_plan runs again"
    _write_v2_upload(uploads_dir, "9" * 32, content, "notes.txt", 991000022)

    plan1 = migrate.build_plan(uploads_dir)
    migrated1 = migrate.apply_plan(plan1, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)
    assert migrated1 == 1

    # A genuine rerun finds nothing left to migrate — build_plan() already
    # excludes the now-v3 sidecar (schema_version != 2) — proving the
    # combined migrate+catalog pipeline is idempotent end to end.
    plan2 = migrate.build_plan(uploads_dir)
    assert len(plan2.candidates) == 0
    assert "already_v3" in plan2.skipped_reasons


def test_apply_plan_revalidates_and_fails_closed_when_source_mutated_after_planning(postgres_db, tmp_path):
    """Stage 5C corrective pass #2, Section 5 — the exact TOCTOU the second
    audit reproduced: plan a legacy v2 upload, mutate the PHYSICAL FILE
    after planning (before apply runs), then apply the stale plan. Apply
    must securely revalidate the source immediately before its own
    irreversible durable transition and fail this candidate closed rather
    than trusting the plan's now-stale content_sha256:
      - no PostgreSQL catalog row is created from the stale hash;
      - the sidecar is never rewritten as v3 from stale metadata;
      - the original v2 sidecar/ownership record is left completely
        untouched;
      - a rerun's own build_plan() must still see it as a fresh, correct
        v2 candidate (never permanently stuck) once the mutation is a
        thing of the past — proving this is a clean skip, not a corrupted
        half-migrated state."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    original_content = b"legacy content as it existed at plan time"
    physical, doc_id = _write_v2_upload(uploads_dir, "c" * 32, original_content, "notes.txt", 991000030)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1
    assert plan.candidates[0].content_sha256 == sha256_hex(original_content)

    # Mutate the physical file AFTER planning, before apply ever runs —
    # the plan now describes stale, no-longer-accurate state.
    mutated_content = b"attacker- or operator-mutated content, different from the plan"
    physical.write_bytes(mutated_content)

    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 0, "a mutated source must never be migrated using the stale plan-time hash"
    document_uuid = uuid.UUID("c" * 32)
    assert db_documents.get_sync(document_id=document_uuid) is None, (
        "no catalog row may be created from a stale/unrevalidated content hash"
    )
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # never rewritten as v3 from stale metadata
    assert loaded["owner_user_id"] == 991000030  # original legacy ownership record intact

    # A rerun against the STILL-mutated file must keep skipping it closed
    # — the v2 sidecar's own recorded hash still describes the ORIGINAL
    # content, so this is (correctly) indistinguishable from any other
    # sidecar/content hash mismatch until an operator resolves it. Never
    # permanently CORRUPTED (no partial catalog/v3 state was created —
    # already asserted above), just consistently held closed.
    rerun_plan_while_mutated = migrate.build_plan(uploads_dir)
    assert len(rerun_plan_while_mutated.candidates) == 0
    assert "sidecar_content_hash_mismatch" in rerun_plan_while_mutated.skipped_reasons

    # Once the source is restored to match what its OWN v2 sidecar
    # recorded, a fresh plan+apply proceeds normally — proving this is a
    # clean, recoverable skip rather than a stuck/corrupted state.
    physical.write_bytes(original_content)
    rerun_plan = migrate.build_plan(uploads_dir)
    assert len(rerun_plan.candidates) == 1
    assert rerun_plan.candidates[0].content_sha256 == sha256_hex(original_content)
    rerun_migrated = migrate.apply_plan(rerun_plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)
    assert rerun_migrated == 1
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is not None
    assert row.content_sha256 == sha256_hex(original_content)


def test_apply_plan_revalidates_and_fails_closed_when_sidecar_already_migrated_after_planning(postgres_db, tmp_path):
    """A second, concurrent apply_plan() run (or an operator re-running an
    old plan after already migrating by hand) must not re-migrate a
    candidate whose sidecar is no longer v2 by the time apply actually
    runs — revalidation must reject it the same way build_plan() would if
    run fresh, never blindly trust the plan's own recorded schema_version."""
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    content = b"content migrated out from under a stale plan"
    physical, doc_id = _write_v2_upload(uploads_dir, "d" * 32, content, "notes.txt", 991000031)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    # Migrate it for real via a fresh plan (simulating a concurrent/prior
    # apply run), so the sidecar is now v3 by the time the STALE `plan`
    # above is applied.
    fresh_plan = migrate.build_plan(uploads_dir)
    first_migrated = migrate.apply_plan(fresh_plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)
    assert first_migrated == 1

    stale_migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert stale_migrated == 0, "a candidate already migrated since planning must not be re-migrated"
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 3
    first_owner_uuid = loaded["owner_user_uuid"]
    # Re-applying the stale plan must not have overwritten the sidecar
    # with a second (potentially different) owner resolution.
    assert first_owner_uuid == str(db_identity.resolve_or_create_user_by_telegram_id_sync(991000031))


# ---------------------------------------------------------------------------
# B (continued). Stage 5C corrective pass #4 (Blocker 6): the single
# top-of-loop revalidation above closes the plan-to-apply gap, but an
# independent audit reproduced a SECOND mutation window this alone doesn't
# close: resolve_owner_uuid() (a real PostgreSQL round trip) can itself take
# arbitrary time, and the source can mutate again during it or during the
# gap between the catalog write and the sidecar write — each is a
# deterministic barrier proof (mutate exactly at that point via a seam,
# never a timing sleep).
# ---------------------------------------------------------------------------

def test_apply_plan_fails_closed_when_source_mutates_during_owner_resolution(postgres_db, tmp_path):
    """Barrier 2: a mutation landing DURING resolve_owner_uuid() — AFTER
    the top-of-loop revalidation already proved freshness, but BEFORE the
    catalog write — must still be caught. No catalog row is created from
    the now-stale hash, and the original v2 ownership record survives
    completely untouched."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    original_content = b"legacy content present during owner resolution"
    physical, doc_id = _write_v2_upload(uploads_dir, "e" * 32, original_content, "notes.txt", 991000040)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    mutated_content = b"mutated while resolve_owner_uuid was running"

    def mutating_resolver(telegram_id):
        # Deterministic barrier: the mutation happens exactly inside the
        # owner-resolution call, the real window Blocker 6 targets — never
        # a timing sleep racing a background thread.
        physical.write_bytes(mutated_content)
        return db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    migrated = migrate.apply_plan(plan, mutating_resolver, uploads_dir)

    assert migrated == 0, "a source mutated during owner resolution must never be migrated"
    document_uuid = uuid.UUID("e" * 32)
    assert db_documents.get_sync(document_id=document_uuid) is None, (
        "no catalog row may be created once the source has mutated past the revalidated snapshot"
    )
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # v2 ownership record preserved
    assert loaded["owner_user_id"] == 991000040


def test_apply_plan_fails_closed_when_source_mutates_after_catalog_reconciliation(postgres_db, tmp_path):
    """Barrier 3 — the most important one: a mutation landing AFTER the
    catalog row has already been reconciled against a genuinely validated
    snapshot, but immediately BEFORE the sidecar is overwritten (the ONE
    irreversible step this script performs — it destroys the v2 ownership
    record), must still be caught. The v2 sidecar is preserved untouched
    even though a catalog row was momentarily reconciled — never claim
    migrated when the source has since moved on, and never destroy the
    only record of legacy ownership over state that's already stale by the
    time it would be written.

    Stage 5C corrective pass #5 (Blocker 3): the pending catalog row this
    attempt created is now also cleaned up when barrier 3 aborts —
    `reconcile_ambiguous_create_pending_sync()` deletes it only because it
    still exactly matches what this attempt itself just wrote (owner/
    stored_name/display_name/content_sha256, status='pending'). Left
    behind, that row would block a rerun that resolves DIFFERENT owner/
    display_name state from ever reconciling the catalog (`_reconcile_
    catalog_row()` raises on any disagreement with an existing row rather
    than overwriting it) — see the dedicated owner/display_name-mutation
    tests below for that exact scenario. This test's own rerun happens to
    resolve identical values either way, so it is unaffected either way;
    it exists to prove the ordinary "mutated then restored" path still
    completes normally now that the interim row is cleaned up rather than
    left dangling."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    original_content = b"legacy content present when the catalog row is reconciled"
    physical, doc_id = _write_v2_upload(uploads_dir, "f" * 32, original_content, "notes.txt", 991000041)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    mutated_content = b"mutated between the catalog write and the sidecar write"

    def mutate_before_sidecar_write(candidate):
        physical.write_bytes(mutated_content)

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=mutate_before_sidecar_write,
    )

    assert migrated == 0, "the sidecar must never be overwritten once a post-catalog-write mutation is detected"
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # v2 ownership record preserved, never destroyed
    assert loaded["owner_user_id"] == 991000041

    # The catalog row WAS momentarily created (the mutation landed after
    # that write), but Blocker 3's cleanup removes it again once barrier 3
    # aborts this candidate — it exactly matched what this attempt itself
    # wrote and was still 'pending', so it is safe to clean up rather than
    # leave dangling (see this test's own docstring).
    document_uuid = uuid.UUID("f" * 32)
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is None

    # Recoverable, not permanently stuck: once the file is restored to what
    # the still-v2 sidecar/already-reconciled catalog row agree on, a fresh
    # apply completes the migration normally.
    physical.write_bytes(original_content)
    rerun_plan = migrate.build_plan(uploads_dir)
    assert len(rerun_plan.candidates) == 1
    rerun_migrated = migrate.apply_plan(rerun_plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)
    assert rerun_migrated == 1
    final_row = db_documents.get_sync(document_id=document_uuid)
    assert final_row.content_sha256 == sha256_hex(original_content)


# ---------------------------------------------------------------------------
# B (continued). Stage 5C corrective pass #5 (Blocker 3): the final barrier
# (immediately before write_sidecar_atomic() destroys the v2 record) must
# validate the COMPLETE current candidate, not merely its content hash —
# and a pending catalog row this attempt created must not make the
# candidate unrecoverable on rerun.
# ---------------------------------------------------------------------------

def test_apply_plan_fails_closed_when_owner_changes_after_catalog_reconciliation(postgres_db, tmp_path):
    """The exact defect an independent audit reproduced: the v2 sidecar's
    owner_user_id changes AFTER _reconcile_catalog_row() already
    reconciled the catalog row against the OLD owner, while the physical
    file's bytes stay completely unchanged. The previous barrier compared
    only content_sha256 and would have let write_sidecar_atomic() overwrite
    the v2 sidecar with a v3 sidecar built from the STALE (old-owner)
    candidate snapshot, permanently destroying the newer, still-legitimate
    ownership change. The fixed barrier compares the COMPLETE candidate and
    aborts instead — and cleans up the pending row this attempt itself
    created, so a rerun correctly picks up the NEWER owner rather than
    being blocked by a disagreeing leftover row."""
    import json

    import db.documents as db_documents
    import db.identity as db_identity
    from rag.sidecar import load_sidecar

    uploads_dir = tmp_path / "uploads"
    content = b"content whose sidecar owner changes after catalog reconciliation"
    uuid_hex = uuid.uuid4().hex
    original_owner = 991000050
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "notes.txt", original_owner)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    new_owner = 991000051

    def mutate_owner_before_sidecar_write(candidate):
        sidecar_path = sidecar_path_for(physical)
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["owner_user_id"] = new_owner
        write_sidecar_atomic(sidecar_path, data)

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=mutate_owner_before_sidecar_write,
    )

    assert migrated == 0, "a sidecar whose owner changed after catalog reconciliation must never be migrated from stale data"
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # never overwritten
    assert loaded["owner_user_id"] == new_owner  # the NEWER owner value survives untouched

    document_uuid = uuid.UUID(uuid_hex)
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is None, "the attempt-created pending row must be cleaned up rather than left blocking a rerun"

    # Recoverable and deterministic: a rerun now resolves the NEW owner
    # correctly, proving this is a clean skip, never a corrupted or stuck
    # half-migrated state.
    rerun_plan = migrate.build_plan(uploads_dir)
    assert len(rerun_plan.candidates) == 1
    assert rerun_plan.candidates[0].telegram_owner_id == new_owner
    rerun_migrated = migrate.apply_plan(rerun_plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)
    assert rerun_migrated == 1
    final_owner_uuid = db_identity.resolve_or_create_user_by_telegram_id_sync(new_owner)
    final_row = db_documents.get_sync(document_id=document_uuid)
    assert final_row.owner_user_id == final_owner_uuid
    final_loaded = load_sidecar(sidecar_path_for(physical))
    assert final_loaded["schema_version"] == 3
    assert final_loaded["owner_user_uuid"] == str(final_owner_uuid)


def test_apply_plan_fails_closed_when_display_name_changes_after_catalog_reconciliation(postgres_db, tmp_path):
    """Same class of defect as the owner-change test above, for
    display_name specifically — same bytes, same owner, only the v2
    sidecar's display_name changes after the catalog row was already
    reconciled against the old value."""
    import json

    import db.documents as db_documents
    import db.identity as db_identity
    from rag.sidecar import load_sidecar

    uploads_dir = tmp_path / "uploads"
    content = b"content whose sidecar display_name changes after catalog reconciliation"
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "original-name.txt", 991000052)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    def mutate_display_name_before_sidecar_write(candidate):
        sidecar_path = sidecar_path_for(physical)
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["display_name"] = "renamed-after-reconciliation.txt"
        write_sidecar_atomic(sidecar_path, data)

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=mutate_display_name_before_sidecar_write,
    )

    assert migrated == 0, "a sidecar whose display_name changed after catalog reconciliation must never be migrated from stale data"
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2
    assert loaded["display_name"] == "renamed-after-reconciliation.txt"  # newer value survives untouched

    document_uuid = uuid.UUID(uuid_hex)
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is None, "the attempt-created pending row must be cleaned up rather than left blocking a rerun"

    rerun_plan = migrate.build_plan(uploads_dir)
    assert len(rerun_plan.candidates) == 1
    assert rerun_plan.candidates[0].display_name == "renamed-after-reconciliation.txt"
    rerun_migrated = migrate.apply_plan(rerun_plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)
    assert rerun_migrated == 1
    final_row = db_documents.get_sync(document_id=document_uuid)
    assert final_row.display_name == "renamed-after-reconciliation.txt"


def test_apply_plan_fails_closed_when_stored_name_identity_changes_after_catalog_reconciliation(postgres_db, tmp_path):
    """'Other candidate identity metadata' changing, same bytes: the v2
    sidecar's stored_name field is corrupted (no longer corresponds to the
    physical filename it's paired with) AFTER catalog reconciliation.
    _validate_candidate()'s own identity-correspondence check then makes
    the final revalidation return None entirely — the fixed barrier
    (`final_check is None or final_check != candidate`) treats that
    exactly like any other disagreement: abort, preserve the sidecar
    exactly as found, clean up the attempt-created pending row."""
    import json

    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    content = b"content whose sidecar stored_name identity is corrupted after catalog reconciliation"
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "notes.txt", 991000053)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    bogus_stored_name = f"{uuid.uuid4().hex}.txt"

    def corrupt_stored_name_before_sidecar_write(candidate):
        sidecar_path = sidecar_path_for(physical)
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        data["stored_name"] = bogus_stored_name
        write_sidecar_atomic(sidecar_path, data)

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=corrupt_stored_name_before_sidecar_write,
    )

    assert migrated == 0
    # The corrupted sidecar no longer round-trips through load_sidecar()
    # itself (document_id/stored_name correspondence is validated at parse
    # time — see rag/sidecar.py) — read the raw JSON directly to prove it
    # was left EXACTLY as the corruption left it, never overwritten with
    # v3 content built from the stale pre-corruption candidate.
    raw = json.loads(sidecar_path_for(physical).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2  # never overwritten
    assert raw["stored_name"] == bogus_stored_name  # left exactly as found, never silently "fixed"

    document_uuid = uuid.UUID(uuid_hex)
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is None


def test_apply_plan_fails_closed_when_sidecar_is_replaced_with_malformed_content_before_write(postgres_db, tmp_path):
    """Sidecar replaced with malformed (non-JSON) content in the exact
    window between the final revalidation and write_sidecar_atomic() —
    the deterministic hook proves this window is closed: the malformed
    replacement is left untouched (never overwritten with v3 content built
    from stale data), migration reports nothing migrated, and the
    attempt-created pending row (if any) is cleaned up."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    content = b"content whose sidecar is replaced with malformed data before the write"
    uuid_hex = uuid.uuid4().hex
    physical, doc_id = _write_v2_upload(uploads_dir, uuid_hex, content, "notes.txt", 991000054)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1

    def replace_with_malformed_sidecar(candidate):
        sidecar_path_for(physical).write_text("not valid json{", encoding="utf-8")

    migrated = migrate.apply_plan(
        plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir,
        _test_pre_sidecar_write_hook=replace_with_malformed_sidecar,
    )

    assert migrated == 0
    raw = sidecar_path_for(physical).read_text(encoding="utf-8")
    assert raw == "not valid json{"  # never overwritten with v3 content built from stale data

    document_uuid = uuid.UUID(uuid_hex)
    row = db_documents.get_sync(document_id=document_uuid)
    assert row is None


# ---------------------------------------------------------------------------
# C. Stage 5C corrective pass #3, Blocker 4: display_name consistency in
# migration catalog reconciliation and rebuild's catalog gate — both
# proven against REAL PostgreSQL, never only the offline fake catalog.
# ---------------------------------------------------------------------------

def test_migration_fails_closed_on_existing_catalog_display_name_mismatch(postgres_db, tmp_path):
    """Codex reproduced: catalog display_name = "DIFFERENT-CATALOG-NAME.txt"
    vs. sidecar display_name = "sidecar-name.txt", with migration accepting
    the mismatch. _reconcile_catalog_row() must now also compare
    display_name (previously it checked only owner/stored_name/
    content_sha256) and fail closed: sidecar never rewritten, existing
    catalog row left completely untouched."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    content = b"content whose catalog display_name will disagree with its legacy sidecar"
    physical, doc_id = _write_v2_upload(uploads_dir, "e" * 32, content, "sidecar-name.txt", 991000040)

    plan = migrate.build_plan(uploads_dir)
    assert len(plan.candidates) == 1
    assert plan.candidates[0].display_name == "sidecar-name.txt"

    document_uuid = uuid.UUID("e" * 32)
    owner = db_identity.resolve_or_create_user_by_telegram_id_sync(991000040)
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=owner,
        stored_name=physical.name, display_name="DIFFERENT-CATALOG-NAME.txt", content_sha256=sha256_hex(content),
    )

    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 0, "a display_name-mismatched candidate must never be migrated"
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 2  # sidecar never rewritten as v3
    assert loaded["owner_user_id"] == 991000040  # legacy ownership record intact
    row = db_documents.get_sync(document_id=document_uuid)
    assert row.display_name == "DIFFERENT-CATALOG-NAME.txt"  # catalog row never overwritten
    assert row.owner_user_id == owner


def test_migration_matching_display_name_still_migrates_normally(postgres_db, tmp_path):
    """Counterpart proof: a genuinely matching display_name is unaffected
    by the new check — migration proceeds exactly as before."""
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    content = b"content whose display_name genuinely matches everywhere"
    physical, doc_id = _write_v2_upload(uploads_dir, "f" * 32, content, "matching-name.txt", 991000042)

    plan = migrate.build_plan(uploads_dir)
    migrated = migrate.apply_plan(plan, db_identity.resolve_or_create_user_by_telegram_id_sync, uploads_dir)

    assert migrated == 1
    from rag.sidecar import load_sidecar
    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded["schema_version"] == 3
    assert loaded["display_name"] == "matching-name.txt"


def test_rebuild_plan_skips_upload_with_catalog_display_name_mismatch(postgres_db, tmp_path):
    """Codex reproduced: rebuild's own catalog gate continued to accept a
    catalog row whose display_name disagreed with its sidecar. Proven
    directly against a v3 sidecar + REAL PostgreSQL catalog row (never the
    migration path) so this exercises rebuild's _plan_upload_documents()
    specifically."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    content = b"content for the rebuild display_name mismatch proof"
    doc_uuid = uuid.uuid4()
    physical = uploads_dir / f"{doc_uuid.hex}.txt"
    physical.write_bytes(content)
    owner = db_identity.resolve_or_create_user_by_telegram_id_sync(991000043)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(
            upload_document_id(doc_uuid.hex), "sidecar-name.txt", physical.name,
            sha256_hex(content), owner_user_uuid=str(owner),
        ),
    )
    db_documents.create_pending_sync(
        document_id=doc_uuid, owner_user_id=owner,
        stored_name=physical.name, display_name="DIFFERENT-CATALOG-NAME.txt", content_sha256=sha256_hex(content),
    )
    db_documents.mark_active_sync(document_id=doc_uuid)

    rebuild_plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(rebuild_plan.upload_documents) == 0
    assert "catalog_display_name_mismatch" in rebuild_plan.skipped_upload_reasons


def test_rebuild_plan_accepts_upload_with_matching_display_name(postgres_db, tmp_path):
    """Counterpart proof: a genuinely matching display_name plans
    normally — the new check does not reject ordinary, consistent
    uploads."""
    import db.documents as db_documents
    import db.identity as db_identity

    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    content = b"content for the rebuild matching display_name proof"
    doc_uuid = uuid.uuid4()
    physical = uploads_dir / f"{doc_uuid.hex}.txt"
    physical.write_bytes(content)
    owner = db_identity.resolve_or_create_user_by_telegram_id_sync(991000044)
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(
            upload_document_id(doc_uuid.hex), "consistent-name.txt", physical.name,
            sha256_hex(content), owner_user_uuid=str(owner),
        ),
    )
    db_documents.create_pending_sync(
        document_id=doc_uuid, owner_user_id=owner,
        stored_name=physical.name, display_name="consistent-name.txt", content_sha256=sha256_hex(content),
    )
    db_documents.mark_active_sync(document_id=doc_uuid)

    rebuild_plan = rebuild.build_plan(documents_dir, uploads_dir, reference_filenames=None)

    assert len(rebuild_plan.upload_documents) == 1
    assert not rebuild_plan.skipped_upload_reasons
