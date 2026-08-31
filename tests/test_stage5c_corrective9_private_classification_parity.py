"""
Stage 5C corrective pass #9 regression tests (the single remaining
release blocker that pass closed):

Retrieval and statistics must classify the SAME non-reserved Qdrant point
identically when its private visibility/owner payload metadata is
incomplete or inconsistent — never one surface accepting what the other
rejects.

An independent acceptance review reproduced: a non-reserved Qdrant point
with `document_id` in this application's own upload-identity shape, an
ACTIVE PostgreSQL catalog row genuinely owned by the requester, but a
Qdrant payload claiming `scope="reference"` with `owner_user_uuid` missing
or inconsistent. Retrieval (rag.query._validated_similarity_search())
validated ONLY `document_id` shape plus the PostgreSQL catalog — never
Qdrant's own `scope`/`owner_user_uuid` — and returned it
(`retrieval_hit=True`). Statistics
(rag.index.VectorIndex.private_chunk_counts_by_document()) already
required Qdrant's own `owner_user_uuid` to equal the requester via its
Qdrant-level query filter, so it excluded the point (`stats_total_
documents=0`). The two user-visible surfaces silently disagreed on the
classification of the exact same kind of point.

The fix: rag.identity.is_eligible_private_candidate() — THE single
non-reserved private-candidate eligibility predicate, applied by BOTH
rag.query._validated_similarity_search() (retrieval) and
rag.index.VectorIndex.private_chunk_counts_by_document() (statistics) to
the SAME Qdrant-derived facts (scope, owner_user_uuid, document_id) before
either surface ever consults the PostgreSQL catalog. PostgreSQL remains
the durable ownership authority, but a derived Qdrant point must ALSO be
internally self-consistent about being private content before it is ever
exposed as such.

Against a REAL disposable PostgreSQL container (see tests/conftest.py's
postgres_container()/postgres_db()) plus a real local-persistent Qdrant
with deterministic fake embeddings (tests/rag_fakes.py) — private catalog
ownership is genuinely exercised, never mocked.
"""

import uuid
from pathlib import Path

import pytest
from qdrant_client.http.models import PointStruct

import db.documents as db_documents
import db.identity as db_identity
import rag.query as rag_query
from rag.identity import is_eligible_private_candidate, point_id, reference_document_id, sha256_hex, upload_document_id
from rag.index import SCOPE_PRIVATE, SCOPE_REFERENCE, VectorIndex
from rag.loader import document_loader
from rag_fakes import DeterministicFakeEmbeddings


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's same-named autouse fixture — this module
    exercises the REAL db.documents functions against postgres_db."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Real identity resolution is needed for genuine owner UUIDs with a
    real backing `users` row (documents.owner_user_id's FK target)."""
    yield


@pytest.fixture
def vi(tmp_path, monkeypatch):
    index = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage5c_corrective9_private_parity_test",
    )
    monkeypatch.setattr(rag_query, "get_vector_index", lambda: index)
    yield index
    index.close()


@pytest.fixture
def owner(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(889000001)


@pytest.fixture
def reference_corpus(tmp_path, monkeypatch):
    """Real, disposable files named exactly like every entry in
    rag.constants.BUILTIN_REFERENCE_FILES, redirecting
    rag.constants.DOCUMENTS_DIR for the duration of one test."""
    import rag.constants as rag_constants

    directory = tmp_path / "reference_documents"
    directory.mkdir()
    for i, filename in enumerate(rag_constants.BUILTIN_REFERENCE_FILES):
        (directory / filename).write_text(
            f"Genuine canonical reference content number {i} for {filename}.\n"
            f"This exact text is independently trusted because it is version-controlled.",
            encoding="utf-8",
        )
    monkeypatch.setattr(rag_constants, "DOCUMENTS_DIR", directory)
    return directory


def _genuine_reference_chunk(directory: Path, filename_index: int = 0):
    import rag.constants as rag_constants

    filename = rag_constants.BUILTIN_REFERENCE_FILES[filename_index]
    file_path = Path(directory) / filename
    resolved_root = Path(directory).resolve()
    relative = file_path.resolve().relative_to(resolved_root).as_posix()
    document_id = reference_document_id(relative)
    chunk = document_loader.load_document(file_path)[0]
    return document_id, chunk.metadata["chunk_index"], chunk.page_content


def _create_active_catalog_row(document_uuid: uuid.UUID, owner_uuid: uuid.UUID) -> None:
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=owner_uuid,
        stored_name=f"{document_uuid.hex}.txt", display_name="notes.txt", content_sha256="a" * 64,
    )
    db_documents.mark_active_sync(document_id=document_uuid)


def _upsert_point(vi, *, document_id, chunk_index, text, scope=None, owner_uuid=None, source="notes.txt") -> str:
    """Lets a test control every visibility-relevant payload field
    independently — including omitting scope/owner entirely, or setting
    them to values that disagree with each other — exactly the derived-
    state inconsistency the blocker requires."""
    pid = point_id(document_id, chunk_index)
    payload = {
        "text": text,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content_sha256": sha256_hex(text.encode("utf-8")),
        "source": source,
    }
    if scope is not None:
        payload["scope"] = scope
    if owner_uuid is not None:
        payload["owner_user_uuid"] = owner_uuid
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(id=pid, vector=DeterministicFakeEmbeddings()._vector_for(text), payload=payload)],
    )
    return pid


def _check(vi, owner, document_id, chunk_index, text):
    """Runs both user-visible surfaces for the given requester and returns
    (retrieval_hit, stats_hit) — the parity pair every case below asserts."""
    results = rag_query._validated_similarity_search(text, str(owner), 5)
    retrieval_hit = any(
        d.page_content == text and d.metadata.get("document_id") == document_id for d, _ in results
    )
    stats = rag_query.get_knowledge_base_stats(str(owner))
    assert stats["status"] == "ok"
    stats_hit = stats["total_documents"] == 1
    return retrieval_hit, stats_hit


# ---------------------------------------------------------------------------
# Unit-level structural proof (no DB/Qdrant needed): the shared predicate
# implements exactly the required contract in isolation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scope, owner_field, document_id, expect_uuid",
    [
        pytest.param(SCOPE_PRIVATE, "SELF", "upload:{u}", True, id="eligible"),
        pytest.param(SCOPE_REFERENCE, None, "upload:{u}", False, id="scope_reference_owner_missing"),
        pytest.param(SCOPE_REFERENCE, "SELF", "upload:{u}", False, id="scope_reference_owner_requester"),
        pytest.param(SCOPE_PRIVATE, None, "upload:{u}", False, id="scope_private_owner_missing"),
        pytest.param(SCOPE_PRIVATE, "OTHER", "upload:{u}", False, id="scope_private_owner_other"),
        pytest.param(SCOPE_PRIVATE, "not-a-uuid-at-all", "upload:{u}", False, id="scope_private_owner_malformed"),
        pytest.param(SCOPE_PRIVATE, "SELF", "ref:not-an-upload-id", False, id="scope_private_wrong_document_id_shape"),
        pytest.param("corrupted-unrecognized-scope", "SELF", "upload:{u}", False, id="scope_garbage"),
        pytest.param(None, "SELF", "upload:{u}", False, id="scope_missing"),
    ],
)
def test_is_eligible_private_candidate_matches_required_contract(scope, owner_field, document_id, expect_uuid):
    self_uuid = str(uuid.uuid4())
    other_uuid = str(uuid.uuid4())
    doc_uuid = uuid.uuid4()
    owner_value = {"SELF": self_uuid, "OTHER": other_uuid, None: None}.get(owner_field, owner_field)
    result = is_eligible_private_candidate(
        scope=scope,
        owner_user_uuid=owner_value,
        requesting_user_uuid=self_uuid,
        document_id=document_id.format(u=doc_uuid.hex),
    )
    if expect_uuid:
        assert result == doc_uuid
    else:
        assert result is None


# ---------------------------------------------------------------------------
# Required tests 1-6: parity proof against a REAL Postgres catalog + Qdrant,
# for the exact scope/owner combinations named by the release blocker.
# ---------------------------------------------------------------------------

def _case_eligible_private(owner, other_user):
    return SCOPE_PRIVATE, str(owner)


def _case_scope_reference_owner_missing(owner, other_user):
    # The EXACT independently reproduced blocker.
    return SCOPE_REFERENCE, None


def _case_scope_reference_owner_requester(owner, other_user):
    return SCOPE_REFERENCE, str(owner)


def _case_scope_private_owner_missing(owner, other_user):
    return SCOPE_PRIVATE, None


def _case_scope_private_owner_other(owner, other_user):
    return SCOPE_PRIVATE, str(other_user)


def _case_scope_private_owner_malformed(owner, other_user):
    return SCOPE_PRIVATE, "not-a-canonical-uuid"


@pytest.mark.parametrize(
    "build_case, expect_visible",
    [
        (_case_eligible_private, True),
        (_case_scope_reference_owner_missing, False),
        (_case_scope_reference_owner_requester, False),
        (_case_scope_private_owner_missing, False),
        (_case_scope_private_owner_other, False),
        (_case_scope_private_owner_malformed, False),
    ],
    ids=[
        "1_eligible_private_point",
        "2_scope_reference_owner_missing_THE_BLOCKER",
        "3_scope_reference_owner_requester",
        "4_scope_private_owner_missing",
        "5_scope_private_owner_other",
        "6_scope_private_owner_malformed",
    ],
)
def test_retrieval_and_stats_agree_on_private_classification(postgres_db, vi, owner, build_case, expect_visible):
    other_user = db_identity.resolve_or_create_user_by_telegram_id_sync(889000098)
    scope, owner_field = build_case(owner, other_user)

    doc_uuid = uuid.uuid4()
    document_id = upload_document_id(doc_uuid.hex)
    _create_active_catalog_row(doc_uuid, owner)  # active, genuinely owned by `owner` in PostgreSQL
    text = f"Non-reserved private classification parity case: {scope}/{owner_field}"
    _upsert_point(vi, document_id=document_id, chunk_index=0, text=text, scope=scope, owner_uuid=owner_field)

    retrieval_hit, stats_hit = _check(vi, owner, document_id, 0, text)

    assert retrieval_hit == expect_visible, "retrieval diverged from the expected outcome"
    assert stats_hit == expect_visible, "stats diverged from the expected outcome"
    assert retrieval_hit == stats_hit, "retrieval and stats disagreed on the same point — the exact blocker shape"


# ---------------------------------------------------------------------------
# Required tests 7-8: Qdrant claims the requester, but the PostgreSQL
# catalog itself disagrees (wrong owner / not yet active) — PostgreSQL
# remains authoritative and must still say no.
# ---------------------------------------------------------------------------

def test_qdrant_owner_requester_but_catalog_owner_is_someone_else(postgres_db, vi, owner):
    other_user = db_identity.resolve_or_create_user_by_telegram_id_sync(889000097)
    doc_uuid = uuid.uuid4()
    document_id = upload_document_id(doc_uuid.hex)
    _create_active_catalog_row(doc_uuid, other_user)  # catalog says the OTHER user owns it
    text = "Qdrant claims the requester as owner, but the catalog's real owner is someone else."
    _upsert_point(vi, document_id=document_id, chunk_index=0, text=text, scope=SCOPE_PRIVATE, owner_uuid=str(owner))

    retrieval_hit, stats_hit = _check(vi, owner, document_id, 0, text)

    assert retrieval_hit is False
    assert stats_hit is False


def test_qdrant_owner_requester_but_catalog_row_is_pending(postgres_db, vi, owner):
    doc_uuid = uuid.uuid4()
    document_id = upload_document_id(doc_uuid.hex)
    db_documents.create_pending_sync(
        document_id=doc_uuid, owner_user_id=owner,
        stored_name=f"{doc_uuid.hex}.txt", display_name="notes.txt", content_sha256="d" * 64,
    )
    # Deliberately never mark_active_sync().
    text = "Qdrant claims the requester as owner, but the catalog row never left 'pending'."
    _upsert_point(vi, document_id=document_id, chunk_index=0, text=text, scope=SCOPE_PRIVATE, owner_uuid=str(owner))

    retrieval_hit, stats_hit = _check(vi, owner, document_id, 0, text)

    assert retrieval_hit is False
    assert stats_hit is False


# ---------------------------------------------------------------------------
# Required test 9: a genuine reserved canonical reference remains visible/
# countable regardless of mutable ordinary visibility metadata, provided
# canonical proof passes — the new private-eligibility tightening must
# never affect reserved-slot classification.
# ---------------------------------------------------------------------------

def test_genuine_reserved_reference_remains_visible_despite_private_shaped_metadata(postgres_db, vi, owner, reference_corpus):
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    # Mutable visibility metadata corrupted to look private and owned by
    # someone else entirely — must not matter for a genuine reserved slot.
    other_uuid = str(uuid.uuid4())
    _upsert_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_PRIVATE, owner_uuid=other_uuid, source="ref.md")

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert any(d.page_content == content for d, _ in results)
    assert stats["total_documents"] == 1


# ---------------------------------------------------------------------------
# Required test 10: an invalid reserved canonical slot still cannot fall
# through to private, even under the new tightened private-eligibility
# rule (this predicate must never be reached for a reserved-slot point).
# ---------------------------------------------------------------------------

def test_invalid_reserved_slot_still_cannot_fall_through_to_private(postgres_db, vi, owner, reference_corpus):
    document_id, chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    tampered = genuine_content + " -- CORRUPTED for corrective pass #9"
    reserved_pid = point_id(document_id, chunk_index)

    forged_uuid = uuid.uuid4()
    forged_document_id = upload_document_id(forged_uuid.hex)
    _create_active_catalog_row(forged_uuid, owner)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=reserved_pid,
            vector=DeterministicFakeEmbeddings()._vector_for(tampered),
            payload={
                "text": tampered,
                "document_id": forged_document_id,
                "chunk_index": 0,
                "content_sha256": sha256_hex(tampered.encode("utf-8")),
                "source": "corrupted.txt",
                "scope": SCOPE_PRIVATE,
                "owner_user_uuid": str(owner),
            },
        )],
    )

    results = rag_query._validated_similarity_search(tampered, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.metadata.get("_qdrant_point_id") != reserved_pid for d, _ in results)
    assert stats["total_documents"] == 0


# ---------------------------------------------------------------------------
# Required test 11: an ordinary eligible private point remains invisible to
# another requester (isolation is unaffected by the tightened rule).
# ---------------------------------------------------------------------------

def test_ordinary_private_point_remains_invisible_to_another_requester(postgres_db, vi, owner):
    other_user = db_identity.resolve_or_create_user_by_telegram_id_sync(889000096)
    doc_uuid = uuid.uuid4()
    document_id = upload_document_id(doc_uuid.hex)
    _create_active_catalog_row(doc_uuid, owner)
    text = "A genuinely eligible private note that must stay isolated to its real owner."
    _upsert_point(vi, document_id=document_id, chunk_index=0, text=text, scope=SCOPE_PRIVATE, owner_uuid=str(owner))

    own_retrieval_hit, own_stats_hit = _check(vi, owner, document_id, 0, text)
    other_retrieval_hit, other_stats_hit = _check(vi, other_user, document_id, 0, text)

    assert own_retrieval_hit is True
    assert own_stats_hit is True
    assert other_retrieval_hit is False
    assert other_stats_hit is False


# ---------------------------------------------------------------------------
# Required test 12: prompt-construction proof — inconsistent private
# derived-state content (the exact blocker scenario) never reaches the LLM
# prompt, including via the no-results fallback path.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inconsistent_private_derived_state_never_reaches_prompt_construction(postgres_db, vi, owner, monkeypatch):
    import rag.query as rag_query_module

    doc_uuid = uuid.uuid4()
    document_id = upload_document_id(doc_uuid.hex)
    _create_active_catalog_row(doc_uuid, owner)  # active, genuinely owned by the requester
    secret_text = "BlockerMarker: scope=reference with missing owner must never reach the LLM prompt."
    # The exact reproduced blocker shape: scope="reference", owner_user_uuid
    # missing entirely, non-reserved point id, active catalog row owned by
    # the requester.
    _upsert_point(vi, document_id=document_id, chunk_index=0, text=secret_text, scope=SCOPE_REFERENCE, owner_uuid=None)

    captured_prompts = []

    async def fake_generate_text_response(messages):
        captured_prompts.append(messages)
        return "fallback answer — knowledge base had no valid results"

    monkeypatch.setattr(rag_query_module.text_llm, "generate_text_response", fake_generate_text_response)

    await rag_query_module.query_knowledge_base(secret_text, str(owner))

    assert captured_prompts, "generate_text_response was never called"
    for messages in captured_prompts:
        system_messages = [m for m in messages if m.get("role") == "system"]
        for message in system_messages:
            assert "BlockerMarker" not in message.get("content", ""), (
                "inconsistent private derived-state content must never reach the system prompt's "
                "retrieved context, including via the no-results fallback path"
            )
