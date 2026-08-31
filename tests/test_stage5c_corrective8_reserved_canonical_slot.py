"""
Stage 5C corrective pass #8 regression tests (Blocker 1):

The deterministic expected canonical reference point-id set (rag.loader.
DocumentLoader.expected_reference_point_hashes().keys()) is a RESERVED
namespace in the active Qdrant collection. For any actual Qdrant point
whose ACTUAL point id belongs to that expected set:

  A. if it passes the complete canonical-reference predicate
     (rag.identity.is_canonical_reference_point()) it classifies as
     reference;
  B. if it does NOT, it is excluded entirely — it must NEVER fall through
     to private-document validation, no matter what private-shaped
     `document_id`/`owner_user_uuid` metadata it also happens to carry.

An independent acceptance review reproduced exactly the gap this pass
closes: a point occupying an expected canonical point id, that FAILED
canonical-reference proof, but whose payload also carried a REAL ACTIVE
private upload's `document_id` (a genuine PostgreSQL `documents` catalog
row), fell through into rag.query._validated_similarity_search()'s
private/catalog validation path and was returned/counted as that private
document — even though rag.query.get_knowledge_base_stats() already
excluded every expected canonical point id from private counting
(`exclude_point_ids=`), so retrieval and statistics silently disagreed on
the classification of the exact same point (`retrieval_hit=True`,
`stats_total_documents=0`).

The fix: rag.query._is_reserved_reference_slot() plus a strict three-way
partition in _validated_similarity_search() — reference / reserved-and-
invalid (dropped, never a private candidate) / ordinary private — so a
point can never be evaluated against the private/catalog path merely
because it failed canonical-reference proof.

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
from rag.identity import point_id, reference_document_id, sha256_hex, upload_document_id
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
        collection_name="stage5c_corrective8_reserved_slot_test",
    )
    monkeypatch.setattr(rag_query, "get_vector_index", lambda: index)
    yield index
    index.close()


@pytest.fixture
def owner(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(887000001)


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


def _upsert_reserved_slot_point(
    vi, *, canonical_document_id, canonical_chunk_index, text,
    forged_document_id=None, forged_chunk_index=0, owner_uuid=None, source="corrupted.txt",
    omit_document_id=False,
) -> str:
    """Upserts a point at the ACTUAL point id a genuine canonical chunk
    (canonical_document_id, canonical_chunk_index) would occupy — a
    RESERVED canonical-reference slot — with arbitrary `text` and either
    an omitted document_id (incomplete-metadata case) or a forged,
    private-upload-shaped document_id/chunk_index/owner payload. Fails
    canonical-reference proof either way (its own claimed identity never
    reconstructs the actual point id) while, in the forged case, still
    carrying metadata that looks exactly like an ordinary private upload.
    Returns the (reserved-slot) point id."""
    pid = point_id(canonical_document_id, canonical_chunk_index)
    payload = {
        "text": text,
        "content_sha256": sha256_hex(text.encode("utf-8")),
        "source": source,
        "scope": SCOPE_PRIVATE,
    }
    if not omit_document_id:
        payload["document_id"] = forged_document_id
        payload["chunk_index"] = forged_chunk_index
    if owner_uuid is not None:
        payload["owner_user_uuid"] = owner_uuid
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(id=pid, vector=DeterministicFakeEmbeddings()._vector_for(text), payload=payload)],
    )
    return pid


def _insert_private_point(vi, *, document_id: str, owner_uuid: str, text: str, source: str = "notes.txt", chunk_index: int = 0) -> None:
    pid = point_id(document_id, chunk_index)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(text),
            payload={
                "text": text,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "content_sha256": sha256_hex(text.encode("utf-8")),
                "source": source,
                "scope": SCOPE_PRIVATE,
                "owner_user_uuid": owner_uuid,
            },
        )],
    )


# ---------------------------------------------------------------------------
# Required test 1: genuine canonical point in canonical slot -> reference
# yes/yes.
# ---------------------------------------------------------------------------

def test_genuine_canonical_point_is_reference_in_retrieval_and_stats(postgres_db, vi, owner, reference_corpus):
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=point_id(document_id, chunk_index),
            vector=DeterministicFakeEmbeddings()._vector_for(content),
            payload={
                "text": content, "document_id": document_id, "chunk_index": chunk_index,
                "content_sha256": sha256_hex(content.encode("utf-8")), "source": "ref.md",
                "scope": SCOPE_REFERENCE,
            },
        )],
    )

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert any(d.page_content == content for d, _ in results)
    assert stats["total_documents"] == 1


# ---------------------------------------------------------------------------
# Required test 2: canonical slot + changed text -> no/no.
# ---------------------------------------------------------------------------

def test_canonical_slot_with_changed_text_is_rejected(postgres_db, vi, owner, reference_corpus):
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    tampered = content + " -- TAMPERED for corrective pass #8"
    _upsert_reserved_slot_point(
        vi, canonical_document_id=document_id, canonical_chunk_index=chunk_index, text=tampered,
        forged_document_id=document_id, forged_chunk_index=chunk_index, source="ref.md",
    )

    results = rag_query._validated_similarity_search(tampered, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.page_content != tampered for d, _ in results)
    assert stats["total_documents"] == 0


# ---------------------------------------------------------------------------
# Required test 3: canonical slot + incomplete canonical metadata -> no/no.
# ---------------------------------------------------------------------------

def test_canonical_slot_with_incomplete_metadata_is_rejected(postgres_db, vi, owner, reference_corpus):
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _upsert_reserved_slot_point(
        vi, canonical_document_id=document_id, canonical_chunk_index=chunk_index, text=content,
        omit_document_id=True, source="ref.md",
    )

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.page_content != content for d, _ in results)
    assert stats["total_documents"] == 0


# ---------------------------------------------------------------------------
# Required tests 4-5: THE independently reproduced release blocker. Canonical
# slot + private metadata pointing to a real ACTIVE private catalog document
# owned by the REQUESTING user must not fall back to private retrieval, and
# must not count as private in stats.
# ---------------------------------------------------------------------------

def test_canonical_slot_with_forged_metadata_naming_a_real_active_document_never_falls_back_to_private(
    postgres_db, vi, owner, reference_corpus
):
    document_id, chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    tampered = genuine_content + " -- CORRUPTED, but claims to be the requester's own private upload"

    forged_uuid = uuid.uuid4()
    forged_document_id = upload_document_id(forged_uuid.hex)
    _create_active_catalog_row(forged_uuid, owner)  # a genuine, ACTIVE private document, really owned by `owner`

    _upsert_reserved_slot_point(
        vi, canonical_document_id=document_id, canonical_chunk_index=chunk_index, text=tampered,
        forged_document_id=forged_document_id, forged_chunk_index=0, owner_uuid=str(owner),
    )

    results = rag_query._validated_similarity_search(tampered, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.page_content != tampered for d, _ in results), (
        "a canonical-slot point that fails reference proof must never be returned as a private "
        "document, even when its forged document_id names a genuine active catalog row the "
        "requester really owns"
    )
    assert all(d.metadata.get("document_id") != forged_document_id for d, _ in results)
    assert stats["total_documents"] == 0, (
        "must not count via the private path just because the catalog independently confirms "
        "the forged document_id as an active document owned by the requester"
    )


# ---------------------------------------------------------------------------
# Required test 6: same, but the corrupted point's owner_user_uuid payload
# names a DIFFERENT user than the real catalog owner of the forged
# document_id — proving classification depends only on the RESERVED-SLOT
# rule, never on Qdrant's own (irrelevant, mutable) owner_user_uuid claim.
# ---------------------------------------------------------------------------

def test_canonical_slot_with_forged_owner_naming_another_user_is_still_dropped(
    postgres_db, vi, owner, reference_corpus
):
    document_id, chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    tampered = genuine_content + " -- CORRUPTED, forged owner names someone else entirely"

    another_user = db_identity.resolve_or_create_user_by_telegram_id_sync(887000099)
    forged_uuid = uuid.uuid4()
    forged_document_id = upload_document_id(forged_uuid.hex)
    # The REAL catalog owner is the requester (`owner`) ...
    _create_active_catalog_row(forged_uuid, owner)

    # ... but the corrupted point's own owner_user_uuid payload lies and
    # names a completely different user.
    _upsert_reserved_slot_point(
        vi, canonical_document_id=document_id, canonical_chunk_index=chunk_index, text=tampered,
        forged_document_id=forged_document_id, forged_chunk_index=0, owner_uuid=str(another_user),
    )

    owner_results = rag_query._validated_similarity_search(tampered, str(owner), 5)
    owner_stats = rag_query.get_knowledge_base_stats(str(owner))
    other_results = rag_query._validated_similarity_search(tampered, str(another_user), 5)
    other_stats = rag_query.get_knowledge_base_stats(str(another_user))

    assert all(d.page_content != tampered for d, _ in owner_results)
    assert owner_stats["total_documents"] == 0
    assert all(d.page_content != tampered for d, _ in other_results)
    assert other_stats["total_documents"] == 0


# ---------------------------------------------------------------------------
# Required tests 7-8: an ordinary private point at a NON-canonical point id
# is unaffected — its owner can retrieve/count it, another user cannot.
# ---------------------------------------------------------------------------

def test_ordinary_private_point_at_noncanonical_id_is_visible_to_its_owner(postgres_db, vi, owner):
    doc_uuid = uuid.uuid4()
    private_document_id = upload_document_id(doc_uuid.hex)
    _create_active_catalog_row(doc_uuid, owner)
    text = "A genuinely private note stored at an ordinary, non-canonical point id."
    _insert_private_point(vi, document_id=private_document_id, owner_uuid=str(owner), text=text)

    results = rag_query._validated_similarity_search(text, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert any(d.metadata.get("document_id") == private_document_id for d, _ in results)
    assert stats["total_documents"] == 1


def test_ordinary_private_point_at_noncanonical_id_is_not_visible_to_another_user(postgres_db, vi, owner):
    other_user = db_identity.resolve_or_create_user_by_telegram_id_sync(887000098)
    doc_uuid = uuid.uuid4()
    private_document_id = upload_document_id(doc_uuid.hex)
    _create_active_catalog_row(doc_uuid, owner)
    text = "A genuinely private note that must stay isolated to its real owner."
    _insert_private_point(vi, document_id=private_document_id, owner_uuid=str(owner), text=text)

    results = rag_query._validated_similarity_search(text, str(other_user), 5)
    stats = rag_query.get_knowledge_base_stats(str(other_user))

    assert all(d.metadata.get("document_id") != private_document_id for d, _ in results)
    assert stats["total_documents"] == 0


# ---------------------------------------------------------------------------
# Required test 9: a point sitting at a reserved canonical slot can never
# be classified twice (reference AND private simultaneously, or double-
# counted). Structural proof at the classification-helper level, plus an
# end-to-end proof that the blocker-4/5 point above never appears at all.
# ---------------------------------------------------------------------------

def test_is_proven_reference_implies_is_reserved_reference_slot():
    """Unit-level structural proof (no DB/Qdrant needed): whenever
    _is_proven_reference() is True for a candidate, _is_reserved_reference_
    slot() must also be True for that SAME candidate — so "reference" and
    "reserved-and-invalid" can never simultaneously misclassify the same
    point, and the three-way partition in _validated_similarity_search() is
    exhaustive and mutually exclusive by construction."""
    class _FakeDoc:
        def __init__(self, metadata, page_content):
            self.metadata = metadata
            self.page_content = page_content

    document_id = "ref:11111111-1111-1111-1111-111111111111"
    chunk_index = 0
    content = "genuine canonical content"
    expected_pid = point_id(document_id, chunk_index)
    trusted = {expected_pid: sha256_hex(content.encode("utf-8"))}

    genuine = _FakeDoc(
        {"_qdrant_point_id": expected_pid, "document_id": document_id, "chunk_index": chunk_index},
        content,
    )
    assert rag_query._is_proven_reference(genuine, trusted) is True
    assert rag_query._is_reserved_reference_slot(genuine, trusted) is True

    tampered = _FakeDoc(
        {"_qdrant_point_id": expected_pid, "document_id": document_id, "chunk_index": chunk_index},
        content + " -- TAMPERED",
    )
    assert rag_query._is_proven_reference(tampered, trusted) is False
    assert rag_query._is_reserved_reference_slot(tampered, trusted) is True, (
        "the reserved-slot check is by ACTUAL point id alone — it must stay True even though "
        "canonical proof now fails"
    )

    elsewhere = _FakeDoc({"_qdrant_point_id": str(uuid.uuid4()), "document_id": document_id, "chunk_index": chunk_index}, content)
    assert rag_query._is_proven_reference(elsewhere, trusted) is False
    assert rag_query._is_reserved_reference_slot(elsewhere, trusted) is False


def test_canonical_slot_forged_private_point_never_appears_in_results_at_all(
    postgres_db, vi, owner, reference_corpus
):
    """End-to-end companion to the structural proof above: the exact
    blocker-4/5 scenario, but asserting the point never appears in the
    results list even once (a stronger guarantee than "not counted twice")
    — its actual point id must be entirely absent."""
    document_id, chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    tampered = genuine_content + " -- CORRUPTED"
    reserved_pid = point_id(document_id, chunk_index)

    forged_uuid = uuid.uuid4()
    forged_document_id = upload_document_id(forged_uuid.hex)
    _create_active_catalog_row(forged_uuid, owner)
    _upsert_reserved_slot_point(
        vi, canonical_document_id=document_id, canonical_chunk_index=chunk_index, text=tampered,
        forged_document_id=forged_document_id, forged_chunk_index=0, owner_uuid=str(owner),
    )

    results = rag_query._validated_similarity_search(tampered, str(owner), 5)

    matches = [d for d, _ in results if d.metadata.get("_qdrant_point_id") == reserved_pid]
    assert matches == [], "a reserved-and-invalid canonical slot must never appear in results, not even once"


# ---------------------------------------------------------------------------
# Required test 10: prompt-construction proof — invalid canonical-slot
# content (from the blocker-4/5 scenario) never reaches the LLM prompt.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_canonical_slot_forged_private_content_never_reaches_prompt_construction(
    postgres_db, vi, owner, monkeypatch, reference_corpus
):
    import rag.query as rag_query_module

    document_id, chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    tampered = genuine_content + " -- CORRUPTED, must never reach the LLM"

    forged_uuid = uuid.uuid4()
    forged_document_id = upload_document_id(forged_uuid.hex)
    _create_active_catalog_row(forged_uuid, owner)
    _upsert_reserved_slot_point(
        vi, canonical_document_id=document_id, canonical_chunk_index=chunk_index, text=tampered,
        forged_document_id=forged_document_id, forged_chunk_index=0, owner_uuid=str(owner),
    )

    captured_prompts = []

    async def fake_generate_text_response(messages):
        captured_prompts.append(messages)
        return "fallback answer — knowledge base had no valid results"

    monkeypatch.setattr(rag_query_module.text_llm, "generate_text_response", fake_generate_text_response)

    await rag_query_module.query_knowledge_base(tampered, str(owner))

    assert captured_prompts, "generate_text_response was never called"
    for messages in captured_prompts:
        # The "user" role message legitimately echoes the caller's own
        # query text (which happens to equal `tampered` here) — that is
        # not the invalid canonical-slot CONTENT reaching the prompt, it
        # is the user's own question. What must never happen is the
        # rejected point's content being injected as retrieved CONTEXT,
        # which would surface in a "system" role message instead.
        system_messages = [m for m in messages if m.get("role") == "system"]
        for message in system_messages:
            assert tampered not in message.get("content", ""), (
                "invalid canonical-slot content must never reach the system prompt's retrieved "
                "context, including via the no-results fallback path"
            )
