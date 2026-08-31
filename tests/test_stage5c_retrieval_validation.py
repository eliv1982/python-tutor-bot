"""
Stage 5C corrective pass regression tests:
rag.query._validated_similarity_search()'s fail-closed cross-store
validation. Qdrant's own `scope`/`owner_user_uuid` payload is never
sufficient proof of private ownership on its own (the accepted Stage 5C
contract: Qdrant must never become an independent ownership authority) — a
private result is returned only if the canonical PostgreSQL `documents`
catalog independently agrees it is an ACTIVE document owned by the
requester.

Against a REAL disposable PostgreSQL container (see tests/conftest.py's
postgres_container()/postgres_db()) plus a real local-persistent Qdrant
with deterministic fake embeddings (tests/rag_fakes.py) — proving genuine
batched-query cross-store behavior, never a mocked stand-in. No real
OpenAI/Qdrant network calls anywhere in this module.
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
        collection_name="stage5c_retrieval_validation_test",
    )
    monkeypatch.setattr(rag_query, "get_vector_index", lambda: index)
    yield index
    index.close()


@pytest.fixture
def owner(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(881000001)


@pytest.fixture
def reference_corpus(tmp_path, monkeypatch):
    """
    Stage 5C corrective pass #3 (Blocker 1): writes real, disposable files
    named exactly like every entry in rag.constants.BUILTIN_REFERENCE_FILES
    into a fresh directory and redirects rag.constants.DOCUMENTS_DIR to it
    for the duration of one test — gives
    rag.loader.DocumentLoader.expected_reference_point_hashes() (and
    therefore _validated_similarity_search()/get_knowledge_base_stats(), both
    of which read rag.constants.DOCUMENTS_DIR fresh at call time) a genuine,
    test-controlled trusted reference corpus to validate against, WITHOUT
    ever touching the real data/documents/. Never a hand-picked
    document_id literal that merely happens to look canonical — every
    "genuine reference" test below derives its expected identity/content
    from these exact files via the same production pipeline
    (_genuine_reference_chunk() below).
    """
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
    """
    Returns (document_id, chunk_index, content) for a GENUINE canonical
    reference chunk under `directory` (see the `reference_corpus` fixture),
    derived via the exact same production pipeline
    expected_reference_point_hashes() itself uses — never a hand-typed
    literal that merely happens to share an id. Every "forged" test below
    reuses this real (document_id, chunk_index) identity but substitutes
    different content, or reuses the real content but different identity,
    to prove _is_proven_reference() requires BOTH to match, never either
    alone.
    """
    import rag.constants as rag_constants

    filename = rag_constants.BUILTIN_REFERENCE_FILES[filename_index]
    file_path = Path(directory) / filename
    resolved_root = Path(directory).resolve()
    relative = file_path.resolve().relative_to(resolved_root).as_posix()
    document_id = reference_document_id(relative)
    chunk = document_loader.load_document(file_path)[0]
    return document_id, chunk.metadata["chunk_index"], chunk.page_content


def _insert_point(vi, *, document_id, chunk_index, text, owner_uuid=None, scope=None, source="notes.txt") -> None:
    """Lower-level than _insert_private_point() below: lets a test control
    every payload field independently (including omitting scope/owner
    entirely, or setting scope/document_id/chunk_index to values that
    disagree with each other) — used by the forged-reference tests, which
    need precise control over exactly which fields a forged point copies."""
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


def _create_active_catalog_row(document_uuid: uuid.UUID, owner_uuid: uuid.UUID) -> None:
    db_documents.create_pending_sync(
        document_id=document_uuid, owner_user_id=owner_uuid,
        stored_name=f"{document_uuid.hex}.txt", display_name="notes.txt", content_sha256="a" * 64,
    )
    db_documents.mark_active_sync(document_id=document_uuid)


# ---------------------------------------------------------------------------
# Matching stores: the happy path
# ---------------------------------------------------------------------------

def test_matching_stores_returns_the_private_result(postgres_db, vi, owner):
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    text = "Alpha: decorators explained in depth for the catalog-backed owner."
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(owner), text=text)

    results = rag_query._validated_similarity_search(text, str(owner), 5)

    assert any(d.metadata.get("document_id") == document_id for d, _ in results)


# ---------------------------------------------------------------------------
# Missing DB row
# ---------------------------------------------------------------------------

def test_missing_db_row_drops_the_private_result(postgres_db, vi, owner):
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    # Deliberately never create a catalog row for this document at all —
    # e.g. a legacy sidecar that was never migrated, or a stale Qdrant
    # point left over from a rolled-back ingest.
    text = "Bravo: content whose catalog row is entirely missing."
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(owner), text=text)

    results = rag_query._validated_similarity_search(text, str(owner), 5)

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)


# ---------------------------------------------------------------------------
# Non-active (stale 'pending') catalog row
# ---------------------------------------------------------------------------

def test_pending_catalog_row_drops_the_private_result(postgres_db, vi, owner):
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    db_documents.create_pending_sync(
        document_id=doc_uuid, owner_user_id=owner,
        stored_name=f"{doc_uuid.hex}.txt", display_name="notes.txt", content_sha256="b" * 64,
    )
    # Deliberately never mark_active_sync() — the row stays 'pending',
    # meaning indexing was never confirmed to complete; it must never be
    # treated as a valid, fully ingested private document for retrieval.
    text = "Charlie: content whose catalog row is still stuck at pending."
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(owner), text=text)

    results = rag_query._validated_similarity_search(text, str(owner), 5)

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)


# ---------------------------------------------------------------------------
# Qdrant owner disagrees with the catalog's owner
# ---------------------------------------------------------------------------

def test_qdrant_owner_disagreeing_with_catalog_owner_drops_the_result(postgres_db, vi, owner):
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    # The Qdrant payload claims a DIFFERENT owner than the catalog row
    # records (a stale/corrupt point, or a payload written before an
    # out-of-band ownership correction). The requester below IS the
    # catalog's real owner, but Qdrant's own payload owner disagrees —
    # this must fail closed exactly like the reverse mismatch, since
    # neither store may act as an independent authority.
    other = uuid.uuid4()
    text = "Delta: content whose Qdrant payload owner disagrees with the catalog."
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(other), text=text)

    results = rag_query._validated_similarity_search(text, str(owner), 5)

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)


# ---------------------------------------------------------------------------
# Reference (shared) results never require a catalog lookup
# ---------------------------------------------------------------------------

def test_reference_scope_result_is_returned_with_no_catalog_row_at_all(postgres_db, vi, owner, reference_corpus):
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_REFERENCE, source="ref.md")

    results = rag_query._validated_similarity_search(content, str(owner), 5)

    assert any(d.metadata.get("source") == "ref.md" for d, _ in results)


def test_valid_canonical_reference_is_visible_to_a_different_user_too(postgres_db, vi, owner, reference_corpus):
    """A genuinely canonical reference document remains visible to every
    user, not just the one who happened to query it first — the shared,
    no-owner side of the classification rule."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_REFERENCE, source="ref.md")
    other_user = db_identity.resolve_or_create_user_by_telegram_id_sync(881000099)

    results = rag_query._validated_similarity_search(content, str(other_user), 5)

    assert any(d.metadata.get("source") == "ref.md" for d, _ in results)


# ---------------------------------------------------------------------------
# Stage 5C corrective pass #2, Section 1: the critical reference-scope
# relabelling bypass and its neighbors
# ---------------------------------------------------------------------------

def test_private_point_relabelled_as_reference_scope_is_not_visible_to_another_user(postgres_db, vi, owner):
    """The exact bypass the second audit reproduced: a private upload
    genuinely owned by one user, but whose Qdrant payload has been
    relabelled `scope="reference"` (a stale/corrupt/adversarial point —
    the label alone, regardless of how it got there), must NOT become
    visible to a different requesting user merely because of that label.
    Classification must be driven by document_id, never by `scope`."""
    real_owner = owner
    attacker = db_identity.resolve_or_create_user_by_telegram_id_sync(881000002)

    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, real_owner)
    text = "Echo: private notes relabelled scope=reference to try to leak across users."
    pid = point_id(document_id, 0)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(text),
            payload={
                "text": text,
                "document_id": document_id,
                "chunk_index": 0,
                "content_sha256": sha256_hex(text.encode("utf-8")),
                "source": "private.txt",
                "scope": SCOPE_REFERENCE,  # relabelled — the actual attack
                "owner_user_uuid": str(real_owner),
            },
        )],
    )

    results = rag_query._validated_similarity_search(text, str(attacker), 5)

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)


# ---------------------------------------------------------------------------
# Stage 5C corrective pass #3, Blocker 1: forged CANONICAL document_id —
# the gap the second corrective pass left open (document_id itself is
# mutable Qdrant payload metadata, not a trust anchor).
# ---------------------------------------------------------------------------

def test_private_point_copying_canonical_document_id_is_not_shared(postgres_db, vi, owner, reference_corpus):
    """The exact bypass the THIRD audit reproduced: an arbitrary private
    point copies the document_id of a real canonical reference document
    (but carries its own, genuinely private content and chunk_index 0,
    the same as the real chunk) — this alone must not be sufficient to
    prove reference provenance, since the actual content differs from
    the real canonical chunk at that exact point identity."""
    real_owner = owner
    attacker = db_identity.resolve_or_create_user_by_telegram_id_sync(881000004)

    genuine_document_id, genuine_chunk_index, _genuine_content = _genuine_reference_chunk(reference_corpus)
    private_text = "November-forged: genuinely private content pretending to be the canonical reference chunk."
    _insert_point(
        vi, document_id=genuine_document_id, chunk_index=genuine_chunk_index, text=private_text,
        owner_uuid=str(real_owner),
    )

    results = rag_query._validated_similarity_search(private_text, str(attacker), 5)

    assert all(d.page_content != private_text for d, _ in results)


def test_private_point_copying_canonical_document_id_and_reference_metadata_is_not_shared(postgres_db, vi, owner, reference_corpus):
    """Same forgery as above, but the point ALSO copies every other
    reference-looking payload field (scope="reference", the genuine
    chunk_index, a plausible source name) — none of scope/document_id/
    chunk_index/source is ever sufficient; only the actual content hash
    against the trusted manifest decides."""
    attacker = db_identity.resolve_or_create_user_by_telegram_id_sync(881000005)

    genuine_document_id, genuine_chunk_index, _genuine_content = _genuine_reference_chunk(reference_corpus)
    private_text = "Oscar-forged: private content with every reference-looking metadata field copied too."
    _insert_point(
        vi, document_id=genuine_document_id, chunk_index=genuine_chunk_index, text=private_text,
        scope=SCOPE_REFERENCE, source="ref.md",
    )

    results = rag_query._validated_similarity_search(private_text, str(attacker), 5)

    assert all(d.page_content != private_text for d, _ in results)


def test_private_point_copying_canonical_point_identity_with_different_content_is_rejected(postgres_db, vi, owner, reference_corpus):
    """Framed around the point id specifically (Blocker 1, item 5): a
    point whose (document_id, chunk_index) pair — the two payload fields
    that TOGETHER determine the expected Qdrant point id via
    rag.identity.point_id(), the same deterministic derivation real
    indexing uses — matches a genuine canonical reference point exactly,
    but whose actual returned content differs, must still be rejected.
    Copying the identity that WOULD PRODUCE the correct point id is not
    sufficient without the content also matching."""
    attacker = db_identity.resolve_or_create_user_by_telegram_id_sync(881000006)

    genuine_document_id, genuine_chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    from rag.identity import point_id as make_point_id
    expected_point_id = make_point_id(genuine_document_id, genuine_chunk_index)

    forged_text = genuine_content + " -- TAMPERED: this byte sequence never appeared in the real reference chunk."
    _insert_point(vi, document_id=genuine_document_id, chunk_index=genuine_chunk_index, text=forged_text, scope=SCOPE_REFERENCE)
    # Confirm the point really was written at the SAME id a genuine chunk
    # would use — proving this test isn't accidentally exercising some
    # other, unrelated point.
    assert point_id(genuine_document_id, genuine_chunk_index) == expected_point_id

    results = rag_query._validated_similarity_search(forged_text, str(attacker), 5)

    assert all(d.page_content != forged_text for d, _ in results)


def test_private_point_with_missing_scope_is_not_treated_as_reference(postgres_db, vi, owner):
    """A point with NO `scope` field at all (e.g. a pre-Stage-3A leftover,
    or any other way the field could be absent) must never default to
    shared/reference — it must be evaluated as a private candidate against
    the catalog like any other non-canonical document_id."""
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    # Deliberately no catalog row either — this candidate must fail closed
    # for the same reason test_missing_db_row_drops_the_private_result
    # does, proving "missing scope" doesn't grant an easier path to
    # visibility than "correctly labelled private with no catalog row".
    text = "Foxtrot: a point with no scope field recorded at all."
    pid = point_id(document_id, 0)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(text),
            payload={
                "text": text,
                "document_id": document_id,
                "chunk_index": 0,
                "content_sha256": sha256_hex(text.encode("utf-8")),
                "source": "private.txt",
                "owner_user_uuid": str(owner),
                # no "scope" key at all
            },
        )],
    )

    results = rag_query._validated_similarity_search(text, str(owner), 5)

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)


def test_unknown_noncanonical_reference_looking_document_id_does_not_become_shared(postgres_db, vi, owner):
    """A document_id that merely LOOKS reference-like (arbitrary string,
    not one of the ids independently recomputed from
    BUILTIN_REFERENCE_FILES) must never be treated as shared, even with
    `scope="reference"` — only exact canonical membership counts."""
    document_id = "ref:not-a-real-manifest-entry"
    text = "Golf: a document_id that merely looks like a reference id."
    pid = point_id(document_id, 0)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(text),
            payload={
                "text": text,
                "document_id": document_id,
                "chunk_index": 0,
                "content_sha256": sha256_hex(text.encode("utf-8")),
                "source": "fake-ref.md",
                "scope": SCOPE_REFERENCE,
            },
        )],
    )

    results = rag_query._validated_similarity_search(text, str(owner), 5)

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)


def test_db_failure_fails_closed_for_a_non_reference_candidate(postgres_db, vi, owner, monkeypatch):
    """A PostgreSQL outage while validating a non-reference candidate must
    never be silently treated as "no disagreement, so it's fine" — the
    candidate is dropped (fail-closed), never returned as unvalidated
    private content.

    Stage 5C corrective pass #3 (Blocker 1, requirement 10): this no
    longer raises past _validated_similarity_search() — a PostgreSQL
    outage must not take down retrieval ENTIRELY (which would also hide
    any independently-proven reference results in the same batch); it
    degrades to "this specific non-reference candidate cannot be proven
    owned" and is dropped, exactly as if the catalog had explicitly
    disagreed."""
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    text = "Hotel: content whose catalog check will hit a database outage."
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(owner), text=text)

    def failing_get_active_owners(document_ids):
        raise RuntimeError("simulated PostgreSQL outage")

    monkeypatch.setattr(db_documents, "get_active_owners_sync", failing_get_active_owners)

    results = rag_query._validated_similarity_search(text, str(owner), 5)

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)


def test_db_outage_does_not_affect_a_proven_reference_result_in_the_same_batch(postgres_db, vi, owner, monkeypatch, reference_corpus):
    """Stage 5C corrective pass #3 (Blocker 1, requirement 10) — the
    headline invariant: a PostgreSQL outage must never weaken reference
    availability. A genuinely proven canonical reference chunk and an
    unrelated non-reference candidate are both present; when the catalog
    is unreachable, the proven reference chunk is STILL returned (it
    never depended on PostgreSQL at all) while the non-reference
    candidate fails closed."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_REFERENCE, source="ref.md")

    doc_uuid = uuid.uuid4()
    private_document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    private_text = "Kilo-real: a genuine private upload that would normally validate fine."
    _insert_private_point(vi, document_id=private_document_id, owner_uuid=str(owner), text=private_text)

    def failing_get_active_owners(document_ids):
        raise RuntimeError("simulated PostgreSQL outage")

    monkeypatch.setattr(db_documents, "get_active_owners_sync", failing_get_active_owners)

    reference_results = rag_query._validated_similarity_search(content, str(owner), 5)
    assert any(d.metadata.get("source") == "ref.md" for d, _ in reference_results), (
        "a genuinely proven reference chunk must remain available during a PostgreSQL outage"
    )

    private_results = rag_query._validated_similarity_search(private_text, str(owner), 5)
    assert all(d.metadata.get("document_id") != private_document_id for d, _ in private_results), (
        "a non-reference candidate must still fail closed during the same outage"
    )


@pytest.mark.asyncio
async def test_filtered_private_text_never_reaches_llm_prompt_construction(postgres_db, vi, owner, monkeypatch, reference_corpus):
    """End-to-end proof at the query_knowledge_base() level: a private
    result that fails provenance/catalog validation must never appear in
    the context string handed to the LLM — "dropped from the result list"
    must actually mean "never reaches prompt construction", not just "the
    test checked the wrong layer". Uses the Blocker 1 attack vector
    directly: forging a genuine canonical document_id onto private
    content, never merely relabelling `scope`."""
    import rag.query as rag_query_module

    attacker = db_identity.resolve_or_create_user_by_telegram_id_sync(881000003)

    genuine_document_id, genuine_chunk_index, _genuine_content = _genuine_reference_chunk(reference_corpus)
    secret_text = "TopSecretMarker: this private content must never reach the LLM prompt for another user."
    _insert_point(
        vi, document_id=genuine_document_id, chunk_index=genuine_chunk_index, text=secret_text,
        scope=SCOPE_REFERENCE, source="secret.txt",
    )

    captured_prompts = []

    async def fake_generate_text_response(messages):
        captured_prompts.append(messages)
        return "fallback answer with no private context"

    monkeypatch.setattr(rag_query_module.text_llm, "generate_text_response", fake_generate_text_response)

    await rag_query_module.query_knowledge_base(secret_text, str(attacker))

    assert captured_prompts, "generate_text_response was never called"
    for messages in captured_prompts:
        # Scoped to the SYSTEM message specifically — the one
        # _prepare_context()/_generate_rag_response() build from
        # retrieved results (query_knowledge_base()'s fallback path
        # necessarily also echoes the attacker's own literal query text
        # back as a "user"-role message, which is not itself a leak: the
        # attacker already typed it. The property under test is that
        # FILTERED CONTENT never gets interpolated into the context the
        # system prompt carries).
        system_messages = [m for m in messages if m.get("role") == "system"]
        assert system_messages
        for message in system_messages:
            assert "TopSecretMarker" not in message.get("content", "")


@pytest.mark.asyncio
async def test_genuine_canonical_reference_reaches_prompt_construction(postgres_db, vi, owner, monkeypatch, reference_corpus):
    """Positive-control counterpart (Blocker 1, item 8): a GENUINELY
    proven canonical reference chunk's content must actually reach the
    system prompt's context — the tightened provenance model must not
    accidentally make every reference chunk unusable."""
    import rag.query as rag_query_module

    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_REFERENCE, source="ref.md")

    captured_prompts = []

    async def fake_generate_text_response(messages):
        captured_prompts.append(messages)
        return "grounded answer"

    monkeypatch.setattr(rag_query_module.text_llm, "generate_text_response", fake_generate_text_response)

    response = await rag_query_module.query_knowledge_base(content, str(owner))

    assert "grounded answer" in response
    assert captured_prompts, "generate_text_response was never called"
    system_messages = [m for m in captured_prompts[0] if m.get("role") == "system"]
    assert any(content in m.get("content", "") for m in system_messages), (
        "genuine canonical reference content must reach the system prompt's context"
    )


# ---------------------------------------------------------------------------
# Stage 5C corrective pass #2, Section 2: statistics isolation
#
# rag.query.get_knowledge_base_stats() must use the SAME fail-closed
# catalog-validated visibility model as retrieval — Qdrant's own
# scope/owner_user_uuid payload alone must never be trusted to define
# accessible private ownership for a user-visible count either.
# ---------------------------------------------------------------------------

def test_relabelled_private_point_does_not_affect_another_users_count(postgres_db, vi, owner):
    """A point genuinely owned (per the catalog) by one user, but whose
    RAW Qdrant owner_user_uuid payload has been forged to claim a
    different user, must not inflate that other user's visible count."""
    real_owner = owner
    attacker = db_identity.resolve_or_create_user_by_telegram_id_sync(882000001)

    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, real_owner)
    # Qdrant payload forged to claim ownership by `attacker`, disagreeing
    # with the catalog's real owner.
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(attacker), text="India: forged owner claim")

    stats = rag_query.get_knowledge_base_stats(str(attacker))

    assert stats["status"] == "ok"
    assert stats["total_documents"] == 0


def test_stale_or_missing_catalog_private_point_does_not_count(postgres_db, vi, owner):
    """A raw private candidate with no catalog row at all (missing), and
    one whose row is still 'pending' (never confirmed indexed), must both
    be excluded from the count."""
    missing_doc_uuid = uuid.uuid4()
    missing_document_id = f"upload:{missing_doc_uuid.hex}"
    _insert_private_point(vi, document_id=missing_document_id, owner_uuid=str(owner), text="Juliet: missing catalog row")

    pending_doc_uuid = uuid.uuid4()
    pending_document_id = f"upload:{pending_doc_uuid.hex}"
    db_documents.create_pending_sync(
        document_id=pending_doc_uuid, owner_user_id=owner,
        stored_name=f"{pending_doc_uuid.hex}.txt", display_name="notes.txt", content_sha256="c" * 64,
    )
    # Deliberately never mark_active_sync().
    _insert_private_point(vi, document_id=pending_document_id, owner_uuid=str(owner), text="Kilo: still pending")

    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert stats["total_documents"] == 0


def test_matching_private_point_counts_for_owner(postgres_db, vi, owner):
    """A genuinely active, correctly owned private chunk counts toward the
    requester's own total."""
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(owner), text="Lima: a genuinely owned document")

    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert stats["total_documents"] == 1


def test_canonical_reference_counts_for_every_user(postgres_db, vi, owner, reference_corpus):
    """The shared reference corpus counts for any user, independent of
    that user's own private documents (or lack thereof). Both genuine
    canonical chunks (one per BUILTIN_REFERENCE_FILES entry) are inserted
    with their REAL, manifest-verifiable identity and content — a forged/
    relabelled chunk must never contribute to this count (see the
    forged-reference stats tests below)."""
    import rag.constants as rag_constants

    for i in range(len(rag_constants.BUILTIN_REFERENCE_FILES)):
        document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus, filename_index=i)
        _insert_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_REFERENCE, source=f"ref{i}.md")
    another_user = db_identity.resolve_or_create_user_by_telegram_id_sync(882000002)

    stats_owner = rag_query.get_knowledge_base_stats(str(owner))
    stats_other = rag_query.get_knowledge_base_stats(str(another_user))

    expected_count = len(rag_constants.BUILTIN_REFERENCE_FILES)
    assert stats_owner["total_documents"] == expected_count
    assert stats_other["total_documents"] == expected_count


# ---------------------------------------------------------------------------
# Stage 5C corrective pass #3, Blocker 2: statistics must use the SAME
# reference provenance model as retrieval — a forged/relabelled point must
# never count as reference, and must never double-count for its real owner.
# ---------------------------------------------------------------------------

def test_forged_canonical_document_id_does_not_count_as_reference(postgres_db, vi, owner, reference_corpus):
    """A private point copying a genuine canonical document_id/chunk_index
    (and scope="reference") but carrying different content must not
    inflate the reference count for ANY user."""
    genuine_document_id, genuine_chunk_index, _genuine_content = _genuine_reference_chunk(reference_corpus)
    forged_text = "Papa-forged: private content masquerading as the canonical reference chunk for stats purposes."
    _insert_point(vi, document_id=genuine_document_id, chunk_index=genuine_chunk_index, text=forged_text, scope=SCOPE_REFERENCE)

    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert stats["total_documents"] == 0


def test_forged_canonical_point_identity_with_wrong_content_does_not_count(postgres_db, vi, owner, reference_corpus):
    """Same forgery, framed around point identity specifically: the
    (document_id, chunk_index) pair matches a genuine canonical point
    exactly (so its Qdrant point id is the SAME id real indexing would
    use), but the actual stored content differs — still must not count."""
    genuine_document_id, genuine_chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    tampered_text = genuine_content + " -- TAMPERED for the stats forgery proof."
    _insert_point(vi, document_id=genuine_document_id, chunk_index=genuine_chunk_index, text=tampered_text, scope=SCOPE_REFERENCE)

    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert stats["total_documents"] == 0


def test_relabelled_scope_reference_point_does_not_count_as_reference_or_private(postgres_db, vi, owner, reference_corpus):
    """Stage 5C corrective pass #9 (the release blocker that pass closed):
    a NON-reserved point relabelled scope="reference" must not be counted
    as reference for anyone (it occupies no reserved canonical point id, so
    it can never pass canonical-reference proof) — and, since corrective
    pass #9, must ALSO no longer be counted via the private path just
    because its `owner_user_uuid` happens to equal the requester and the
    PostgreSQL catalog independently agrees the underlying document is
    ACTIVE and owned by them. `scope` must be exactly "private" for a
    non-reserved point to ever be an eligible private candidate at all
    (rag.identity.is_eligible_private_candidate()) — PostgreSQL agreeing
    about the underlying document is not sufficient on its own when the
    derived Qdrant point's own visibility metadata disagrees. This
    supersedes this test's prior (pre-pass-#9) expectation that such a
    point counted once via the private path."""
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    text = "Quebec: private, genuinely owned, but relabelled scope=reference."
    _insert_point(vi, document_id=document_id, chunk_index=0, text=text, scope=SCOPE_REFERENCE, owner_uuid=str(owner))

    results = rag_query._validated_similarity_search(text, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.metadata.get("document_id") != document_id for d, _ in results)
    assert stats["total_documents"] == 0


# ---------------------------------------------------------------------------
# Stage 5C corrective pass #4, Blocker 1: ACTUAL Qdrant point-ID binding.
# The third pass proved (document_id, chunk_index) must map to an expected
# point id AND the content hash must match — but never verified the point
# was actually STORED at that id. A point whose payload claims a genuine
# canonical (document_id, chunk_index)/content but is physically stored
# under a DIFFERENT actual Qdrant point id must never be accepted.
# ---------------------------------------------------------------------------

def test_forged_reference_at_wrong_actual_point_id_is_not_accepted(postgres_db, vi, owner, reference_corpus):
    """The exact Blocker 1 reproduction: genuine canonical document_id/
    chunk_index metadata AND byte-identical genuine canonical content, but
    stored at an arbitrary actual point id that does NOT equal
    rag.identity.point_id(document_id, chunk_index). Must not be accepted
    as reference — and, since its document_id doesn't match this
    application's own upload-identity shape either, must not be returned
    at all."""
    genuine_document_id, genuine_chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    wrong_pid = str(uuid.uuid4())
    assert wrong_pid != point_id(genuine_document_id, genuine_chunk_index)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=wrong_pid,
            vector=DeterministicFakeEmbeddings()._vector_for(genuine_content),
            payload={
                "text": genuine_content,
                "document_id": genuine_document_id,
                "chunk_index": genuine_chunk_index,
                "content_sha256": sha256_hex(genuine_content.encode("utf-8")),
                "source": "ref.md",
                "scope": SCOPE_REFERENCE,
            },
        )],
    )

    results = rag_query._validated_similarity_search(genuine_content, str(owner), 5)

    assert all(d.page_content != genuine_content for d, _ in results)


def test_genuine_reference_at_correct_actual_point_id_carries_the_point_id_metadata(postgres_db, vi, owner, reference_corpus):
    """Positive control: a genuine reference chunk, stored at its correct
    actual point id, is accepted — and similarity_search_with_score()
    actually threads the real Qdrant point id through as
    `_qdrant_point_id` metadata (the mechanism the forgery test above
    depends on existing at all)."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_REFERENCE, source="ref.md")

    raw_results = vi.similarity_search_with_score(content, requesting_user_uuid=str(owner), k=5)
    matching = [d for d, _ in raw_results if d.page_content == content]
    assert matching, "the genuine chunk must be returned at all"
    assert matching[0].metadata.get("_qdrant_point_id") == point_id(document_id, chunk_index)

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    assert any(d.page_content == content for d, _ in results)


def test_private_upload_with_reference_identical_text_remains_private_not_reference(postgres_db, vi, owner, reference_corpus):
    """A genuine private upload whose content happens to be byte-identical
    to a real canonical reference chunk must remain private (visible only
    to its real owner) — its own actual point id (derived from its OWN
    upload document_id/chunk_index) differs from the reference's expected
    point id, so it must never be misclassified as reference regardless of
    content equality."""
    genuine_document_id, genuine_chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    _insert_private_point(vi, document_id=document_id, owner_uuid=str(owner), text=genuine_content)

    other = db_identity.resolve_or_create_user_by_telegram_id_sync(883000001)

    own_results = rag_query._validated_similarity_search(genuine_content, str(owner), 5)
    assert any(d.metadata.get("document_id") == document_id for d, _ in own_results)

    other_results = rag_query._validated_similarity_search(genuine_content, str(other), 5)
    assert all(d.metadata.get("document_id") != document_id for d, _ in other_results)


@pytest.mark.asyncio
async def test_forged_point_id_reference_content_never_reaches_prompt_construction(postgres_db, vi, owner, monkeypatch, reference_corpus):
    """End-to-end proof at the query_knowledge_base() level for the
    Blocker 1 attack vector specifically: forged content stored at a
    wrong actual point id, with genuine canonical metadata/content
    otherwise, must never reach the LLM prompt."""
    import rag.query as rag_query_module

    genuine_document_id, genuine_chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    secret_marker_text = genuine_content + " WRONGPOINTIDMARKER"
    wrong_pid = str(uuid.uuid4())
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=wrong_pid,
            vector=DeterministicFakeEmbeddings()._vector_for(secret_marker_text),
            payload={
                "text": secret_marker_text,
                "document_id": genuine_document_id,
                "chunk_index": genuine_chunk_index,
                "content_sha256": sha256_hex(secret_marker_text.encode("utf-8")),
                "source": "ref.md",
                "scope": SCOPE_REFERENCE,
            },
        )],
    )

    captured_prompts = []

    async def fake_generate_text_response(messages):
        captured_prompts.append(messages)
        return "fallback answer with no forged content"

    monkeypatch.setattr(rag_query_module.text_llm, "generate_text_response", fake_generate_text_response)

    await rag_query_module.query_knowledge_base(secret_marker_text, str(owner))

    assert captured_prompts
    for messages in captured_prompts:
        system_messages = [m for m in messages if m.get("role") == "system"]
        for message in system_messages:
            assert "WRONGPOINTIDMARKER" not in message.get("content", "")


# ---------------------------------------------------------------------------
# Stage 5C corrective pass #4, Blocker 2: mutual-exclusivity of stats
# classification against the actual-point-id-bound reference model above.
# ---------------------------------------------------------------------------

def test_forged_point_id_reference_does_not_count_as_reference_or_private(postgres_db, vi, owner, reference_corpus):
    """The Blocker 1 forgery, viewed through get_knowledge_base_stats():
    genuine canonical document_id/chunk_index/content at the WRONG actual
    point id must not count as reference (fails point-id-bound proof), and
    — since its document_id isn't this application's own upload-identity
    shape — must not count as private either. Total contribution: zero."""
    genuine_document_id, genuine_chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    wrong_pid = str(uuid.uuid4())
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=wrong_pid,
            vector=DeterministicFakeEmbeddings()._vector_for(genuine_content),
            payload={
                "text": genuine_content,
                "document_id": genuine_document_id,
                "chunk_index": genuine_chunk_index,
                "content_sha256": sha256_hex(genuine_content.encode("utf-8")),
                "source": "ref.md",
                "scope": SCOPE_REFERENCE,
            },
        )],
    )

    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert stats["total_documents"] == 0


def test_point_at_expected_reference_id_with_forged_private_owner_never_double_counts(postgres_db, vi, owner, reference_corpus):
    """Structural mutual-exclusivity proof: a point sitting AT a genuine
    canonical reference point id, but with corrupted/forged content (so it
    fails the reference proof) AND a forged owner_user_uuid claiming the
    requester, must not be counted as private either — the exclusion is by
    ACTUAL POINT ID membership in the expected-reference-id set, never by
    whether the point happens to currently verify as reference."""
    genuine_document_id, genuine_chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    expected_pid = point_id(genuine_document_id, genuine_chunk_index)
    tampered_text = genuine_content + " -- CORRUPTED"
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=expected_pid,
            vector=DeterministicFakeEmbeddings()._vector_for(tampered_text),
            payload={
                "text": tampered_text,
                "document_id": f"upload:{uuid.uuid4().hex}",  # forged, upload-shaped
                "chunk_index": 0,
                "content_sha256": sha256_hex(tampered_text.encode("utf-8")),
                "source": "corrupted.txt",
                "scope": SCOPE_PRIVATE,
                "owner_user_uuid": str(owner),
            },
        )],
    )

    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert stats["total_documents"] == 0, (
        "a point sitting at a canonical reference point id must never count as private, "
        "even with a forged owner claim and an upload-shaped document_id"
    )


def test_user_a_and_b_private_counts_remain_isolated(postgres_db, vi, owner):
    """Each user's own genuinely-owned private chunk(s) count only for
    them, never leaking into or being replaced by the other's count."""
    user_a = owner
    user_b = db_identity.resolve_or_create_user_by_telegram_id_sync(882000003)

    doc_a = uuid.uuid4()
    document_id_a = f"upload:{doc_a.hex}"
    _create_active_catalog_row(doc_a, user_a)
    _insert_private_point(vi, document_id=document_id_a, owner_uuid=str(user_a), text="Mike: user A's own document")

    doc_b = uuid.uuid4()
    document_id_b = f"upload:{doc_b.hex}"
    _create_active_catalog_row(doc_b, user_b)
    _insert_private_point(vi, document_id=document_id_b, owner_uuid=str(user_b), text="November: user B's own document", chunk_index=0)
    _insert_private_point(vi, document_id=document_id_b, owner_uuid=str(user_b), text="Oscar: user B's second chunk", chunk_index=1)

    stats_a = rag_query.get_knowledge_base_stats(str(user_a))
    stats_b = rag_query.get_knowledge_base_stats(str(user_b))

    assert stats_a["total_documents"] == 1
    assert stats_b["total_documents"] == 2


# ---------------------------------------------------------------------------
# Stage 5C corrective pass #5, Blocker 4: retrieval and statistics must use
# the SAME canonical-reference classification predicate
# (rag.identity.is_canonical_reference_point()) — a candidate qualifies as
# reference in BOTH surfaces or NEITHER, never one without the other. An
# independent audit reproduced a point with the correct actual Qdrant point
# id and byte-identical canonical text, but INCOMPLETE reference metadata
# (missing document_id/chunk_index payload fields): retrieval correctly
# excluded it (it needs those fields to reconstruct/cross-check the expected
# point id) while the PREVIOUS statistics implementation counted it anyway
# (it looked the point up directly by id and never inspected those fields at
# all). Parameterized so every case is asserted against BOTH surfaces from
# one shared candidate construction, never two independently-written checks
# that could quietly drift apart again.
# ---------------------------------------------------------------------------

def _parity_case_genuine_complete(document_id, chunk_index, content, wrong_pid):
    pid = point_id(document_id, chunk_index)
    return pid, content, {"document_id": document_id, "chunk_index": chunk_index, "source": "ref.md", "scope": SCOPE_REFERENCE}, True


def _parity_case_missing_document_id(document_id, chunk_index, content, wrong_pid):
    pid = point_id(document_id, chunk_index)
    return pid, content, {"chunk_index": chunk_index, "source": "ref.md", "scope": SCOPE_REFERENCE}, False


def _parity_case_missing_chunk_index(document_id, chunk_index, content, wrong_pid):
    pid = point_id(document_id, chunk_index)
    return pid, content, {"document_id": document_id, "source": "ref.md", "scope": SCOPE_REFERENCE}, False


def _parity_case_inconsistent_chunk_index(document_id, chunk_index, content, wrong_pid):
    # Stored at the pid for the REAL chunk_index, but its own payload
    # claims a DIFFERENT chunk_index — the (document_id, chunk_index) pair
    # the point claims no longer maps to the id it's actually stored at.
    pid = point_id(document_id, chunk_index)
    return pid, content, {"document_id": document_id, "chunk_index": chunk_index + 999, "source": "ref.md", "scope": SCOPE_REFERENCE}, False


def _parity_case_changed_text(document_id, chunk_index, content, wrong_pid):
    pid = point_id(document_id, chunk_index)
    tampered = content + " -- TAMPERED for the parity proof"
    return pid, tampered, {"document_id": document_id, "chunk_index": chunk_index, "source": "ref.md", "scope": SCOPE_REFERENCE}, False


def _parity_case_wrong_point_id(document_id, chunk_index, content, wrong_pid):
    return wrong_pid, content, {"document_id": document_id, "chunk_index": chunk_index, "source": "ref.md", "scope": SCOPE_REFERENCE}, False


@pytest.mark.parametrize(
    "build_case",
    [
        _parity_case_genuine_complete,
        _parity_case_missing_document_id,
        _parity_case_missing_chunk_index,
        _parity_case_inconsistent_chunk_index,
        _parity_case_changed_text,
        _parity_case_wrong_point_id,
    ],
    ids=[
        "genuine_complete",
        "missing_document_id",
        "missing_chunk_index",
        "inconsistent_chunk_index",
        "changed_text",
        "wrong_point_id",
    ],
)
def test_retrieval_and_stats_agree_on_canonical_reference_classification(postgres_db, vi, owner, reference_corpus, build_case):
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    wrong_pid = str(uuid.uuid4())
    pid, text, payload_overrides, expect_reference = build_case(document_id, chunk_index, content, wrong_pid)
    payload = {"text": text, **payload_overrides}
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(id=pid, vector=DeterministicFakeEmbeddings()._vector_for(text), payload=payload)],
    )

    retrieval_results = rag_query._validated_similarity_search(text, str(owner), 5)
    retrieval_hit = any(d.page_content == text and d.metadata.get("source") == "ref.md" for d, _ in retrieval_results)

    stats = rag_query.get_knowledge_base_stats(str(owner))
    assert stats["status"] == "ok"
    stats_hit = stats["total_documents"] == 1

    assert retrieval_hit == expect_reference, "retrieval classification diverged from the expected outcome"
    assert stats_hit == expect_reference, "stats classification diverged from the expected outcome"
    # The headline parity invariant: both surfaces must agree with EACH
    # OTHER, never just with the expected outcome independently — this is
    # what actually catches a future re-divergence even if both surfaces
    # happened to be wrong in the same direction.
    assert retrieval_hit == stats_hit, "retrieval and stats disagreed on the same candidate"


def test_incomplete_reference_metadata_at_canonical_point_id_does_not_count_as_private_either(postgres_db, vi, owner, reference_corpus):
    """Mutual-exclusivity extension of the parity proof above: a point
    sitting at a genuine canonical reference point id, with INCOMPLETE
    reference metadata (missing document_id) AND a forged owner_user_uuid
    claiming the requester, must still not count as private — exclusion
    from the private scan is by ACTUAL POINT ID membership in the expected-
    reference-id set, never by whether the point currently verifies as
    reference (see private_chunk_counts_by_document()'s exclude_point_ids)."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    pid = point_id(document_id, chunk_index)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(content),
            payload={
                "text": content,
                # document_id deliberately omitted — incomplete reference
                # metadata, same as the parity case above.
                "chunk_index": chunk_index,
                "source": "ref.md",
                "scope": SCOPE_PRIVATE,
                "owner_user_uuid": str(owner),
            },
        )],
    )

    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert stats["total_documents"] == 0
