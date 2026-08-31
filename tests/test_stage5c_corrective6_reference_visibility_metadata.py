"""
Stage 5C corrective pass #6 regression tests (Blocker 1):

Retrieval and statistics must classify canonical shared reference content
via the SAME predicate — rag.identity.is_canonical_reference_point() —
never via a Qdrant-side candidate query that filters on MUTABLE
`scope`/`owner_user_uuid` metadata before that predicate ever gets a
chance to run.

An independent acceptance review reproduced: a genuine canonical point id,
genuine canonical text, and otherwise canonical (document_id, chunk_index)
identity, whose mutable `scope`/`owner_user_uuid` Qdrant payload was
changed to look private/differently-owned. Retrieval's Qdrant-side
candidate query (rag.index.VectorIndex._visibility_filter(), which filters
on scope/owner) excluded the point BEFORE rag.query._is_proven_reference()
ever got a chance to classify it, while statistics
(VectorIndex.count_verified_reference_points(), which already retrieves
its expected canonical point ids directly by id, ignoring scope entirely)
counted the same point as canonical reference. The two user-visible
surfaces silently used two different candidate pools for what was supposed
to be one shared classification pipeline.

The fix (rag.index.VectorIndex.similarity_search_with_score()'s new
`reference_candidate_point_ids` parameter, wired in by
rag.query._validated_similarity_search()): retrieval now ALSO runs a
second, deterministic Qdrant query restricted (via a `HasIdCondition`
filter) to exactly the expected canonical reference point ids — never a
scope/owner condition — so a genuine canonical point remains a retrieval
CANDIDATE regardless of its mutable visibility metadata. Candidate
acquisition still never decides reference status by itself:
_is_proven_reference() independently re-verifies every candidate from
either pool against the trust anchor before ever treating it as reference.

Against a REAL disposable PostgreSQL container (see tests/conftest.py's
postgres_container()/postgres_db()) plus a real local-persistent Qdrant
with deterministic fake embeddings (tests/rag_fakes.py) — never a mocked
stand-in.
"""

import uuid
from pathlib import Path

import pytest
from qdrant_client.http.models import PointStruct

import db.documents as db_documents
import db.identity as db_identity
import rag.query as rag_query
from rag.identity import point_id, reference_document_id, sha256_hex
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
        collection_name="stage5c_corrective6_reference_visibility_test",
    )
    monkeypatch.setattr(rag_query, "get_vector_index", lambda: index)
    yield index
    index.close()


@pytest.fixture
def owner(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(886000001)


@pytest.fixture
def reference_corpus(tmp_path, monkeypatch):
    """Real, disposable files named exactly like every entry in
    rag.constants.BUILTIN_REFERENCE_FILES, redirecting
    rag.constants.DOCUMENTS_DIR for the duration of one test — gives
    rag.loader.DocumentLoader.expected_reference_point_hashes() (and
    therefore the retrieval/stats surfaces under test) a genuine,
    test-controlled trusted corpus, without ever touching the real
    data/documents/."""
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
    """(document_id, chunk_index, content) for a GENUINE canonical
    reference chunk under `directory`, derived via the exact same
    production pipeline expected_reference_point_hashes() itself uses."""
    import rag.constants as rag_constants

    filename = rag_constants.BUILTIN_REFERENCE_FILES[filename_index]
    file_path = Path(directory) / filename
    resolved_root = Path(directory).resolve()
    relative = file_path.resolve().relative_to(resolved_root).as_posix()
    document_id = reference_document_id(relative)
    chunk = document_loader.load_document(file_path)[0]
    return document_id, chunk.metadata["chunk_index"], chunk.page_content


def _upsert_canonical_point(
    vi, *, document_id, chunk_index, text, scope=None, owner_uuid=None, source="ref.md"
) -> str:
    """Inserts a point at the point id/content a GENUINE canonical chunk
    would use, but lets the test independently control scope/owner_user_uuid
    — including values that DISAGREE with genuine reference provenance, or
    omitting them entirely — to prove classification must never depend on
    them. Returns the point id."""
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
# Required tests 1-4: a genuine canonical point survives retrieval AND
# stats regardless of its mutable scope/owner_user_uuid metadata state.
# ---------------------------------------------------------------------------

def test_genuine_canonical_point_with_ordinary_metadata_is_visible(postgres_db, vi, owner, reference_corpus):
    """Case 1: the honest, unmodified happy path — must keep working."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _upsert_canonical_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_REFERENCE)

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert any(d.page_content == content for d, _ in results)
    assert stats["total_documents"] == 1


def test_genuine_canonical_point_with_changed_scope_is_still_visible(postgres_db, vi, owner, reference_corpus):
    """Case 2: scope relabelled to 'private' with no owner recorded at
    all — the base visibility filter's `should` matches neither branch
    (not scope="reference"; not scope="private" AND owner==requester), so
    the OLD candidate query excluded this point before classification ever
    ran. Must now still be visible via the new candidate path."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _upsert_canonical_point(vi, document_id=document_id, chunk_index=chunk_index, text=content, scope=SCOPE_PRIVATE)

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert any(d.page_content == content for d, _ in results), (
        "a genuine canonical point must remain a retrieval candidate even with a changed scope"
    )
    assert stats["total_documents"] == 1


def test_genuine_canonical_point_with_changed_owner_is_still_visible(postgres_db, vi, owner, reference_corpus):
    """Case 3: scope="private" (a prerequisite for owner_user_uuid to mean
    anything at all) with owner_user_uuid claiming a DIFFERENT, unrelated
    user — again excluded by the base filter's `should` for this
    requester specifically."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    other_uuid = str(uuid.uuid4())
    _upsert_canonical_point(
        vi, document_id=document_id, chunk_index=chunk_index, text=content,
        scope=SCOPE_PRIVATE, owner_uuid=other_uuid,
    )

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert any(d.page_content == content for d, _ in results)
    assert stats["total_documents"] == 1


def test_genuine_canonical_point_with_both_visibility_fields_inconsistent(postgres_db, vi, owner, reference_corpus):
    """Case 4: both scope AND owner_user_uuid are simultaneously wrong
    (scope garbage/unrecognized, owner an unrelated user) — the most
    thoroughly corrupted mutable-metadata state — and remains visible to
    an entirely different requester too, not merely the one the forged
    owner field happens to name."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    other_uuid = str(uuid.uuid4())
    _upsert_canonical_point(
        vi, document_id=document_id, chunk_index=chunk_index, text=content,
        scope="corrupted-unrecognized-scope", owner_uuid=other_uuid,
    )
    another_user = db_identity.resolve_or_create_user_by_telegram_id_sync(886000099)

    results = rag_query._validated_similarity_search(content, str(another_user), 5)
    stats = rag_query.get_knowledge_base_stats(str(another_user))

    assert any(d.page_content == content for d, _ in results)
    assert stats["total_documents"] == 1


# ---------------------------------------------------------------------------
# Required tests 5-7: the new, broader candidate pool must never admit a
# genuinely invalid/forged candidate — combining each existing forgery
# vector with corrupted scope/owner metadata, proving the wider candidate
# net doesn't loosen the actual provenance proof.
# ---------------------------------------------------------------------------

def test_changed_text_with_corrupted_scope_is_still_rejected(postgres_db, vi, owner, reference_corpus):
    """Case 5: correct canonical point id, but tampered text — and scope/
    owner metadata ALSO corrupted (so it's a candidate via the new path
    too). Must still be rejected by both surfaces."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    tampered = content + " -- TAMPERED for the corrective pass #6 proof"
    _upsert_canonical_point(
        vi, document_id=document_id, chunk_index=chunk_index, text=tampered,
        scope=SCOPE_PRIVATE, owner_uuid=str(owner),
    )

    results = rag_query._validated_similarity_search(tampered, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.page_content != tampered for d, _ in results)
    assert stats["total_documents"] == 0


def test_correct_text_wrong_actual_point_id_with_corrupted_scope_is_still_rejected(postgres_db, vi, owner, reference_corpus):
    """Case 6: byte-identical canonical text, but stored at an arbitrary
    actual point id (never a member of the expected-id set at all) with
    corrupted scope/owner too. Must still be rejected."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    wrong_pid = str(uuid.uuid4())
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=wrong_pid,
            vector=DeterministicFakeEmbeddings()._vector_for(content),
            payload={
                "text": content,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "content_sha256": sha256_hex(content.encode("utf-8")),
                "source": "ref.md",
                "scope": SCOPE_PRIVATE,
                "owner_user_uuid": str(owner),
            },
        )],
    )

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.page_content != content for d, _ in results)
    assert stats["total_documents"] == 0


def test_incomplete_canonical_identity_with_corrupted_scope_is_still_rejected(postgres_db, vi, owner, reference_corpus):
    """Case 7: correct actual point id and content, but INCOMPLETE
    reference identity metadata (document_id missing) — plus corrupted
    scope/owner. Must still be rejected on both surfaces."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    pid = point_id(document_id, chunk_index)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(content),
            payload={
                "text": content,
                # document_id deliberately omitted — incomplete identity.
                "chunk_index": chunk_index,
                "source": "ref.md",
                "scope": SCOPE_PRIVATE,
                "owner_user_uuid": str(owner),
            },
        )],
    )

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    assert all(d.page_content != content for d, _ in results)
    assert stats["total_documents"] == 0


# ---------------------------------------------------------------------------
# Required tests 8-9: ordinary private isolation is unaffected by the
# broader reference-candidate pool.
# ---------------------------------------------------------------------------

def test_ordinary_private_point_visible_only_to_its_owner(postgres_db, vi, owner, reference_corpus):
    """Case 8/9: a genuinely private point is visible to its own owner but
    not to another user — proven WHILE a genuine reference corpus (and
    therefore a non-empty reference_candidate_point_ids set) is also in
    play, so the merge/dedup logic is actually exercised."""
    document_id, chunk_index, ref_content = _genuine_reference_chunk(reference_corpus)
    _upsert_canonical_point(vi, document_id=document_id, chunk_index=chunk_index, text=ref_content, scope=SCOPE_REFERENCE)

    doc_uuid = uuid.uuid4()
    private_document_id = f"upload:{doc_uuid.hex}"
    _create_active_catalog_row(doc_uuid, owner)
    private_text = "A genuinely private note that must stay isolated to its owner."
    _insert_private_point(vi, document_id=private_document_id, owner_uuid=str(owner), text=private_text)

    other_user = db_identity.resolve_or_create_user_by_telegram_id_sync(886000098)

    owner_results = rag_query._validated_similarity_search(private_text, str(owner), 5)
    other_results = rag_query._validated_similarity_search(private_text, str(other_user), 5)

    assert any(d.metadata.get("document_id") == private_document_id for d, _ in owner_results)
    assert all(d.metadata.get("document_id") != private_document_id for d, _ in other_results)


# ---------------------------------------------------------------------------
# Required test 10: a canonical point must not ALSO classify/count as
# private, even when its forged owner_user_uuid names the requester.
# ---------------------------------------------------------------------------

def test_canonical_point_with_forged_requester_owner_does_not_double_count(postgres_db, vi, owner, reference_corpus):
    """The genuine canonical point's forged owner_user_uuid names the
    REQUESTER itself — the double-count risk the new merge logic must
    avoid: this point is now a raw candidate through BOTH the base
    private-scoped query (scope="private", owner==requester) AND the new
    has_id-restricted reference-candidate query. It must appear at most
    once in retrieval results, classified as reference, and stats must
    count it exactly once (never twice)."""
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _upsert_canonical_point(
        vi, document_id=document_id, chunk_index=chunk_index, text=content,
        scope=SCOPE_PRIVATE, owner_uuid=str(owner),
    )

    results = rag_query._validated_similarity_search(content, str(owner), 5)
    stats = rag_query.get_knowledge_base_stats(str(owner))

    matches = [d for d, _ in results if d.page_content == content]
    assert len(matches) == 1, "a canonical point must appear exactly once, never duplicated across candidate pools"
    assert stats["total_documents"] == 1, "must count once via the reference path, never twice (reference + private)"


def test_point_at_canonical_id_with_forged_private_metadata_and_wrong_content_is_never_returned(postgres_db, vi, owner, reference_corpus):
    """Retrieval-layer counterpart of the existing stats mutual-exclusivity
    proof (test_point_at_expected_reference_id_with_forged_private_owner_never_double_counts
    in tests/test_stage5c_retrieval_validation.py): a point sitting AT a
    genuine canonical point id, but with CORRUPTED content (fails the
    reference proof) and a forged upload-shaped document_id/owner claiming
    the requester, must not be returned via retrieval either — not as
    reference (content hash fails) and not as private (its real document_id
    payload here is the forged upload: id, which itself never matches a
    real catalog row)."""
    document_id, chunk_index, genuine_content = _genuine_reference_chunk(reference_corpus)
    pid = point_id(document_id, chunk_index)
    tampered_text = genuine_content + " -- CORRUPTED"
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[PointStruct(
            id=pid,
            vector=DeterministicFakeEmbeddings()._vector_for(tampered_text),
            payload={
                "text": tampered_text,
                "document_id": f"upload:{uuid.uuid4().hex}",
                "chunk_index": 0,
                "content_sha256": sha256_hex(tampered_text.encode("utf-8")),
                "source": "corrupted.txt",
                "scope": SCOPE_PRIVATE,
                "owner_user_uuid": str(owner),
            },
        )],
    )

    results = rag_query._validated_similarity_search(tampered_text, str(owner), 5)

    assert all(d.page_content != tampered_text for d, _ in results)


# ---------------------------------------------------------------------------
# Low-level mechanism proof: similarity_search_with_score()'s new
# reference_candidate_point_ids parameter actually surfaces a candidate the
# base visibility filter alone would exclude, carrying correct point id
# metadata for the caller's own classification to use.
# ---------------------------------------------------------------------------

def test_similarity_search_with_score_admits_candidate_via_reference_candidate_point_ids(postgres_db, vi, owner, reference_corpus):
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    other_uuid = str(uuid.uuid4())
    pid = _upsert_canonical_point(
        vi, document_id=document_id, chunk_index=chunk_index, text=content,
        scope=SCOPE_PRIVATE, owner_uuid=other_uuid,
    )

    # Without the new parameter: excluded by the base visibility filter.
    baseline = vi.similarity_search_with_score(content, requesting_user_uuid=str(owner), k=5)
    assert all(d.page_content != content for d, _ in baseline), (
        "sanity check: the base visibility-filtered query alone must exclude this corrupted-metadata point"
    )

    # With the new parameter: admitted as a raw candidate, carrying the
    # real actual Qdrant point id as metadata.
    trusted = document_loader.expected_reference_point_hashes()
    augmented = vi.similarity_search_with_score(
        content, requesting_user_uuid=str(owner), k=5,
        reference_candidate_point_ids=set(trusted.keys()),
    )
    matching = [d for d, _ in augmented if d.page_content == content]
    assert matching, "the has_id-restricted reference candidate query must surface this point"
    assert matching[0].metadata.get("_qdrant_point_id") == pid


# ---------------------------------------------------------------------------
# Prompt-construction proof: a canonical point admitted only through the
# NEW reference-candidate path (its mutable metadata would otherwise have
# excluded it) actually reaches the LLM prompt context end-to-end.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_canonical_point_admitted_via_reference_candidate_path_reaches_prompt_construction(
    postgres_db, vi, owner, monkeypatch, reference_corpus
):
    import rag.query as rag_query_module

    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    other_uuid = str(uuid.uuid4())
    _upsert_canonical_point(
        vi, document_id=document_id, chunk_index=chunk_index, text=content,
        scope=SCOPE_PRIVATE, owner_uuid=other_uuid, source="ref.md",
    )

    captured_prompts = []

    async def fake_generate_text_response(messages):
        captured_prompts.append(messages)
        return "grounded answer from a corrupted-metadata canonical point"

    monkeypatch.setattr(rag_query_module.text_llm, "generate_text_response", fake_generate_text_response)

    response = await rag_query_module.query_knowledge_base(content, str(owner))

    assert "grounded answer from a corrupted-metadata canonical point" in response
    assert captured_prompts, "generate_text_response was never called"
    system_messages = [m for m in captured_prompts[0] if m.get("role") == "system"]
    assert any(content in m.get("content", "") for m in system_messages), (
        "the canonical point's content must reach the system prompt's context even though its "
        "mutable scope/owner metadata would have excluded it from the old candidate query"
    )
