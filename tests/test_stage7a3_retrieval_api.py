"""
Stage 7A-3 regression tests: authenticated POST /api/retrieval/search.

Real disposable PostgreSQL (tests/conftest.py's postgres_db) plus a real
local-persistent Qdrant (temp dir, DeterministicFakeEmbeddings — never a
real OpenAI/Qdrant network call) — mirrors tests/test_stage5c_retrieval_
validation.py's own house style for exercising rag.query's fail-closed
cross-store validation genuinely, never mocked. A query string identical
to a chunk's own content always retrieves it as the top hit
(DeterministicFakeEmbeddings is a pure function of the input text).

Never calls real text generation — this endpoint's whole point is raw
retrieval with zero LLM involvement (see app/retrieval.py, rag/query.py's
search_documents()).
"""

import uuid
from pathlib import Path

import pytest
from qdrant_client.http.models import PointStruct
from starlette.testclient import TestClient

import app.documents as app_documents
import app.retrieval as app_retrieval
import db.documents as db_documents
import db.identity as db_identity
import rag.query as rag_query
import web_config
from rag.identity import point_id, reference_document_id, sha256_hex
from rag.index import SCOPE_PRIVATE, SCOPE_REFERENCE, VectorIndex
from rag.loader import document_loader
from rag_fakes import DeterministicFakeEmbeddings
from web.app import create_app
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME, get_current_user_id

SESSION_TOKEN = "stage7a3-retrieval-dummy-token"
INVALID = {"detail": "Invalid request"}
KB_UNAVAILABLE = {"detail": "Knowledge base unavailable"}
RESULT_FIELDS = {"document_id", "source", "chunk_index", "page", "content"}


@pytest.fixture(autouse=True)
def _insecure_cookie_posture(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    yield


@pytest.fixture
def owner_uuid(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(882000001)


@pytest.fixture
def other_owner_uuid(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(882000002)


@pytest.fixture
def real_vector_index(tmp_path, monkeypatch):
    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage7a3_retrieval_api_test",
    )
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    monkeypatch.setattr(rag_query, "get_vector_index", lambda: vi)
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path / "uploads")
    yield vi
    vi.close()


@pytest.fixture
def reference_corpus(tmp_path, monkeypatch):
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


def _insert_reference_point(vi, *, document_id: str, chunk_index: int, text: str, source: str) -> None:
    pid = point_id(document_id, chunk_index)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[
            PointStruct(
                id=pid,
                vector=DeterministicFakeEmbeddings()._vector_for(text),
                payload={
                    "text": text,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "content_sha256": sha256_hex(text.encode("utf-8")),
                    "source": source,
                    "scope": SCOPE_REFERENCE,
                },
            )
        ],
    )


def _insert_private_point(vi, *, document_id: str, owner_uuid: str, text: str, source: str = "notes.txt", chunk_index: int = 0) -> None:
    pid = point_id(document_id, chunk_index)
    vi.client.upsert(
        collection_name=vi.collection_name,
        points=[
            PointStruct(
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
            )
        ],
    )


def _client(user_id: uuid.UUID) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    client = TestClient(app)
    client.cookies.set(web_config.session_cookie_name(), SESSION_TOKEN)
    return client


def _csrf(token: str = SESSION_TOKEN) -> dict:
    return {CSRF_HEADER_NAME: derive_csrf_token(token)}


def _search(user_id, query, top_k=None, headers=None):
    body = {"query": query}
    if top_k is not None:
        body["top_k"] = top_k
    return _client(user_id).post("/api/retrieval/search", json=body, headers=headers if headers is not None else _csrf())


# ============================================================================
# A. Authentication / CSRF.
# ============================================================================


def test_unauthenticated_search_is_401():
    response = TestClient(create_app()).post(
        "/api/retrieval/search", json={"query": "q"}, headers={CSRF_HEADER_NAME: "irrelevant"}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_search_without_csrf_is_403(postgres_db, owner_uuid):
    response = _client(owner_uuid).post("/api/retrieval/search", json={"query": "q"})
    assert response.status_code == 403
    assert response.json() == {"detail": "CSRF validation failed"}


# ============================================================================
# B. top_k validation.
# ============================================================================


def test_default_top_k_is_3(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    captured = {}
    real_search_documents = rag_query.search_documents

    def spy(query, requesting_user_uuid, k):
        captured["k"] = k
        return real_search_documents(query, requesting_user_uuid, k)

    monkeypatch.setattr(app_retrieval, "search_documents", spy)
    response = _search(owner_uuid, "q")
    assert response.status_code == 200
    assert captured["k"] == 3


@pytest.mark.parametrize("top_k", [1, 10])
def test_top_k_boundaries_are_accepted(postgres_db, owner_uuid, real_vector_index, top_k):
    response = _search(owner_uuid, "q", top_k=top_k)
    assert response.status_code == 200


@pytest.mark.parametrize("top_k", [0, -1, 11])
def test_top_k_out_of_range_is_422(postgres_db, owner_uuid, top_k):
    response = _search(owner_uuid, "q", top_k=top_k)
    assert response.status_code == 422
    assert response.json() == INVALID


# ============================================================================
# C. Query validation.
# ============================================================================


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_or_whitespace_query_is_422(postgres_db, owner_uuid, query):
    response = _search(owner_uuid, query)
    assert response.status_code == 422
    assert response.json() == INVALID


# ============================================================================
# D. No results.
# ============================================================================


def test_no_results_is_200_with_empty_list(postgres_db, owner_uuid, real_vector_index):
    response = _search(owner_uuid, "nothing in this empty index matches anything at all")
    assert response.status_code == 200
    assert response.json() == {"results": []}


# ============================================================================
# E. Private hits.
# ============================================================================


@pytest.mark.asyncio
async def test_private_hit_uses_catalog_display_name_and_normalized_uuid(postgres_db, owner_uuid, real_vector_index):
    content = "Alpha: a private document about recursion in Python."
    result = await app_documents.ingest_document(
        file_bytes=content.encode("utf-8"), extension=".txt", display_name="recursion-notes.txt", owner_user_id=owner_uuid,
    )
    assert result.success is True
    doc_uuid = result.stored.document_uuid

    response = _search(owner_uuid, content)
    assert response.status_code == 200
    hits = response.json()["results"]
    hit = next(h for h in hits if h["document_id"] == str(doc_uuid))
    assert set(hit) == RESULT_FIELDS
    assert hit["source"] == "recursion-notes.txt"
    assert hit["source"] != result.stored.physical_path.name  # never the opaque stored_name
    assert hit["content"] == content
    uuid.UUID(hit["document_id"])  # normalized to a plain catalog UUID string


@pytest.mark.asyncio
async def test_foreign_private_hit_is_excluded(postgres_db, owner_uuid, other_owner_uuid, real_vector_index):
    content = "Bravo: someone else's private notes about closures."
    result = await app_documents.ingest_document(
        file_bytes=content.encode("utf-8"), extension=".txt", display_name="closures.txt", owner_user_id=other_owner_uuid,
    )
    assert result.success is True

    response = _search(owner_uuid, content)
    assert response.status_code == 200
    assert response.json() == {"results": []}


@pytest.mark.asyncio
async def test_deleting_private_hit_is_excluded_through_the_catalog_gate(postgres_db, owner_uuid, real_vector_index):
    content = "Charlie: a private document that is about to be deleted."
    result = await app_documents.ingest_document(
        file_bytes=content.encode("utf-8"), extension=".txt", display_name="soon-gone.txt", owner_user_id=owner_uuid,
    )
    assert result.success is True
    db_documents.begin_or_resume_delete_sync(document_id=result.stored.document_uuid, owner_user_id=owner_uuid)

    response = _search(owner_uuid, content)
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_pending_private_hit_is_excluded_through_the_catalog_gate(postgres_db, owner_uuid, real_vector_index):
    doc_uuid = uuid.uuid4()
    document_id = f"upload:{doc_uuid.hex}"
    db_documents.create_pending_sync(
        document_id=doc_uuid, owner_user_id=owner_uuid,
        stored_name=f"{doc_uuid.hex}.txt", display_name="notes.txt", content_sha256="e" * 64,
    )
    text = "Delta: content whose catalog row is still stuck at pending."
    _insert_private_point(real_vector_index, document_id=document_id, owner_uuid=str(owner_uuid), text=text)

    response = _search(owner_uuid, text)
    assert response.status_code == 200
    assert response.json() == {"results": []}


# ============================================================================
# F. Reference hits.
# ============================================================================


def _reference_hit_for(user_id, document_id: str, content: str) -> dict:
    response = _search(user_id, content)
    assert response.status_code == 200
    return next(h for h in response.json()["results"] if h["document_id"] == document_id)


def test_reference_hit_uses_safe_source_and_non_uuid_id(postgres_db, owner_uuid, real_vector_index, reference_corpus):
    import rag.constants as rag_constants

    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_reference_point(real_vector_index, document_id=document_id, chunk_index=chunk_index, text=content, source="ref.md")

    hit = _reference_hit_for(owner_uuid, document_id, content)
    assert set(hit) == RESULT_FIELDS
    assert hit["document_id"] == document_id
    with pytest.raises(ValueError):
        uuid.UUID(hit["document_id"])  # reference ids are not UUIDs
    # The label comes from the trusted version-controlled manifest, NOT from
    # the (mutable) Qdrant payload's own `source` ("ref.md" above).
    assert hit["source"] == rag_constants.BUILTIN_REFERENCE_FILES[0]
    assert hit["source"] != "ref.md"


def test_reference_source_ignores_an_injected_internal_path_in_qdrant_metadata(
    postgres_db, owner_uuid, real_vector_index, reference_corpus
):
    """Finding 3 regression: a canonical (fully proven) reference point whose
    mutable Qdrant `source` payload was replaced with an internal filesystem
    path must never surface that value over HTTP."""
    import rag.constants as rag_constants

    injected = "C:\\internal\\secret-path.md"
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_reference_point(real_vector_index, document_id=document_id, chunk_index=chunk_index, text=content, source=injected)

    response = _search(owner_uuid, content)
    assert response.status_code == 200
    assert injected not in response.text
    assert "secret-path" not in response.text and "C:" not in response.text
    hit = next(h for h in response.json()["results"] if h["document_id"] == document_id)
    assert hit["source"] != injected
    assert hit["source"] == rag_constants.BUILTIN_REFERENCE_FILES[0]  # the trusted canonical/reference-derived label


@pytest.mark.parametrize(
    "mutated_source",
    ["totally-different.md", "../../etc/passwd", "/var/lib/internal/notes.md", "", "python-fundamentals.md.exe", "ref:forged"],
)
def test_reference_source_is_independent_of_whatever_the_qdrant_source_field_says(
    postgres_db, owner_uuid, real_vector_index, reference_corpus, mutated_source
):
    import rag.constants as rag_constants

    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_reference_point(
        real_vector_index, document_id=document_id, chunk_index=chunk_index, text=content, source=mutated_source
    )

    hit = _reference_hit_for(owner_uuid, document_id, content)
    assert hit["source"] == rag_constants.BUILTIN_REFERENCE_FILES[0]
    assert hit["document_id"] == document_id and document_id.startswith("ref:")


def test_reference_source_is_bound_to_the_reference_identity_not_to_the_payload_or_position(
    postgres_db, owner_uuid, real_vector_index, reference_corpus
):
    """Each of the four canonical reference documents maps to ITS OWN
    manifest filename, even when every point's Qdrant `source` claims the
    same (wrong) label."""
    import rag.constants as rag_constants

    contents = {}
    for index, filename in enumerate(rag_constants.BUILTIN_REFERENCE_FILES):
        document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus, index)
        _insert_reference_point(
            real_vector_index, document_id=document_id, chunk_index=chunk_index, text=content, source="one-wrong-label.md"
        )
        contents[filename] = (document_id, content)

    for filename, (document_id, content) in contents.items():
        assert _reference_hit_for(owner_uuid, document_id, content)["source"] == filename


def test_reference_source_falls_back_to_the_proven_reference_id_never_to_payload_metadata():
    """Direct unit: an id not in the trusted manifest yields the id string
    itself — there is no code path from payload metadata to `source`."""
    import rag.constants as rag_constants

    known = reference_document_id(rag_constants.BUILTIN_REFERENCE_FILES[2])
    assert app_retrieval._reference_source_label(known) == rag_constants.BUILTIN_REFERENCE_FILES[2]

    unknown = reference_document_id("not-in-the-manifest.md")
    assert app_retrieval._reference_source_label(unknown) == unknown


def test_reference_hit_never_triggers_generation_and_keeps_a_safe_string_id(
    postgres_db, owner_uuid, real_vector_index, reference_corpus, monkeypatch
):
    from services import text_llm

    def _forbidden(*args, **kwargs):
        raise AssertionError("retrieval must never call text generation")

    monkeypatch.setattr(text_llm, "generate_text_response", _forbidden)
    document_id, chunk_index, content = _genuine_reference_chunk(reference_corpus)
    _insert_reference_point(
        real_vector_index, document_id=document_id, chunk_index=chunk_index, text=content, source="C:\\internal\\secret-path.md"
    )

    hit = _reference_hit_for(owner_uuid, document_id, content)
    assert isinstance(hit["document_id"], str) and hit["document_id"] == document_id


@pytest.mark.asyncio
async def test_private_hit_source_still_comes_from_the_catalog_even_when_qdrant_source_is_mutated(
    postgres_db, owner_uuid, real_vector_index
):
    """Private behavior is unchanged: catalog display_name, plain catalog
    UUID — independent of a mutated Qdrant `source` payload."""
    content = "Foxtrot: a private document whose Qdrant payload source gets tampered with."
    result = await app_documents.ingest_document(
        file_bytes=content.encode("utf-8"), extension=".txt", display_name="real-name.txt", owner_user_id=owner_uuid,
    )
    assert result.success is True
    real_vector_index.client.set_payload(
        collection_name=real_vector_index.collection_name,
        payload={"source": "C:\\internal\\secret-path.md"},
        points=[point_id(result.stored.document_id, 0)],
    )

    response = _search(owner_uuid, content)
    hit = next(h for h in response.json()["results"] if h["document_id"] == str(result.stored.document_uuid))
    assert hit["source"] == "real-name.txt"
    assert "secret-path" not in response.text


# ============================================================================
# G. No internal metadata ever exposed.
# ============================================================================


@pytest.mark.asyncio
async def test_no_score_or_internal_metadata_fields_are_exposed(postgres_db, owner_uuid, real_vector_index):
    content = "Echo: checked for accidental metadata leakage."
    await app_documents.ingest_document(
        file_bytes=content.encode("utf-8"), extension=".txt", display_name="echo.txt", owner_user_id=owner_uuid,
    )
    response = _search(owner_uuid, content)
    hits = response.json()["results"]
    assert hits
    for hit in hits:
        assert set(hit) == RESULT_FIELDS
        assert "score" not in hit and "scope" not in hit and "owner_user_uuid" not in hit
        assert "stored_name" not in hit and "content_sha256" not in hit


# ============================================================================
# H. Availability and generation boundary.
# ============================================================================


def test_qdrant_unavailable_maps_to_503(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    monkeypatch.setattr(
        real_vector_index, "similarity_search_with_score",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("qdrant down")),
    )
    response = _search(owner_uuid, "q")
    assert response.status_code == 503
    assert response.json() == KB_UNAVAILABLE


def test_never_calls_text_generation(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    from services import text_llm

    def _forbidden(*args, **kwargs):
        raise AssertionError("retrieval must never call text generation")

    monkeypatch.setattr(text_llm, "generate_text_response", _forbidden)
    response = _search(owner_uuid, "q")
    assert response.status_code == 200


# ============================================================================
# I. Shares the 64 KiB small-JSON body cap (web.app.JSON_BODY_LIMITED_ROUTES)
# with POST /api/chat and PATCH /api/settings — see
# tests/test_stage7a2_body_limit.py for the shared middleware's own
# byte-boundary proofs; this just confirms this route is actually wired in.
# ============================================================================


def test_oversized_json_body_is_413(postgres_db, owner_uuid):
    import web_config

    over_limit_query = "q" * web_config.MAX_JSON_BODY_BYTES
    response = _client(owner_uuid).post(
        "/api/retrieval/search", json={"query": over_limit_query}, headers=_csrf()
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
