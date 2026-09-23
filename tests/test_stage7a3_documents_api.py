"""
Stage 7A-3 regression tests: authenticated POST/GET/DELETE /api/documents
and GET /api/documents/{id}.

Real disposable PostgreSQL (tests/conftest.py's postgres_db) for genuine
catalog/concurrency behavior, and a real local-persistent Qdrant (temp
dir, DeterministicFakeEmbeddings — never a real OpenAI/Qdrant network
call) for genuine ingest/delete behavior — never mocked stand-ins for
either, mirroring tests/test_stage5c_documents_catalog.py's own house
style. Auth is faked via web.dependencies.get_current_user_id's dependency
override (no real session needed); the REAL require_csrf dependency runs
against a dummy session-cookie value.
"""

import contextlib
import os
import uuid
from pathlib import Path

import httpx
import openai
import pytest
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

import app.documents as app_documents
import db.documents as db_documents
import db.identity as db_identity
import web_config
from config import MAX_DOCUMENT_SIZE_BYTES
from db.engine import get_sync_engine
from db.models import Document
from rag.index import SourceMutatedError, VectorIndex, VectorIndexUnavailableError, is_index_unavailable_error
from rag_fakes import DeterministicFakeEmbeddings
from web.app import DOCUMENT_BODY_LIMITED_ROUTES, DOCUMENT_BODY_MAX_BYTES, create_app
from web.body_limit import RequestBodyLimitMiddleware
from web.csrf import derive_csrf_token
from web.dependencies import CSRF_HEADER_NAME, get_current_user_id

SESSION_TOKEN = "stage7a3-documents-dummy-token"
NOT_FOUND = {"detail": "Document not found"}
INVALID = {"detail": "Invalid request"}
UNSUPPORTED_TYPE = {"detail": "Unsupported file type"}
FILE_TOO_LARGE = {"detail": "File too large"}
BODY_TOO_LARGE = {"detail": "Request body too large"}
PROCESSING_FAILED = {"detail": "Document processing failed"}
DELETION_FAILED = {"detail": "Document deletion failed"}
KB_UNAVAILABLE = {"detail": "Knowledge base unavailable"}


@pytest.fixture(autouse=True)
def _insecure_cookie_posture(monkeypatch):
    monkeypatch.setattr(web_config, "COOKIE_SECURE", False)


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    """Shadows conftest.py's autouse fixture — real `users` rows are
    required for the documents.owner_user_id foreign key."""
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's autouse fixture — this module exercises the
    REAL db.documents functions against postgres_db."""
    yield


@pytest.fixture
def owner_uuid(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(781000001)


@pytest.fixture
def other_owner_uuid(postgres_db):
    return db_identity.resolve_or_create_user_by_telegram_id_sync(781000002)


@pytest.fixture
def real_vector_index(tmp_path, monkeypatch):
    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage7a3_documents_api_test",
    )
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: vi)
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", uploads_dir)
    yield vi
    vi.close()


def _client(user_id: uuid.UUID) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    client = TestClient(app)
    client.cookies.set(web_config.session_cookie_name(), SESSION_TOKEN)
    return client


def _csrf(token: str = SESSION_TOKEN) -> dict:
    return {CSRF_HEADER_NAME: derive_csrf_token(token)}


def _upload(client, filename, content, content_type="text/plain"):
    return client.post("/api/documents", files={"file": (filename, content, content_type)}, headers=_csrf())


def _create_active_document(client, filename="notes.txt", content=b"hello world, a real ingestible document body.") -> uuid.UUID:
    response = _upload(client, filename, content)
    assert response.status_code == 201, response.text
    return uuid.UUID(response.json()["id"])


# ============================================================================
# A. Authentication / CSRF — shared across all five routes.
# ============================================================================


@pytest.mark.parametrize(
    "method, path",
    [("GET", "/api/documents"), ("GET", "/api/documents/{id}"), ("POST", "/api/documents"), ("DELETE", "/api/documents/{id}")],
)
def test_unauthenticated_document_routes_are_401(method, path):
    client = TestClient(create_app())
    url = path.format(id=uuid.uuid4())
    response = client.request(method, url, headers={CSRF_HEADER_NAME: "irrelevant"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_upload_without_csrf_is_403(real_vector_index):
    client = _client(uuid.uuid4())
    response = client.post("/api/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 403
    assert response.json() == {"detail": "CSRF validation failed"}


def test_delete_without_csrf_is_403():
    client = _client(uuid.uuid4())
    response = client.delete(f"/api/documents/{uuid.uuid4()}")
    assert response.status_code == 403
    assert response.json() == {"detail": "CSRF validation failed"}


def test_get_routes_require_no_csrf(postgres_db, owner_uuid):
    """GET is safe/idempotent — no CSRF dependency at all."""
    client = _client(owner_uuid)
    assert client.get("/api/documents").status_code == 200
    assert client.get(f"/api/documents/{uuid.uuid4()}").status_code == 404


# ============================================================================
# B. Upload.
# ============================================================================


def test_valid_upload_succeeds_with_201_and_safe_shape(postgres_db, owner_uuid, real_vector_index):
    response = _upload(_client(owner_uuid), "notes.txt", b"a perfectly ordinary text document.")
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "display_name", "created_at"}
    assert body["display_name"] == "notes.txt"
    uuid.UUID(body["id"])  # does not raise


def test_extension_matching_is_case_insensitive(postgres_db, owner_uuid, real_vector_index):
    response = _upload(_client(owner_uuid), "NOTES.TXT", b"case insensitive extension body.")
    assert response.status_code == 201
    assert response.json()["display_name"] == "NOTES.TXT"


def test_empty_filename_is_422_invalid(postgres_db, owner_uuid, real_vector_index):
    response = _upload(_client(owner_uuid), "", b"content")
    assert response.status_code == 422
    assert response.json() == INVALID


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_path_like_filename_uses_only_the_final_leaf_component(postgres_db, owner_uuid, real_vector_index, separator):
    filename = f"some{separator}nested{separator}folder{separator}real-name.txt"
    response = _upload(_client(owner_uuid), filename, b"path-like filename body.")
    assert response.status_code == 201
    assert response.json()["display_name"] == "real-name.txt"


def test_unicode_filename_is_preserved(postgres_db, owner_uuid, real_vector_index):
    filename = "заметки-по-питону-🐍.txt"
    response = _upload(_client(owner_uuid), filename, b"unicode filename body.")
    assert response.status_code == 201
    assert response.json()["display_name"] == filename


def test_255_char_filename_is_accepted(postgres_db, owner_uuid, real_vector_index):
    filename = ("a" * 251) + ".txt"  # 255 code points total
    assert len(filename) == 255
    response = _upload(_client(owner_uuid), filename, b"boundary-length filename body.")
    assert response.status_code == 201
    assert response.json()["display_name"] == filename


def test_256_char_filename_is_rejected(postgres_db, owner_uuid, real_vector_index):
    filename = ("a" * 252) + ".txt"  # 256 code points total
    assert len(filename) == 256
    response = _upload(_client(owner_uuid), filename, b"over-boundary filename body.")
    assert response.status_code == 422
    assert response.json() == INVALID


def test_unsupported_extension_is_422(postgres_db, owner_uuid, real_vector_index):
    response = _upload(_client(owner_uuid), "malware.exe", b"unsupported extension body.", content_type="application/octet-stream")
    assert response.status_code == 422
    assert response.json() == UNSUPPORTED_TYPE


def test_exactly_10mib_file_is_accepted_and_not_rejected_by_multipart_overhead(postgres_db, owner_uuid, real_vector_index):
    """Also proves the coarse multipart-envelope cap (10 MiB + 16 KiB)
    never rejects a genuinely valid, exactly-at-limit multipart upload —
    only the semantic ingest_document() check applies precisely."""
    content = b"A" * MAX_DOCUMENT_SIZE_BYTES
    response = _upload(_client(owner_uuid), "big.txt", content)
    assert response.status_code == 201


def test_10mib_plus_one_byte_is_rejected_as_file_too_large(postgres_db, owner_uuid, real_vector_index):
    content = b"A" * (MAX_DOCUMENT_SIZE_BYTES + 1)
    response = _upload(_client(owner_uuid), "toobig.txt", content)
    assert response.status_code == 413
    assert response.json() == FILE_TOO_LARGE


def test_ingest_failure_maps_to_500_with_no_orphan(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    monkeypatch.setattr(
        real_vector_index, "reconcile_document",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated indexing failure")),
    )
    response = _upload(_client(owner_uuid), "notes.txt", b"content that will fail to index.")
    assert response.status_code == 500
    assert response.json() == PROCESSING_FAILED

    # No orphan catalog row of any status for this owner.
    from db.engine import get_sync_engine
    from db.models import Document
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []


# ============================================================================
# C. Body-size limits (raw ASGI — precise byte-boundary behavior for the
# document route's own, larger middleware instance).
# ============================================================================


class _RecordingApp:
    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class _Receive:
    def __init__(self, chunks, *, fail_if_called=False):
        self.chunks = list(chunks)
        self.fail_if_called = fail_if_called

    async def __call__(self):
        if self.fail_if_called:
            raise AssertionError("body must not be read")
        if self.chunks:
            chunk = self.chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(self.chunks)}
        return {"type": "http.disconnect"}


def _scope(content_length=None):
    headers = [(b"content-type", b"multipart/form-data; boundary=x")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    return {
        "type": "http", "method": "POST", "path": "/api/documents",
        "headers": headers, "client": ("127.0.0.1", 1), "server": ("testserver", 80),
    }


def _middleware(inner):
    return RequestBodyLimitMiddleware(inner, max_body_bytes=DOCUMENT_BODY_MAX_BYTES, limited_routes=DOCUMENT_BODY_LIMITED_ROUTES)


def _run(middleware, scope, receive):
    import asyncio

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def test_declared_oversized_content_length_is_rejected_before_reading_body():
    inner = _RecordingApp()
    receive = _Receive([b"x"], fail_if_called=True)
    sent = _run(_middleware(inner), _scope(content_length=DOCUMENT_BODY_MAX_BYTES + 1), receive)
    assert not inner.called
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_dishonest_small_content_length_is_not_trusted():
    inner = _RecordingApp()
    oversized = [b"a" * 4096 for _ in range(DOCUMENT_BODY_MAX_BYTES // 4096 + 2)]
    sent = _run(_middleware(inner), _scope(content_length=10), _Receive(oversized))
    assert not inner.called
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_missing_content_length_still_bounded_by_actual_byte_counting():
    inner = _RecordingApp()
    oversized = [b"a" * 4096 for _ in range(DOCUMENT_BODY_MAX_BYTES // 4096 + 2)]
    sent = _run(_middleware(inner), _scope(content_length=None), _Receive(oversized))
    assert not inner.called
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_document_route_body_limit_is_larger_than_json_limit():
    assert DOCUMENT_BODY_MAX_BYTES == MAX_DOCUMENT_SIZE_BYTES + 16 * 1024
    assert DOCUMENT_BODY_LIMITED_ROUTES == frozenset({("POST", "/api/documents")})


# ============================================================================
# D. List.
# ============================================================================


def test_list_default_limit_is_20(postgres_db, owner_uuid, real_vector_index):
    client = _client(owner_uuid)
    for i in range(25):
        _create_active_document(client, filename=f"doc-{i}.txt", content=f"content {i}".encode())
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 20


def test_list_limit_can_be_raised_to_100_but_not_beyond(postgres_db, owner_uuid, real_vector_index):
    client = _client(owner_uuid)
    assert client.get("/api/documents?limit=100").status_code == 200
    over = client.get("/api/documents?limit=101")
    assert over.status_code == 422
    assert over.json() == INVALID


def test_list_negative_offset_is_422(postgres_db, owner_uuid):
    response = _client(owner_uuid).get("/api/documents?offset=-1")
    assert response.status_code == 422
    assert response.json() == INVALID


def test_list_shows_only_active_documents_owned_by_caller(postgres_db, owner_uuid, other_owner_uuid, real_vector_index):
    client = _client(owner_uuid)
    active_id = _create_active_document(client, "active.txt", b"active content body.")

    pending_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=pending_id, owner_user_id=owner_uuid,
        stored_name=f"{pending_id.hex}.txt", display_name="pending.txt", content_sha256="a" * 64,
    )

    deleting_id = _create_active_document(client, "deleting.txt", b"soon to be deleted body.")
    db_documents.begin_or_resume_delete_sync(document_id=deleting_id, owner_user_id=owner_uuid)

    _create_active_document(_client(other_owner_uuid), "foreign.txt", b"someone else's document body.")

    response = client.get("/api/documents")
    ids = {item["id"] for item in response.json()["items"]}
    assert ids == {str(active_id)}


def test_list_pagination_is_deterministic_and_covers_every_row_exactly_once(postgres_db, owner_uuid, real_vector_index):
    client = _client(owner_uuid)
    created = [_create_active_document(client, f"doc-{i}.txt", f"body {i}".encode()) for i in range(5)]

    seen = []
    for offset in (0, 2, 4):
        page = client.get(f"/api/documents?limit=2&offset={offset}").json()["items"]
        seen.extend(item["id"] for item in page)

    assert len(seen) == len(set(seen)) == 5
    assert set(seen) == {str(i) for i in created}


# ============================================================================
# E. Detail.
# ============================================================================


def test_detail_active_own_document_succeeds(postgres_db, owner_uuid, real_vector_index):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    response = client.get(f"/api/documents/{doc_id}")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"id", "display_name", "created_at"}
    assert body["id"] == str(doc_id)


def test_detail_missing_is_404(postgres_db, owner_uuid):
    response = _client(owner_uuid).get(f"/api/documents/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_detail_foreign_document_is_404(postgres_db, owner_uuid, other_owner_uuid, real_vector_index):
    doc_id = _create_active_document(_client(other_owner_uuid))
    response = _client(owner_uuid).get(f"/api/documents/{doc_id}")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_detail_pending_own_document_is_404(postgres_db, owner_uuid):
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="pending.txt", content_sha256="b" * 64,
    )
    response = _client(owner_uuid).get(f"/api/documents/{doc_id}")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_detail_deleting_own_document_is_404(postgres_db, owner_uuid, real_vector_index):
    doc_id = _create_active_document(_client(owner_uuid))
    db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)
    response = _client(owner_uuid).get(f"/api/documents/{doc_id}")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_detail_missing_foreign_pending_deleting_are_identical_public_shape(
    postgres_db, owner_uuid, other_owner_uuid, real_vector_index
):
    missing = _client(owner_uuid).get(f"/api/documents/{uuid.uuid4()}")

    foreign_id = _create_active_document(_client(other_owner_uuid))
    foreign = _client(owner_uuid).get(f"/api/documents/{foreign_id}")

    pending_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=pending_id, owner_user_id=owner_uuid,
        stored_name=f"{pending_id.hex}.txt", display_name="p.txt", content_sha256="c" * 64,
    )
    pending = _client(owner_uuid).get(f"/api/documents/{pending_id}")

    deleting_id = _create_active_document(_client(owner_uuid))
    db_documents.begin_or_resume_delete_sync(document_id=deleting_id, owner_user_id=owner_uuid)
    deleting = _client(owner_uuid).get(f"/api/documents/{deleting_id}")

    for response in (missing, foreign, pending, deleting):
        assert response.status_code == 404
        assert response.json() == NOT_FOUND


# ============================================================================
# F. Delete.
# ============================================================================


def test_delete_success_returns_204_with_no_body(postgres_db, owner_uuid, real_vector_index):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert response.status_code == 204
    assert response.content == b""
    assert db_documents.get_sync(document_id=doc_id) is None


def test_delete_completed_repeat_is_404(postgres_db, owner_uuid, real_vector_index):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    assert client.delete(f"/api/documents/{doc_id}", headers=_csrf()).status_code == 204
    second = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert second.status_code == 404
    assert second.json() == NOT_FOUND


def test_delete_foreign_document_is_404_and_untouched(postgres_db, owner_uuid, other_owner_uuid, real_vector_index):
    doc_id = _create_active_document(_client(other_owner_uuid))
    response = _client(owner_uuid).delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    record = db_documents.get_sync(document_id=doc_id)
    assert record is not None and record.status == "active"


def test_delete_pending_document_is_404(postgres_db, owner_uuid):
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="pending.txt", content_sha256="d" * 64,
    )
    response = _client(owner_uuid).delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_delete_qdrant_failure_leaves_deleting_hidden_and_503(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    monkeypatch.setattr(
        real_vector_index, "delete_document", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("qdrant down"))
    )

    response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert response.status_code == 503
    assert response.json() == KB_UNAVAILABLE
    assert db_documents.get_sync(document_id=doc_id).status == "deleting"
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.get("/api/documents").json()["items"] == []


def test_delete_retries_after_qdrant_failure_succeed(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    real_delete_document = real_vector_index.delete_document
    monkeypatch.setattr(
        real_vector_index, "delete_document", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("qdrant down"))
    )
    assert client.delete(f"/api/documents/{doc_id}", headers=_csrf()).status_code == 503

    monkeypatch.setattr(real_vector_index, "delete_document", real_delete_document)
    retry = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert retry.status_code == 204
    assert db_documents.get_sync(document_id=doc_id) is None


def test_delete_file_cleanup_failure_leaves_deleting_hidden_and_500(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    monkeypatch.setattr(app_documents, "cleanup_file", lambda path: False)

    response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert response.status_code == 500
    assert response.json() == DELETION_FAILED
    assert db_documents.get_sync(document_id=doc_id).status == "deleting"
    assert client.get(f"/api/documents/{doc_id}").status_code == 404


def test_delete_retries_after_file_cleanup_failure_succeed(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    real_cleanup_file = app_documents.cleanup_file
    monkeypatch.setattr(app_documents, "cleanup_file", lambda path: False)
    assert client.delete(f"/api/documents/{doc_id}", headers=_csrf()).status_code == 500

    monkeypatch.setattr(app_documents, "cleanup_file", real_cleanup_file)
    retry = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert retry.status_code == 204
    assert db_documents.get_sync(document_id=doc_id) is None


def test_delete_sidecar_cleanup_failure_retries_successfully(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    """cleanup_file() is the SAME primitive for both the physical file and
    the sidecar (see app.documents._perform_delete_cleanup_sync) — this
    exercises the failure landing specifically on the SECOND of the two
    calls (the sidecar), proving the retry still converges."""
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)

    real_cleanup_file = app_documents.cleanup_file
    calls = {"n": 0}

    def flaky_cleanup(path):
        calls["n"] += 1
        if calls["n"] == 2:
            return False
        return real_cleanup_file(path)

    monkeypatch.setattr(app_documents, "cleanup_file", flaky_cleanup)
    assert client.delete(f"/api/documents/{doc_id}", headers=_csrf()).status_code == 500
    assert db_documents.get_sync(document_id=doc_id).status == "deleting"

    monkeypatch.setattr(app_documents, "cleanup_file", real_cleanup_file)
    retry = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert retry.status_code == 204
    assert db_documents.get_sync(document_id=doc_id) is None


def test_delete_final_catalog_delete_failure_retries_successfully(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    real_delete_sync = db_documents.delete_sync
    monkeypatch.setattr(db_documents, "delete_sync", lambda **k: (_ for _ in ()).throw(RuntimeError("db down")))

    response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert response.status_code == 500
    assert response.json() == DELETION_FAILED
    assert db_documents.get_sync(document_id=doc_id).status == "deleting"

    monkeypatch.setattr(db_documents, "delete_sync", real_delete_sync)
    retry = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert retry.status_code == 204
    assert db_documents.get_sync(document_id=doc_id) is None


def test_delete_never_derives_a_path_from_the_client_supplied_display_name(postgres_db, owner_uuid, real_vector_index):
    """The physical/sidecar paths deleted are derived from the catalog's
    internal `stored_name`, never from `display_name` — proven by
    uploading with a path-traversal-shaped display name and confirming the
    upload's own real (opaque, UUID-named) physical file is the one that
    disappears, with no error from any attempted traversal."""
    client = _client(owner_uuid)
    doc_id = _create_active_document(client, filename="../../../etc/passwd.txt", content=b"traversal-shaped display name.")
    record_before = db_documents.get_sync(document_id=doc_id)
    physical_path = app_documents.MANAGED_UPLOADS_DIR / record_before.stored_name
    assert physical_path.exists()

    response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert response.status_code == 204
    assert not physical_path.exists()


@contextlib.contextmanager
def _request_a_finishes_deletion_right_after_first_transaction_ends(document_id: uuid.UUID, stored_name: str):
    """Deterministic stand-in (no threads, no sleeps) for a concurrent
    request A that completes the ENTIRE real deletion — Qdrant points,
    physical file, sidecar, and the final catalog-row removal — at the exact
    instant the request under test's FIRST database transaction has ended
    and before it does anything else. Fires once, on the first Session
    commit/rollback after arming: `after_commit`/`after_rollback` run AFTER
    the real DBAPI commit/rollback, so the request under test holds no row
    lock (A's own DELETE cannot deadlock against it). Reproduces the exact
    audited window: UPDATE -> [row disappears] -> whatever the request does
    next."""
    fired: list = []

    def finish_deletion(_session):
        if fired:
            return
        fired.append(True)
        app_documents._perform_delete_cleanup_sync(document_id, stored_name)

    event.listen(Session, "after_commit", finish_deletion)
    event.listen(Session, "after_rollback", finish_deletion)
    try:
        yield fired
    finally:
        event.remove(Session, "after_commit", finish_deletion)
        event.remove(Session, "after_rollback", finish_deletion)


@pytest.mark.asyncio
async def test_concurrent_delete_converges_when_a_racing_request_finishes_first(
    postgres_db, owner_uuid, real_vector_index
):
    """Finding 1 regression (Section 8 convergence rule: 'if two DELETE
    requests overlap and both establish that the document belonged to the
    caller before final catalog removal, both are allowed to converge to
    204').

    Request A has already flipped the row to 'deleting' (its cleanup is
    mid-flight). Request B — the production delete_document() under test —
    reaches the row while it still exists as own 'deleting'; A then finishes
    the ENTIRE cleanup, including removing the catalog row, immediately
    after B's first transaction ends. With the old UPDATE-then-SELECT shape
    B's UPDATE matched zero rows and its follow-up SELECT found nothing —
    a spurious 404. B's authorization is now one atomic UPDATE ... RETURNING
    made while the row existed, so B converges to success."""
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    stored_name = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)  # request A
    assert stored_name is not None
    physical_path = app_documents.MANAGED_UPLOADS_DIR / stored_name
    assert physical_path.exists()

    with _request_a_finishes_deletion_right_after_first_transaction_ends(doc_id, stored_name) as fired:
        result = await app_documents.delete_document(owner_uuid, doc_id)  # request B

    assert fired  # the window really was exercised
    assert result is True
    assert db_documents.get_sync(document_id=doc_id) is None
    assert not physical_path.exists()


def test_concurrent_delete_converges_to_204_over_http_when_a_racing_request_finishes_first(
    postgres_db, owner_uuid, real_vector_index
):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    stored_name = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)  # request A

    with _request_a_finishes_deletion_right_after_first_transaction_ends(doc_id, stored_name) as fired:
        response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())  # request B

    assert fired
    assert response.status_code == 204
    assert db_documents.get_sync(document_id=doc_id) is None

    # B beginning only AFTER the row is fully gone is the one legitimate 404.
    late = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert late.status_code == 404
    assert late.json() == NOT_FOUND


@pytest.mark.asyncio
async def test_authorization_established_on_an_own_active_row_survives_the_row_vanishing_immediately_after(
    postgres_db, owner_uuid, real_vector_index
):
    """Same window, but request B is the one that flips 'active' ->
    'deleting' itself: the row (completed by a concurrent request) is gone
    right after B's own transition commits, yet B was authorized."""
    doc_id = _create_active_document(_client(owner_uuid))
    stored_name = db_documents.get_sync(document_id=doc_id).stored_name

    with _request_a_finishes_deletion_right_after_first_transaction_ends(doc_id, stored_name) as fired:
        result = await app_documents.delete_document(owner_uuid, doc_id)

    assert fired
    assert result is True


def test_foreign_already_deleting_row_is_404_and_triggers_no_cleanup(
    postgres_db, owner_uuid, other_owner_uuid, real_vector_index, monkeypatch
):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    stored_name = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)
    assert stored_name is not None  # owner's own deletion is in flight (row 'deleting')

    qdrant_delete_calls = []
    monkeypatch.setattr(real_vector_index, "delete_document", lambda *a, **k: qdrant_delete_calls.append(a))
    cleanup_calls = []
    monkeypatch.setattr(app_documents, "cleanup_file", lambda path: cleanup_calls.append(path) or True)

    response = _client(other_owner_uuid).delete(f"/api/documents/{doc_id}", headers=_csrf())

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert qdrant_delete_calls == []
    assert cleanup_calls == []
    assert db_documents.get_sync(document_id=doc_id).status == "deleting"
    assert (app_documents.MANAGED_UPLOADS_DIR / stored_name).exists()


def test_pending_row_delete_is_404_and_triggers_no_cleanup(postgres_db, owner_uuid, real_vector_index, monkeypatch):
    doc_id = uuid.uuid4()
    db_documents.create_pending_sync(
        document_id=doc_id, owner_user_id=owner_uuid,
        stored_name=f"{doc_id.hex}.txt", display_name="pending.txt", content_sha256="f" * 64,
    )
    qdrant_delete_calls = []
    monkeypatch.setattr(real_vector_index, "delete_document", lambda *a, **k: qdrant_delete_calls.append(a))

    response = _client(owner_uuid).delete(f"/api/documents/{doc_id}", headers=_csrf())

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert qdrant_delete_calls == []
    assert db_documents.get_sync(document_id=doc_id).status == "pending"


def test_concurrent_authorized_cleanup_is_idempotent(postgres_db, owner_uuid, real_vector_index):
    """Two authorized requests that BOTH run the full cleanup sequence (the
    convergence case) never error on the second run: every step no-ops on
    already-removed state."""
    doc_id = _create_active_document(_client(owner_uuid))
    stored_name = db_documents.begin_or_resume_delete_sync(document_id=doc_id, owner_user_id=owner_uuid)
    physical_path = app_documents.MANAGED_UPLOADS_DIR / stored_name

    app_documents._perform_delete_cleanup_sync(doc_id, stored_name)
    app_documents._perform_delete_cleanup_sync(doc_id, stored_name)  # second, converging request

    assert not physical_path.exists()
    assert not physical_path.with_suffix(".meta.json").exists()
    assert db_documents.get_sync(document_id=doc_id) is None


# ============================================================================
# G. Finding 2: DELETE never leaves managed storage, whatever the catalog
# `stored_name` says.
# ============================================================================


def _set_catalog_stored_name(document_id: uuid.UUID, stored_name: str) -> None:
    with Session(get_sync_engine()) as session:
        session.execute(update(Document).where(Document.id == document_id).values(stored_name=stored_name))
        session.commit()


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"decoy that must never be deleted")
    return path


# (case id, stored_name factory, decoy-path factory). `deep` is
# <tmp>/managed/deep/uploads, so "../../x" lands in <tmp>/managed (inside the
# pytest tmp dir, never anywhere shared).
def _corrupt_cases():
    return [
        (
            "posix_traversal",
            lambda ctx: "../../outside.pdf",
            lambda ctx: [ctx.root / "outside.pdf", ctx.root / "outside.meta.json"],
        ),
        (
            "windows_traversal",
            lambda ctx: "..\\..\\outside.pdf",
            lambda ctx: [ctx.root / "outside.pdf", ctx.root / "outside.meta.json"],
        ),
        (
            "subdirectory",
            lambda ctx: "subdir/file.pdf",
            lambda ctx: [ctx.deep / "subdir" / "file.pdf", ctx.deep / "subdir" / "file.meta.json"],
        ),
        (
            "absolute_path",
            lambda ctx: str(ctx.root / "absolute-victim.pdf"),
            lambda ctx: [ctx.root / "absolute-victim.pdf", ctx.root / "absolute-victim.meta.json"],
        ),
        (
            "wrong_uuid_stem",
            lambda ctx: f"{ctx.other_hex}.txt",
            lambda ctx: [ctx.deep / f"{ctx.other_hex}.txt", ctx.deep / f"{ctx.other_hex}.meta.json"],
        ),
        (
            "unsupported_extension",
            lambda ctx: f"{ctx.doc_id.hex}.exe",
            lambda ctx: [ctx.deep / f"{ctx.doc_id.hex}.exe"],
        ),
        (
            "empty",
            lambda ctx: "",
            lambda ctx: [],
        ),
    ]


class _DeleteCtx:
    def __init__(self, root: Path, deep: Path, doc_id: uuid.UUID):
        self.root, self.deep, self.doc_id = root, deep, doc_id
        self.other_hex = uuid.uuid4().hex


@pytest.mark.parametrize("case_id, name_factory, decoy_factory", _corrupt_cases(), ids=[c[0] for c in _corrupt_cases()])
def test_delete_with_corrupt_catalog_stored_name_unlinks_nothing_and_stays_deleting_with_500(
    postgres_db, owner_uuid, real_vector_index, tmp_path, monkeypatch, case_id, name_factory, decoy_factory
):
    root = tmp_path / "managed"
    deep = root / "deep" / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", deep)

    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    original_name = db_documents.get_sync(document_id=doc_id).stored_name
    own_physical = deep / original_name
    own_sidecar = own_physical.with_suffix(".meta.json")
    assert own_physical.exists() and own_sidecar.exists()

    ctx = _DeleteCtx(root, deep, doc_id)
    corrupt_name = name_factory(ctx)
    decoys = [_touch(p) for p in decoy_factory(ctx)]
    _set_catalog_stored_name(doc_id, corrupt_name)

    qdrant_delete_calls = []
    real_delete_document = real_vector_index.delete_document
    monkeypatch.setattr(
        real_vector_index, "delete_document",
        lambda *a, **k: qdrant_delete_calls.append(a) or real_delete_document(*a, **k),
    )

    response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())

    assert response.status_code == 500
    assert response.json() == DELETION_FAILED  # fixed detail; never the value or the reason
    if corrupt_name:
        assert corrupt_name not in response.text
    # Nothing was unlinked — outside the managed directory OR inside it —
    # and validation ran before ANY side effect (no Qdrant call either).
    for decoy in decoys:
        assert decoy.exists(), f"{case_id}: decoy {decoy} was deleted"
    assert own_physical.exists() and own_sidecar.exists()
    assert qdrant_delete_calls == []
    # Row remains 'deleting' (hidden), retryable after data correction.
    assert db_documents.get_sync(document_id=doc_id).status == "deleting"
    assert client.get(f"/api/documents/{doc_id}").status_code == 404

    _set_catalog_stored_name(doc_id, original_name)  # administrative correction
    retry = client.delete(f"/api/documents/{doc_id}", headers=_csrf())
    assert retry.status_code == 204
    assert not own_physical.exists() and not own_sidecar.exists()
    assert db_documents.get_sync(document_id=doc_id) is None
    for decoy in decoys:
        assert decoy.exists(), f"{case_id}: decoy {decoy} was deleted by the retry"


def test_delete_with_symlinked_physical_file_escaping_managed_storage_unlinks_nothing(
    postgres_db, owner_uuid, real_vector_index, tmp_path, monkeypatch
):
    deep = tmp_path / "managed" / "uploads"
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", deep)
    client = _client(owner_uuid)
    doc_id = _create_active_document(client)
    original_name = db_documents.get_sync(document_id=doc_id).stored_name
    own_physical = deep / original_name

    outside_target = _touch(tmp_path / "outside-target.txt")
    own_physical.unlink()
    try:
        own_physical.symlink_to(outside_target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/account")

    response = client.delete(f"/api/documents/{doc_id}", headers=_csrf())

    assert response.status_code == 500
    assert response.json() == DELETION_FAILED
    assert outside_target.exists()
    assert os.path.lexists(own_physical)  # the symlink itself was not followed or removed either
    assert db_documents.get_sync(document_id=doc_id).status == "deleting"


def test_delete_with_valid_managed_name_removes_exactly_the_documents_own_file_and_sidecar(
    postgres_db, owner_uuid, real_vector_index
):
    client = _client(owner_uuid)
    doc_id = _create_active_document(client, "own.txt", b"the document that gets deleted.")
    bystander_id = _create_active_document(client, "bystander.txt", b"a different, unrelated document.")
    own = app_documents.MANAGED_UPLOADS_DIR / db_documents.get_sync(document_id=doc_id).stored_name
    bystander = app_documents.MANAGED_UPLOADS_DIR / db_documents.get_sync(document_id=bystander_id).stored_name

    assert client.delete(f"/api/documents/{doc_id}", headers=_csrf()).status_code == 204

    assert not own.exists() and not own.with_suffix(".meta.json").exists()
    assert bystander.exists() and bystander.with_suffix(".meta.json").exists()
    assert db_documents.get_sync(document_id=bystander_id).status == "active"


# ============================================================================
# H. Finding 4: upload maps a genuine Qdrant/index availability failure to
# 503, and everything else to 500 — never swapped.
# ============================================================================


class _FailingQdrantClient:
    """Delegates everything to the REAL local QdrantClient except the named
    methods, which raise `exc` — a controlled failure at the ACTUAL index
    boundary (VectorIndex.reconcile_document()'s own client calls), never a
    stub of reconcile_document() itself."""

    def __init__(self, real, methods, exc):
        self._real = real
        self._methods = set(methods)
        self._exc = exc
        self.failed_calls = 0

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name in self._methods:
            def _boom(*args, **kwargs):
                self.failed_calls += 1
                raise self._exc
            return _boom
        return attr


def _assert_ingest_rolled_back_with_no_orphan(owner_uuid: uuid.UUID, uploads_dir: Path) -> None:
    with Session(get_sync_engine()) as session:
        remaining = session.execute(select(Document).where(Document.owner_user_id == owner_uuid)).scalars().all()
    assert remaining == []  # no orphan catalog row of ANY status (no active document)
    assert not uploads_dir.exists() or list(uploads_dir.iterdir()) == []  # physical file + sidecar rolled back


_TRANSPORT = httpx.ConnectError("connection refused")
_AVAILABILITY_FAILURES = [
    ("transport_connect_at_scroll", ("scroll",), ResponseHandlingException(_TRANSPORT)),
    ("transport_timeout_at_upsert", ("upsert",), ResponseHandlingException(httpx.ReadTimeout("timed out"))),
    ("http_503_at_upsert", ("upsert",), UnexpectedResponse(503, "Service Unavailable", b"", httpx.Headers())),
    ("http_500_at_upsert", ("upsert",), UnexpectedResponse(500, "Internal Server Error", b"", httpx.Headers())),
    ("http_429_at_scroll", ("scroll",), UnexpectedResponse(429, "Too Many Requests", b"", httpx.Headers())),
    ("resource_exhausted_at_upsert", ("upsert",), ResourceExhaustedResponse("busy", 5)),
    # Qdrant fully down: even the rollback's own delete fails — the catalog
    # row/file/sidecar must STILL all be rolled back.
    ("qdrant_fully_down", ("scroll", "upsert", "delete"), ResponseHandlingException(_TRANSPORT)),
]


@pytest.mark.parametrize(
    "case_id, methods, exc", _AVAILABILITY_FAILURES, ids=[c[0] for c in _AVAILABILITY_FAILURES]
)
def test_genuine_qdrant_availability_failure_during_upload_maps_to_503_with_full_rollback(
    postgres_db, owner_uuid, real_vector_index, case_id, methods, exc
):
    failing = _FailingQdrantClient(real_vector_index.client, methods, exc)
    real_vector_index.client = failing

    response = _upload(_client(owner_uuid), "notes.txt", b"content whose indexing hits an unavailable Qdrant.")

    assert failing.failed_calls >= 1  # the failure really was injected at the index boundary
    assert response.status_code == 503
    assert response.json() == KB_UNAVAILABLE
    assert "refused" not in response.text and "timed out" not in response.text  # never raw exception text
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


def test_locked_local_qdrant_storage_during_upload_maps_to_503_with_full_rollback(
    postgres_db, owner_uuid, real_vector_index, monkeypatch
):
    """The embedded (`path=`) Qdrant client's own availability failure —
    another client instance already holds the storage folder — reached for
    real, through VectorIndex's own constructor."""

    def _open_second_instance_on_the_same_storage():
        return VectorIndex(
            persist_directory=real_vector_index.persist_directory,
            embeddings=DeterministicFakeEmbeddings(),
            collection_name=real_vector_index.collection_name,
        )

    with pytest.raises(VectorIndexUnavailableError):
        _open_second_instance_on_the_same_storage()

    monkeypatch.setattr(app_documents, "get_vector_index", _open_second_instance_on_the_same_storage)
    response = _upload(_client(owner_uuid), "notes.txt", b"content whose Qdrant storage is locked elsewhere.")

    assert response.status_code == 503
    assert response.json() == KB_UNAVAILABLE
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


def _openai_connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"))


class _FailingEmbeddings:
    def embed_documents(self, texts):
        raise _openai_connection_error()

    def embed_query(self, text):
        raise _openai_connection_error()


_GENERIC_INDEX_BOUNDARY_FAILURES = [
    # Qdrant-typed, but NOT availability:
    ("http_400_client_error", ("upsert",), UnexpectedResponse(400, "Bad Request", b"", httpx.Headers())),
    ("http_404_collection_missing", ("scroll",), UnexpectedResponse(404, "Not Found", b"", httpx.Headers())),
    ("malformed_response_not_transport", ("upsert",), ResponseHandlingException(ValueError("malformed body"))),
    # Not Qdrant-typed at all:
    ("plain_runtime_error_from_client", ("upsert",), RuntimeError("boom")),
    ("value_error_from_client", ("scroll",), ValueError("Collection not found")),
    ("os_error_from_client", ("upsert",), OSError("disk full")),
]


@pytest.mark.parametrize(
    "case_id, methods, exc",
    _GENERIC_INDEX_BOUNDARY_FAILURES,
    ids=[c[0] for c in _GENERIC_INDEX_BOUNDARY_FAILURES],
)
def test_non_availability_failure_at_the_index_boundary_stays_500_with_full_rollback(
    postgres_db, owner_uuid, real_vector_index, case_id, methods, exc
):
    failing = _FailingQdrantClient(real_vector_index.client, methods, exc)
    real_vector_index.client = failing

    response = _upload(_client(owner_uuid), "notes.txt", b"content that fails for a non-availability reason.")

    assert failing.failed_calls >= 1
    assert response.status_code == 500
    assert response.json() == PROCESSING_FAILED
    assert "boom" not in response.text and "disk full" not in response.text
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


def test_embedding_provider_network_failure_is_not_a_knowledge_base_outage(
    postgres_db, owner_uuid, real_vector_index
):
    """An OpenAI (embedding provider) connection failure is a DIFFERENT
    service's network failure — never classified as Qdrant availability."""
    real_vector_index.embeddings = _FailingEmbeddings()

    response = _upload(_client(owner_uuid), "notes.txt", b"content whose embedding call fails.")

    assert response.status_code == 500
    assert response.json() == PROCESSING_FAILED
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


def test_parser_failure_is_a_generic_processing_failure(postgres_db, owner_uuid, real_vector_index):
    response = _upload(
        _client(owner_uuid), "broken.pdf", b"this is definitely not a valid pdf document", content_type="application/pdf"
    )

    assert response.status_code == 500
    assert response.json() == PROCESSING_FAILED
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


def test_sidecar_storage_failure_is_a_generic_processing_failure(
    postgres_db, owner_uuid, real_vector_index, monkeypatch
):
    monkeypatch.setattr(
        app_documents, "write_sidecar_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("sidecar disk error"))
    )

    response = _upload(_client(owner_uuid), "notes.txt", b"content whose sidecar cannot be written.")

    assert response.status_code == 500
    assert response.json() == PROCESSING_FAILED
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


def test_catalog_failure_at_activation_is_a_generic_processing_failure(
    postgres_db, owner_uuid, real_vector_index, monkeypatch
):
    def _db_down(**kwargs):
        raise OperationalError("UPDATE documents ...", {}, Exception("postgres unreachable"))

    monkeypatch.setattr(db_documents, "mark_active_sync", _db_down)

    response = _upload(_client(owner_uuid), "notes.txt", b"content whose catalog activation fails.")

    assert response.status_code == 500
    assert response.json() == PROCESSING_FAILED
    assert "postgres unreachable" not in response.text
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


def test_source_mutation_during_indexing_is_a_generic_processing_failure(
    postgres_db, owner_uuid, real_vector_index, monkeypatch
):
    monkeypatch.setattr(
        real_vector_index, "reconcile_document",
        lambda *a, **k: (_ for _ in ()).throw(SourceMutatedError("upload:x")),
    )

    response = _upload(_client(owner_uuid), "notes.txt", b"content whose source mutates mid-index.")

    assert response.status_code == 500
    assert response.json() == PROCESSING_FAILED
    _assert_ingest_rolled_back_with_no_orphan(owner_uuid, app_documents.MANAGED_UPLOADS_DIR)


@pytest.mark.asyncio
async def test_ingest_document_reports_a_structured_failure_reason_only_for_availability(
    postgres_db, owner_uuid, real_vector_index
):
    """The application boundary itself (ingest_document()), not just the
    route: failure_reason is the structured, exception-text-free code."""
    real_vector_index.client = _FailingQdrantClient(
        real_vector_index.client, ("upsert",), ResponseHandlingException(_TRANSPORT)
    )
    unavailable = await app_documents.ingest_document(
        file_bytes=b"availability failure body.", extension=".txt", display_name="a.txt", owner_user_id=owner_uuid,
    )
    assert unavailable.success is False
    assert unavailable.failure_reason == app_documents.INGEST_FAILURE_KNOWLEDGE_BASE_UNAVAILABLE

    real_vector_index.client = _FailingQdrantClient(
        real_vector_index.client._real, ("upsert",), RuntimeError("generic")
    )
    generic = await app_documents.ingest_document(
        file_bytes=b"generic failure body.", extension=".txt", display_name="b.txt", owner_user_id=owner_uuid,
    )
    assert generic.success is False
    assert generic.failure_reason is None

    real_vector_index.client = real_vector_index.client._real
    ok = await app_documents.ingest_document(
        file_bytes=b"healthy body.", extension=".txt", display_name="c.txt", owner_user_id=owner_uuid,
    )
    assert ok.success is True and ok.failure_reason is None


def test_is_index_unavailable_error_taxonomy_is_narrow():
    unavailable = [
        VectorIndexUnavailableError("x"),
        ResponseHandlingException(httpx.ConnectError("x")),
        ResponseHandlingException(httpx.ReadTimeout("x")),
        ResponseHandlingException(httpx.RemoteProtocolError("x")),
        ResourceExhaustedResponse("busy", 1),
        UnexpectedResponse(429, "", b"", httpx.Headers()),
        UnexpectedResponse(500, "", b"", httpx.Headers()),
        UnexpectedResponse(503, "", b"", httpx.Headers()),
    ]
    not_unavailable = [
        ResponseHandlingException(ValueError("validation")),
        UnexpectedResponse(400, "", b"", httpx.Headers()),
        UnexpectedResponse(404, "", b"", httpx.Headers()),
        UnexpectedResponse(None, "", b"", httpx.Headers()),
        SourceMutatedError("x"),
        RuntimeError("Storage folder is already accessed by another instance of Qdrant client."),
        ValueError("x"),
        OSError("x"),
        OperationalError("stmt", {}, Exception("db")),
        _openai_connection_error(),
        Exception("x"),
    ]
    assert all(is_index_unavailable_error(e) for e in unavailable)
    assert not any(is_index_unavailable_error(e) for e in not_unavailable)
