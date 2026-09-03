"""
Stage 6C regression tests: Qdrant ownership preservation through merge and
unlink (Section N; Stage 6C corrective pass, independent-audit MINOR 3).
Uses the SAME real, local-persistent Qdrant test boundary
tests/test_stage5c_qdrant_ownership.py already established (VectorIndex +
tests/rag_fakes.py's DeterministicFakeEmbeddings) — never a fake PostgreSQL
field standing in for it.

Strengthened over the original version of this module (independent-audit
finding): rather than only diffing a point's raw payload before/after,
this module now:
  1. spies on the ACTUAL Qdrant client mutation methods VectorIndex itself
     calls (`upsert`/`delete` — the only two mutation entry points
     rag/index.py's own production code ever reaches; there is no
     set_payload/overwrite_payload call anywhere in that module), wrapping
     (never replacing) the real bound methods of the SAME client instance
     the seeding/retrieval calls below also use;
  2. runs merge/unlink while that spy is installed and asserts EXACTLY
     ZERO mutation calls of either kind — not merely "the payload looks
     the same afterward", which could not distinguish "never touched" from
     "touched and coincidentally rewritten identically";
  3. proves retrieval through rag.query._validated_similarity_search() —
     the actual application/catalog-backed retrieval boundary
     rag.query.query_knowledge_base() itself calls (with
     rag.query.get_vector_index monkeypatched to this test's own `vi`
     instance) — never only the lower-level
     VectorIndex.similarity_search_with_score() this module used to call
     directly, which skips the PostgreSQL catalog-ownership cross-check
     entirely. This requires real upload-shaped document ids
     (`upload:<32 hex chars>`, via rag.identity.upload_document_id()) and a
     genuinely ACTIVE `documents` catalog row for each seeded point — a
     private candidate that fails EITHER check is dropped by
     `_validated_similarity_search()` regardless of what Qdrant itself
     says, so this is what proves ownership survives at the boundary the
     application ACTUALLY queries through, not just at the raw vector
     store.
"""

import hashlib
import random
import uuid
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

import db.documents as db_documents
import db.github_identity as db_github_identity
import db.identity as db_identity
import db.telegram_link as db_telegram_link
import rag.query as rag_query
from rag.identity import upload_document_id
from rag.index import VectorIndex
from rag_fakes import DeterministicFakeEmbeddings


@pytest.fixture(autouse=True)
def _default_fake_preferences():
    yield


@pytest.fixture(autouse=True)
def _default_fake_documents_catalog():
    """Shadows conftest.py's same-named autouse fixture — this module now
    seeds REAL `documents` catalog rows (via db.documents.create_pending_sync()/
    mark_active_sync()) so rag.query._validated_similarity_search()'s own
    catalog cross-check (db.documents.get_active_owners_sync()) exercises
    genuine PostgreSQL, never the offline in-memory fake."""
    yield


@pytest.fixture
def vi(tmp_path):
    index = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="stage6c_preservation_test",
    )
    yield index
    index.close()


def _doc(text, document_id, owner_user_uuid, chunk_index=0, source="private.md"):
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "chunk_index": chunk_index,
            "source": source,
            "owner_user_uuid": owner_user_uuid,
        },
    )


def _github_only_user():
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    return db_github_identity.resolve_or_create_user_by_github_id_sync(github_id)


def _payload_for(vi, doc_id: str):
    records, _ = vi.client.scroll(collection_name=vi.collection_name, limit=100, with_payload=True)
    for r in records:
        if r.payload.get("document_id") == doc_id:
            return dict(r.payload)
    return None


def _seed_private_upload(vi, *, owner_user_id: uuid.UUID, content: str) -> str:
    """Seeds ONE real, catalog-backed private upload — a genuine
    `upload:<32 hex chars>` document_id (rag.identity.upload_document_id())
    with a matching, ACTIVE `documents` catalog row — indexes it into `vi`,
    and returns the document_id. Only a document_id of this exact shape,
    backed by an active catalog row, can ever survive
    rag.query._validated_similarity_search()'s combined Qdrant-metadata +
    PostgreSQL-catalog eligibility check (see rag.identity.
    is_eligible_private_candidate())."""
    storage_uuid = uuid.uuid4()
    doc_id = upload_document_id(storage_uuid.hex)
    db_documents.create_pending_sync(
        document_id=storage_uuid,
        owner_user_id=owner_user_id,
        stored_name=f"{storage_uuid.hex}.txt",
        display_name="preservation-test.txt",
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    db_documents.mark_active_sync(document_id=storage_uuid)
    vi.add_documents([_doc(content, doc_id, owner_user_uuid=str(owner_user_id))])
    return doc_id


def _retrieve(vi, monkeypatch, *, query: str, requesting_user_uuid: str):
    """Retrieval through the REAL application/catalog-backed boundary
    (rag.query._validated_similarity_search()), with rag.query's own
    get_vector_index() redirected to THIS test's `vi` instance — never a
    disconnected/second VectorIndex."""
    monkeypatch.setattr(rag_query, "get_vector_index", lambda: vi)
    return rag_query._validated_similarity_search(query, requesting_user_uuid, k=5)


# ---------------------------------------------------------------------------
# A. Merge preserves ownership, makes zero Qdrant mutation calls, and
# survives the real catalog-backed retrieval boundary.
# ---------------------------------------------------------------------------


def test_merge_makes_zero_qdrant_mutations_and_preserves_catalog_backed_retrieval(vi, postgres_db, monkeypatch):
    source = _github_only_user()
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    target = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)

    content = "Target's private notes on Python decorators, seeded before merge."
    doc_id = _seed_private_upload(vi, owner_user_id=target, content=content)
    before_payload = _payload_for(vi, doc_id)
    assert before_payload is not None
    assert before_payload["owner_user_uuid"] == str(target)
    assert before_payload["scope"] == "private"

    raw_secret = __import__("secrets").token_urlsafe(32)
    import hashlib as _hashlib
    from datetime import datetime, timedelta, timezone

    db_telegram_link.create_attempt_sync(
        web_user_id=source,
        link_secret_hash=_hashlib.sha256(raw_secret.encode()).digest(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )

    with patch.object(vi.client, "upsert", wraps=vi.client.upsert) as upsert_spy, \
            patch.object(vi.client, "delete", wraps=vi.client.delete) as delete_spy:
        result = db_telegram_link.redeem_attempt_sync(
            link_secret_hash=_hashlib.sha256(raw_secret.encode()).digest(), telegram_user_id=telegram_id
        )
        assert result.outcome == db_telegram_link.RedemptionOutcome.MERGED
        assert result.target_user_id == target

        assert upsert_spy.call_count == 0, "merge must never write to Qdrant"
        assert delete_spy.call_count == 0, "merge must never delete from Qdrant"

    after_payload = _payload_for(vi, doc_id)
    assert after_payload == before_payload  # bit-for-bit unchanged — no reownership write happened

    results = _retrieve(vi, monkeypatch, query=content, requesting_user_uuid=str(target))
    assert any(d.metadata.get("document_id") == doc_id for d, _ in results)

    # The retired source UUID can never retrieve it (it never owned it,
    # and it no longer exists as a canonical user at all).
    results_as_source = _retrieve(vi, monkeypatch, query=content, requesting_user_uuid=str(source))
    assert all(d.metadata.get("document_id") != doc_id for d, _ in results_as_source)


# ---------------------------------------------------------------------------
# B. Unlink preserves ownership, makes zero Qdrant mutation calls, and
# survives the real catalog-backed retrieval boundary.
# ---------------------------------------------------------------------------


def test_unlink_makes_zero_qdrant_mutations_and_preserves_catalog_backed_retrieval(vi, postgres_db, monkeypatch):
    telegram_id = random.randint(10 ** 11, 10 ** 12 - 1)
    user_id = db_identity.resolve_or_create_user_by_telegram_id_sync(telegram_id)
    github_id = random.randint(10 ** 8, 10 ** 9 - 1)
    from db.engine import get_sync_engine
    from db.models import GithubAccount
    from sqlalchemy.orm import Session

    with Session(get_sync_engine()) as session:
        session.add(GithubAccount(github_user_id=github_id, user_id=user_id))
        session.commit()

    content = "This user's private RAG notes, seeded before unlink."
    doc_id = _seed_private_upload(vi, owner_user_id=user_id, content=content)
    before_payload = _payload_for(vi, doc_id)
    assert before_payload is not None

    with patch.object(vi.client, "upsert", wraps=vi.client.upsert) as upsert_spy, \
            patch.object(vi.client, "delete", wraps=vi.client.delete) as delete_spy:
        outcome = db_telegram_link.unlink_github_sync(user_id=user_id)
        assert outcome == db_telegram_link.UnlinkOutcome.TELEGRAM_KEPT

        assert upsert_spy.call_count == 0, "unlink must never write to Qdrant"
        assert delete_spy.call_count == 0, "unlink must never delete from Qdrant"

    after_payload = _payload_for(vi, doc_id)
    assert after_payload == before_payload

    results = _retrieve(vi, monkeypatch, query=content, requesting_user_uuid=str(user_id))
    assert any(d.metadata.get("document_id") == doc_id for d, _ in results)


# ---------------------------------------------------------------------------
# C. Structural proof: zero Qdrant/rag imports anywhere in Stage 6C's own
# persistence/application modules.
# ---------------------------------------------------------------------------


def test_telegram_link_modules_never_import_rag_or_qdrant():
    """Structural boundary check via a fresh subprocess (never a docstring/
    source-text substring check, since both modules' own docstrings
    legitimately discuss Qdrant/rag in prose): importing ONLY
    db.telegram_link/app.telegram_link must never pull `rag` or
    `qdrant_client` into sys.modules as a side effect."""
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = (
        "import sys, db.telegram_link, app.telegram_link; "
        "loaded = [m for m in sys.modules if m == 'rag' or m.startswith('rag.') "
        "or m == 'qdrant_client' or m.startswith('qdrant_client.')]; "
        "assert not loaded, loaded"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, cwd=str(project_root)
    )
    assert result.returncode == 0, result.stderr
