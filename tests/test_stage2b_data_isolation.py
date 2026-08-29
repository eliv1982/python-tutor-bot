"""
Stage 2B-B regression tests: extends the Stage 1F real-data isolation
proof (tests/test_stage1f_data_isolation.py) to cover the new Qdrant
path and managed-upload sidecars introduced in Stage 2B.

Never inspects private/real file CONTENT — only existence, directory
listings (names only), and mtime/size, exactly like the Stage 1F proof
this extends.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import config as app_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = (PROJECT_ROOT / "data").resolve()
REAL_CHROMA_DB = REAL_DATA_DIR / "chroma_db"
REAL_QDRANT_DIR = REAL_DATA_DIR / "qdrant"
REAL_UPLOADS_DIR = REAL_DATA_DIR / "documents" / "uploads"
REAL_ENV_FILE = PROJECT_ROOT / ".env"
REAL_BOT_LOG = PROJECT_ROOT / "bot.log"


def test_config_data_dir_redirect_also_covers_the_qdrant_path():
    """Session-wide redirect (tests/conftest.py's pytest_configure) covers
    DATA_DIR itself, so anything computed as DATA_DIR / "qdrant" (the
    production VectorIndex default — see rag/index.py) also never
    resolves into the real repo's data/ directory."""
    resolved = app_config.DATA_DIR.resolve()
    assert resolved != REAL_DATA_DIR
    assert REAL_DATA_DIR not in resolved.parents

    production_qdrant_path = (app_config.DATA_DIR / "qdrant").resolve()
    assert production_qdrant_path != REAL_QDRANT_DIR
    assert not str(production_qdrant_path).startswith(str(REAL_DATA_DIR))


def test_real_qdrant_directory_was_never_created():
    """No test in this suite may create the real, gitignored data/qdrant
    directory — every VectorIndex constructed by tests points at a
    pytest tmp_path instead (see rag_fakes.py-based tests)."""
    assert not REAL_QDRANT_DIR.exists(), (
        "data/qdrant exists on disk — something touched the real Qdrant "
        "path instead of a redirected/temporary one"
    )


def test_real_managed_uploads_directory_has_no_test_residue():
    """Every document-upload test in this suite monkeypatches
    MANAGED_UPLOADS_DIR to a pytest tmp_path — the real
    data/documents/uploads (and any .meta.json sidecars in it) must never
    gain new entries from running this suite."""
    if not REAL_UPLOADS_DIR.exists():
        return  # matches the accepted stop-state baseline: none exist yet
    for entry in REAL_UPLOADS_DIR.iterdir():
        assert entry.name == ".gitkeep", (
            f"unexpected entry in real data/documents/uploads: {entry.name!r}"
        )


@pytest.mark.asyncio
async def test_real_repo_state_unchanged_around_a_representative_stage2b_workload(monkeypatch, tmp_path):
    """
    Before/after proof (same technique as Stage 1F's
    test_real_voice_handler_path_does_not_touch_real_repo_data_dir):
    snapshot the real repo's sensitive paths, run a representative slice
    of Stage 2B work — a real document upload through
    handlers.document_upload.process_document_upload() (Telegram/loader/
    vector-index calls mocked, so no network and no real embeddings) plus
    constructing and using a throwaway local VectorIndex — then snapshot
    again and assert nothing changed.

    Never reads .env or bot.log content — only existence + mtime/size.
    """
    def _snapshot():
        return {
            "chroma_listing": sorted(p.name for p in REAL_CHROMA_DB.iterdir()) if REAL_CHROMA_DB.exists() else None,
            "qdrant_exists": REAL_QDRANT_DIR.exists(),
            "uploads_listing": sorted(p.name for p in REAL_UPLOADS_DIR.iterdir()) if REAL_UPLOADS_DIR.exists() else None,
            "env_stat": (REAL_ENV_FILE.stat().st_mtime, REAL_ENV_FILE.stat().st_size) if REAL_ENV_FILE.exists() else None,
            "bot_log_stat": (REAL_BOT_LOG.stat().st_mtime, REAL_BOT_LOG.stat().st_size) if REAL_BOT_LOG.exists() else None,
        }

    before = _snapshot()

    # --- representative workload 1: a full document-upload pipeline ---
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_documents.document_loader, "load_document", Mock(return_value=[]))
    monkeypatch.setattr(app_documents.get_vector_index(), "add_documents", Mock())
    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"isolation proof content"))
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=99999),
        chat=SimpleNamespace(id=99999),
        document=SimpleNamespace(file_name="isolation_proof.txt", mime_type="text/plain", file_id="fid", file_size=10),
    )
    await document_upload.process_document_upload(message, message.document)

    # --- representative workload 2: a throwaway real-local VectorIndex ---
    from rag.index import VectorIndex
    from rag_fakes import DeterministicFakeEmbeddings
    from langchain_core.documents import Document

    vi = VectorIndex(
        persist_directory=tmp_path / "qdrant",
        embeddings=DeterministicFakeEmbeddings(),
        collection_name="isolation_proof_collection",
    )
    try:
        vi.add_documents([Document(page_content="isolation proof", metadata={"document_id": "d1", "chunk_index": 0, "source": "x.md"})])
        vi.similarity_search("isolation proof", requesting_user_id=1, k=1)
        vi.get_stats(requesting_user_id=1)
    finally:
        vi.close()

    after = _snapshot()
    assert after == before, f"real repo state changed during a representative Stage 2B workload: before={before}, after={after}"
