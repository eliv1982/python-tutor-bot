"""
Stage 2B-D regression tests: final import/lifecycle hardening after the
second independent Codex REJECTED verdict on Stage 2B-C.

Covers the four NEW blockers from that verdict, distinct from (and never
reopening) the five previously-fixed Stage 2B blockers:

  1. sidecar-path containment (see tests/test_stage2b_sidecar.py's new
     resolve_sidecar_path() tests and tests/test_stage2b_rebuild.py's new
     symlinked-sidecar build_plan() exploit reproduction — not duplicated
     here);
  2. filesystem cleanup status must be true (this module, Section C/O);
  3. rebuild dry-run must be credential-independent (this module,
     Section D/L/M);
  4. no import-time vector-store/logging state creation (this module,
     Section F/G/L).

All subprocess-based tests here run against an ISOLATED temp copy of only
the modules they need, or against the real repository tree with every
path an app module could touch redirected to a temp directory — never the
real data/qdrant, bot.log, .env, or data/chroma_db.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

REAL_REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_subprocess_env(extra: dict | None = None) -> dict:
    """A minimal environment for a subprocess probe: no TELEGRAM/OpenAI/
    Anthropic credentials, no .env-influencing variables, no proxy
    variables. `extra` (if given) is layered on top."""
    env = dict(os.environ)
    for key in (
        "TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy", "OPENAI_PROXY",
    ):
        env.pop(key, None)
    if extra:
        env.update(extra)
    return env


def _copy_importable_tree(dest_root: Path) -> None:
    """Copy ONLY the modules the credential-free import/dry-run paths
    actually need into `dest_root` — never the real repo's data/, .env, or
    bot.log. Mirrors test_stage2b_rebuild.py's own isolated-subprocess
    fixture. `services` is included too: `rag.query` (unlike `rag.index`)
    imports `services.text_llm`, which is not required to be credential-
    independent, only state-free — see test_import_rag_query_creates_no_qdrant_or_log_state."""
    for name in ("config.py",):
        shutil.copy2(REAL_REPO_ROOT / name, dest_root / name)
    for pkg in ("rag", "scripts", "utils", "services"):
        shutil.copytree(REAL_REPO_ROOT / pkg, dest_root / pkg, ignore=shutil.ignore_patterns("__pycache__"))


# ===========================================================================
# Section C / O: utils.helpers.cleanup_file() must report its REAL outcome,
# and handlers.app_documents._cleanup_new_upload() must aggregate every
# component honestly.
# ===========================================================================

def test_cleanup_file_returns_true_when_already_absent(tmp_path):
    from utils.helpers import cleanup_file

    missing = tmp_path / "does_not_exist.txt"
    assert cleanup_file(missing) is True


def test_cleanup_file_returns_true_on_successful_unlink(tmp_path):
    from utils.helpers import cleanup_file

    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    assert cleanup_file(target) is True
    assert not target.exists()


def test_cleanup_file_returns_none_treated_as_true_for_none_input():
    from utils.helpers import cleanup_file

    assert cleanup_file(None) is True


def test_cleanup_file_returns_false_when_removal_genuinely_fails(tmp_path):
    """Controlled filesystem failure injection (Section O): pass a
    non-empty DIRECTORY rather than mocking cleanup_file() or unlink()
    itself — Path.unlink() genuinely cannot remove a directory on either
    platform, so this is a real, reproducible failure."""
    from utils.helpers import cleanup_file

    target = tmp_path / "actually_a_directory"
    target.mkdir()
    (target / "child.txt").write_text("x", encoding="utf-8")

    assert cleanup_file(target) is False
    assert target.exists()  # genuinely still there


def _make_stored_upload(physical_path: Path, sidecar_path: Path):
    import uuid
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    document_uuid = uuid.UUID("a" * 32)
    return app_documents.StoredUpload(
        physical_path=physical_path,
        sidecar_path=sidecar_path,
        document_id=f"upload:{document_uuid.hex}",
        document_uuid=document_uuid,
        content_sha256="b" * 64,
        owner_user_id=uuid.uuid4(),
    )


def test_cleanup_new_upload_returns_false_when_physical_unlink_fails(monkeypatch, tmp_path):
    """Requirement 1: Qdrant succeeds + physical unlink fails -> False."""
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    physical = tmp_path / "physical_dir"
    physical.mkdir()
    (physical / "child").write_text("x", encoding="utf-8")
    sidecar = tmp_path / "sidecar.meta.json"
    sidecar.write_text("{}", encoding="utf-8")

    stored = _make_stored_upload(physical, sidecar)
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: Mock())

    result = app_documents._cleanup_new_upload(stored)

    assert result is False
    assert physical.exists()  # the unlink genuinely failed
    assert not sidecar.exists()  # the OTHER component still completed


def test_cleanup_new_upload_returns_false_when_sidecar_unlink_fails(monkeypatch, tmp_path):
    """Requirement 2: Qdrant succeeds + sidecar unlink fails -> False."""
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    physical = tmp_path / "physical.txt"
    physical.write_text("x", encoding="utf-8")
    sidecar = tmp_path / "sidecar_dir"
    sidecar.mkdir()
    (sidecar / "child").write_text("x", encoding="utf-8")

    stored = _make_stored_upload(physical, sidecar)
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: Mock())

    result = app_documents._cleanup_new_upload(stored)

    assert result is False
    assert not physical.exists()
    assert sidecar.exists()  # the unlink genuinely failed


def test_cleanup_new_upload_attempts_every_component_even_when_multiple_fail(monkeypatch, tmp_path):
    """Requirement 4: multiple cleanup failures -> every component is
    still attempted (never short-circuited after the first failure)."""
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    physical = tmp_path / "physical_dir"
    physical.mkdir()
    (physical / "child").write_text("x", encoding="utf-8")
    sidecar = tmp_path / "sidecar_dir"
    sidecar.mkdir()
    (sidecar / "child").write_text("x", encoding="utf-8")

    stored = _make_stored_upload(physical, sidecar)

    qdrant_mock = Mock()
    qdrant_mock.delete_document = Mock(side_effect=RuntimeError("simulated Qdrant failure"))
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: qdrant_mock)

    calls = []
    real_cleanup_file = app_documents.cleanup_file

    def spy_cleanup_file(path):
        calls.append(path)
        return real_cleanup_file(path)

    monkeypatch.setattr(app_documents, "cleanup_file", spy_cleanup_file)

    result = app_documents._cleanup_new_upload(stored)

    assert result is False
    assert calls == [physical, sidecar]  # both attempted despite Qdrant already failing
    qdrant_mock.delete_document.assert_called_once()
    assert physical.exists() and sidecar.exists()  # both attempts genuinely failed


def test_cleanup_new_upload_treats_already_missing_files_as_clean(monkeypatch, tmp_path):
    """Requirement 5: missing file counts as already cleaned."""
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    physical = tmp_path / "already_gone.txt"  # never created
    sidecar = tmp_path / "already_gone.meta.json"  # never created

    stored = _make_stored_upload(physical, sidecar)
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: Mock())

    assert app_documents._cleanup_new_upload(stored) is True


@pytest.mark.asyncio
async def test_lifecycle_never_claims_complete_cleanup_while_physical_file_still_exists(monkeypatch, tmp_path, caplog):
    """Requirement 8, exercised through the REAL process_document_upload()
    lifecycle (not merely _cleanup_new_upload() in isolation): a genuine
    unlink failure on the physical file must surface as a logged
    "cleanup incomplete" warning, and the physical file must genuinely
    still be present afterward — never silently reported as clean.

    Controlled filesystem failure injection: Path.unlink() is patched to
    fail ONLY for the specific physical file this test creates (matched by
    its actual resolved path, discovered via a spy on the real storage
    step) — every other unlink call (e.g. the sidecar's own) behaves
    normally. This exercises the real cleanup_file()/_cleanup_new_upload()
    code paths end to end, not a mock of either.
    """
    import logging
    import pathlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import handlers.document_upload as document_upload
    import app.documents as app_documents
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(app_documents.document_loader, "load_document", Mock(side_effect=ValueError("simulated parse failure")))
    monkeypatch.setattr(app_documents, "get_vector_index", lambda: Mock())

    monkeypatch.setattr(document_upload.bot, "get_file", AsyncMock(return_value=SimpleNamespace(file_path="documents/notes.txt")))
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=b"some content"))
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())

    real_unlink = pathlib.Path.unlink

    def failing_unlink_for_txt(self, *args, **kwargs):
        if self.suffix == ".txt":
            raise OSError("simulated disk failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", failing_unlink_for_txt)

    message = SimpleNamespace(
        from_user=SimpleNamespace(id=1), chat=SimpleNamespace(id=1),
        document=SimpleNamespace(file_name="notes.txt", mime_type="text/plain", file_id="fid", file_size=100),
    )

    with caplog.at_level(logging.WARNING):
        await document_upload.process_document_upload(message, message.document)

    assert "cleanup incomplete" in caplog.text.lower()
    remaining = list(tmp_path.iterdir())
    assert len(remaining) == 1
    assert remaining[0].suffix == ".txt"  # the physical file genuinely still exists
    assert not any(p.name.endswith(".meta.json") for p in remaining)  # the sidecar was removed fine


# ===========================================================================
# Section F/L: merely importing rag.index / rag.query must never construct
# Qdrant state, and get_vector_index()/close_vector_index() form a correct
# lazy-singleton lifecycle.
# ===========================================================================

def test_get_vector_index_is_lazy_singleton_and_close_resets_it(monkeypatch, tmp_path):
    """Uses the real (dummy-keyed, per conftest.py) default OpenAIEmbeddings
    construction path — client construction alone makes no network call,
    only embed_documents()/embed_query() would, and neither is exercised
    here — exactly like test_stage1f_offline_enforcement.py's existing
    real-singleton tests."""
    import rag.constants as rag_constants
    import rag.index as rag_index

    monkeypatch.setattr(rag_constants, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rag_index, "_vector_index", None)

    assert not (tmp_path / "qdrant").exists()

    first = rag_index.get_vector_index()
    assert (tmp_path / "qdrant").exists()  # constructed on first EXPLICIT call

    second = rag_index.get_vector_index()
    assert second is first  # same cached singleton

    rag_index.close_vector_index()
    assert rag_index._vector_index is None

    third = rag_index.get_vector_index()
    assert third is not first  # a fresh instance after close+reset
    third.close()


def test_close_vector_index_is_a_safe_noop_when_never_constructed(monkeypatch):
    import rag.index as rag_index

    monkeypatch.setattr(rag_index, "_vector_index", None)
    rag_index.close_vector_index()  # must not raise
    assert rag_index._vector_index is None


# ===========================================================================
# Section D/L/M: credential-free import/dry-run subprocess proofs. These
# run against ISOLATED temp copies of the repo's importable modules — never
# the real repository configuration, and with credentials explicitly
# ABSENT from the subprocess environment.
# ===========================================================================

@pytest.fixture
def isolated_tree(tmp_path):
    root = tmp_path / "isolated_repo"
    root.mkdir()
    _copy_importable_tree(root)

    documents_dir = root / "data" / "documents"
    documents_dir.mkdir(parents=True)
    import config as app_config
    for filename in app_config.BUILTIN_REFERENCE_FILES:
        (documents_dir / filename).write_text(f"Isolated reference content for {filename}.", encoding="utf-8")

    return root


def _run(args, cwd, env, timeout=60):
    return subprocess.run(
        [sys.executable, *args], cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout,
    )


def test_import_rag_index_creates_no_qdrant_or_log_state(isolated_tree):
    """Section L.1: merely `import rag.index` must exit 0, create no
    data/qdrant, and create no bot.log — with NO credentials in the
    subprocess environment at all."""
    script = "import rag.index\nprint('IMPORT_OK')\n"
    result = _run(["-c", script], cwd=isolated_tree, env=_clean_subprocess_env())

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "IMPORT_OK" in result.stdout
    assert not (isolated_tree / "data" / "qdrant").exists()
    assert not (isolated_tree / "bot.log").exists()


def test_import_rag_query_creates_no_qdrant_or_log_state(isolated_tree):
    """Section L.2: same state-free result for `import rag.query`
    (transitively imports rag.index). rag.query itself still imports the
    full `config` module (RAG_TOP_K) and services.text_llm — unlike
    rag.index, it is not required to be credential-independent, only
    state-free — so minimal dummy credentials are supplied here (never real
    ones; ANTHROPIC_API_KEY is required too since services/anthropic_client.py
    unconditionally constructs its client at its own import time,
    independent of LLM_PROVIDER — pre-existing behavior, not a Stage 2B-D
    concern)."""
    script = "import rag.query\nprint('IMPORT_OK')\n"
    env = _clean_subprocess_env({
        "TELEGRAM_BOT_TOKEN": "123456789:TEST-TOKEN-DO-NOT-USE",
        "OPENAI_API_KEY": "sk-test-dummy-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-dummy-key",
        "LLM_PROVIDER": "openai",
    })
    result = _run(["-c", script], cwd=isolated_tree, env=env)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "IMPORT_OK" in result.stdout
    assert not (isolated_tree / "data" / "qdrant").exists()
    assert not (isolated_tree / "bot.log").exists()


def test_import_utils_logging_creates_no_bot_log(isolated_tree):
    """Section L.3: merely `import utils.logging` must never open bot.log,
    with NO credentials required at all (utils.logging never imports
    config at module level after Stage 2B-D)."""
    script = "import utils.logging\nprint('IMPORT_OK')\n"
    result = _run(["-c", script], cwd=isolated_tree, env=_clean_subprocess_env())

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "IMPORT_OK" in result.stdout
    assert not (isolated_tree / "bot.log").exists()


def test_configure_logging_creates_log_only_at_explicit_call_and_is_idempotent(isolated_tree):
    """Section L.4: configure_logging() creates the log only when
    explicitly called, and a second call does not duplicate FileHandlers."""
    script = textwrap.dedent(
        """
        import logging
        import os

        os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-key"

        import utils.logging as ul
        assert not os.path.exists("bot.log"), "bot.log existed before configure_logging() was ever called"

        ul.configure_logging()
        assert os.path.exists("bot.log"), "configure_logging() did not create bot.log"
        file_handlers_after_first = [h for h in ul.logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers_after_first) == 1

        ul.configure_logging()
        file_handlers_after_second = [h for h in ul.logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers_after_second) == 1, "second configure_logging() call duplicated FileHandlers"

        print("CONFIGURE_OK")
        """
    )
    result = _run(["-c", script], cwd=isolated_tree, env=_clean_subprocess_env())

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "CONFIGURE_OK" in result.stdout


def test_get_vector_index_lifecycle_creates_temp_qdrant_only_on_explicit_call(isolated_tree):
    """Section L.5: get_vector_index() creates Qdrant state only on its
    first explicit call, returns the same singleton on a second call, and
    close+reset allows a clean recreation. Client construction alone (no
    embed_documents()/embed_query() call) makes no network call, so the
    real default OpenAIEmbeddings construction path is exercised directly
    with a dummy key — no fake embeddings double needed here."""
    script = textwrap.dedent(
        """
        import os
        os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:TEST-TOKEN-DO-NOT-USE"
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-key"

        import rag.index as ri

        assert not os.path.exists(os.path.join("data", "qdrant")), "Qdrant state existed before any explicit call"

        first = ri.get_vector_index()
        assert os.path.exists(os.path.join("data", "qdrant"))
        second = ri.get_vector_index()
        assert second is first

        ri.close_vector_index()
        assert ri._vector_index is None

        third = ri.get_vector_index()
        assert third is not first
        third.close()

        print("LIFECYCLE_OK")
        """
    )
    result = _run(["-c", script], cwd=isolated_tree, env=_clean_subprocess_env())

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "LIFECYCLE_OK" in result.stdout


# ===========================================================================
# Section M: `python -m scripts.rebuild_qdrant` dry-run and --help must
# succeed with NO provider credentials and NO `.env` present at all.
# ===========================================================================

def test_rebuild_dry_run_succeeds_with_no_credentials_and_no_env_file(isolated_tree):
    env = _clean_subprocess_env()
    assert not (isolated_tree / ".env").exists()

    result = _run(["-m", "scripts.rebuild_qdrant"], cwd=isolated_tree, env=env)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ModuleNotFoundError" not in result.stderr
    assert "ValueError" not in result.stderr

    import config as app_config
    assert f"Built-in reference documents found: {len(app_config.BUILTIN_REFERENCE_FILES)}" in result.stdout
    assert "Managed uploads with valid sidecars found: 0" in result.stdout
    assert "Dry run only" in result.stdout

    assert not (isolated_tree / "data" / "qdrant").exists()
    assert not (isolated_tree / "bot.log").exists()


def test_rebuild_help_succeeds_with_no_credentials_and_creates_no_state(isolated_tree):
    env = _clean_subprocess_env()

    result = _run(["-m", "scripts.rebuild_qdrant", "--help"], cwd=isolated_tree, env=env)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "--apply" in result.stdout
    assert not (isolated_tree / "data" / "qdrant").exists()
    assert not (isolated_tree / "bot.log").exists()


def test_rebuild_dry_run_finds_managed_uploads_when_present(isolated_tree):
    """Same credential-free contract, but with a real managed upload
    present — proves the dry-run plan genuinely enumerates uploads_dir
    too, not merely reference documents. This isolated tree deliberately
    has no `db` package at all (see _copy_importable_tree()), so it also
    doubles as the Stage 5C corrective pass proof that a syntactically
    valid v3 sidecar is NEVER, by itself, sufficient to plan a private
    document for indexing: with the PostgreSQL catalog entirely
    unreachable (not even importable here), the upload is enumerated but
    fails closed — reported as skipped, never as a planned document — and
    the dry run still completes cleanly (exit 0), never crashing merely
    because the catalog authority happens to be unavailable."""
    import uuid
    from rag.identity import sha256_hex, upload_document_id
    from rag.sidecar import build_sidecar, sidecar_path_for, write_sidecar_atomic

    uploads_dir = isolated_tree / "data" / "documents" / "uploads"
    uploads_dir.mkdir(parents=True)
    stem = "f" * 32
    physical = uploads_dir / f"{stem}.txt"
    physical.write_bytes(b"isolated managed upload content")
    write_sidecar_atomic(
        sidecar_path_for(physical),
        build_sidecar(upload_document_id(stem), "notes.txt", physical.name, sha256_hex(b"isolated managed upload content"), owner_user_uuid=str(uuid.uuid4())),
    )

    env = _clean_subprocess_env()
    result = _run(["-m", "scripts.rebuild_qdrant"], cwd=isolated_tree, env=env)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    # A syntactically valid sidecar alone is never sufficient (Stage 5C
    # corrective pass) — with no reachable/importable PostgreSQL catalog,
    # this upload fails closed rather than being planned.
    assert "Managed uploads with valid sidecars found: 0" in result.stdout
    assert "Managed uploads skipped" in result.stdout
    assert not (isolated_tree / "data" / "qdrant").exists()
    assert not (isolated_tree / "bot.log").exists()
