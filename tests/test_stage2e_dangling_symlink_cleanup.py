"""
Stage 2B-E regression tests: Blocker 2 (dangling-symlink cleanup falsely
succeeds) from the latest independent Codex REJECTED verdict, plus the two
small non-blocking fixes bundled into this pass (Section M test-isolation
consistency, Section N production shutdown lifecycle).

Codex reproduced: cleanup_file(dangling_symlink) -> True while the lexical
directory entry (the symlink itself) still existed on disk. Root cause:
Path.exists() FOLLOWS a symlink to check whether its TARGET exists, so a
dangling symlink (target missing) reported exists()==False on both the
"should I even try to unlink this?" guard and the final "did cleanup
succeed?" check — the unlink call was skipped entirely, and the function
still reported success. The fix (utils/helpers.py) uses
os.path.lexists() throughout: it reports the lexical entry's own
presence, regardless of whether it's a symlink or whether the symlink's
target exists.

Symlink-dependent tests are skipped (never xfail/soft-skip silently
treated as pass) if the current platform/user cannot create a symlink —
same convention as tests/test_stage2b_sidecar.py and
tests/test_stage2b_rebuild.py. Codex reproduced this on both Windows and
Linux, so these are attempted for real rather than unconditionally
skipped on Windows.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rag.sidecar import build_sidecar, write_sidecar_atomic
from utils.helpers import cleanup_file


def _make_dangling_symlink(tmp_path: Path, name: str = "dangling_link.txt") -> Path:
    link = tmp_path / name
    target = tmp_path / f"__target_for_{name}"
    target.write_bytes(b"temporary target content")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")
    target.unlink()  # link is now genuinely dangling
    assert os.path.lexists(link)
    return link


# ===========================================================================
# Section L: utils.helpers.cleanup_file() dangling-symlink behavior.
# (Absent-file / regular-file / directory-collision cases are already
# covered by tests/test_stage2d_hardening.py and are unaffected by this
# fix — not duplicated here.)
# ===========================================================================

def test_cleanup_file_unlinks_a_dangling_symlink_itself(tmp_path):
    link = _make_dangling_symlink(tmp_path)

    assert cleanup_file(link) is True
    assert not os.path.lexists(link)  # the symlink entry itself is gone


def test_cleanup_file_returns_false_when_dangling_symlink_unlink_fails(tmp_path, monkeypatch):
    """Controlled failure injection scoped to symlinks only — a genuine
    unlink failure (not a mock of cleanup_file() itself) that leaves the
    lexical entry in place must be reported as incomplete, never as
    success."""
    import pathlib

    link = _make_dangling_symlink(tmp_path)
    real_unlink = pathlib.Path.unlink

    def failing_unlink_for_symlinks(self, *args, **kwargs):
        if self.is_symlink():
            raise OSError("simulated disk failure removing symlink")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", failing_unlink_for_symlinks)

    assert cleanup_file(link) is False
    assert os.path.lexists(link)  # the unlink genuinely failed; entry remains


def test_cleanup_file_dangling_symlink_never_reported_via_exists(tmp_path):
    """Direct proof of the root cause: Path.exists() on a dangling symlink
    is False even BEFORE cleanup — the bug was trusting that as proof of
    absence. Confirms the fixture itself reproduces the exact condition
    Codex found, independent of cleanup_file()'s own correctness (proven
    separately above)."""
    link = _make_dangling_symlink(tmp_path)
    assert link.exists() is False  # exists() follows the symlink -> False
    assert os.path.lexists(link) is True  # but the entry itself is present


# ===========================================================================
# Section J.1 (Stage 2B-F Blocker 2): rag.sidecar.write_sidecar_atomic()'s
# own failure-cleanup had the EXACT SAME root cause as Section L above, one
# level down: on any failure it used to guard the temp file's unlink with
# `tmp_path.exists()`, which follows a symlink to check whether ITS TARGET
# exists. If the managed temporary entry `write_sidecar_atomic()` itself
# just created is replaced with a DANGLING symlink before the failure path
# runs (a race — simulated here via the very os.replace() call whose
# failure triggers cleanup), `exists()` reports False, the unlink is
# skipped, and that dangling symlink is left behind at the managed
# pathname. The fix mirrors Section L's: `tmp_path.unlink(missing_ok=True)`
# operates on the lexical directory entry itself — like the underlying OS
# unlink call, it never follows the entry to its target — so a dangling
# symlink there is removed correctly, and (proven below) a symlink
# pointing at a still-live file has ONLY the link removed, never its
# target.
# ===========================================================================

def test_write_sidecar_atomic_cleanup_removes_a_dangling_symlink_at_the_temp_path(monkeypatch, tmp_path):
    import rag.sidecar as sidecar_module

    sidecar_path = tmp_path / "target.meta.json"
    captured = {}

    def replace_races_to_dangling_symlink_then_fails(src, dst):
        # `src` is write_sidecar_atomic()'s own freshly written temp file.
        # Simulate a race landing exactly where the vulnerable window used
        # to be: something replaces it with a DANGLING symlink (its target
        # already gone) immediately before the failure this test forces.
        captured["tmp_path"] = Path(src)
        Path(src).unlink()
        swap_target = tmp_path / "swap_target_removed_immediately.txt"
        swap_target.write_bytes(b"temporary")
        try:
            os.symlink(swap_target, src)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        swap_target.unlink()  # now genuinely dangling
        assert os.path.lexists(src)
        assert not Path(src).exists()  # exists() would wrongly report absence — the bug this closes
        raise OSError("simulated os.replace failure")

    monkeypatch.setattr(sidecar_module.os, "replace", replace_races_to_dangling_symlink_then_fails)

    data = build_sidecar("upload:" + "a" * 32, "n.txt", "a" * 32 + ".txt", "b" * 64, owner_user_id=1)
    with pytest.raises(OSError):
        write_sidecar_atomic(sidecar_path, data)

    tmp_path_used = captured["tmp_path"]
    assert not os.path.lexists(tmp_path_used)  # the dangling symlink entry itself is gone
    assert not sidecar_path.exists()  # the real sidecar was of course never created


def test_write_sidecar_atomic_cleanup_unlinks_symlink_without_touching_a_live_target(monkeypatch, tmp_path):
    """Same race, but the symlink at the temp path points at a file that
    is still very much alive when cleanup runs — proves the fix's lexical
    unlink removes only the managed temporary entry (the link) and never
    reaches through to delete/modify whatever it points at."""
    import rag.sidecar as sidecar_module

    sidecar_path = tmp_path / "target2.meta.json"
    live_target = tmp_path / "live_target_must_survive.txt"
    live_target.write_bytes(b"content that must survive cleanup untouched")
    captured = {}

    def replace_races_to_live_symlink_then_fails(src, dst):
        captured["tmp_path"] = Path(src)
        Path(src).unlink()
        try:
            os.symlink(live_target, src)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        raise OSError("simulated os.replace failure")

    monkeypatch.setattr(sidecar_module.os, "replace", replace_races_to_live_symlink_then_fails)

    data = build_sidecar("upload:" + "c" * 32, "n.txt", "c" * 32 + ".txt", "d" * 64, owner_user_id=1)
    with pytest.raises(OSError):
        write_sidecar_atomic(sidecar_path, data)

    tmp_path_used = captured["tmp_path"]
    assert not os.path.lexists(tmp_path_used)  # the symlink itself is gone
    assert live_target.exists()  # its target is completely untouched
    assert live_target.read_bytes() == b"content that must survive cleanup untouched"


# ===========================================================================
# Section K/L: handlers.document_upload._cleanup_new_upload() aggregation
# over dangling-symlink physical/sidecar artifacts.
# ===========================================================================

def _make_stored_upload(physical_path: Path, sidecar_path: Path):
    import handlers.document_upload as document_upload

    return document_upload.StoredUpload(
        physical_path=physical_path,
        sidecar_path=sidecar_path,
        document_id="upload:" + "a" * 32,
        content_sha256="b" * 64,
        owner_user_id=1,
    )


def test_cleanup_new_upload_removes_both_dangling_symlinks(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload

    physical_link = _make_dangling_symlink(tmp_path, "physical.txt")
    sidecar_link = _make_dangling_symlink(tmp_path, "sidecar.meta.json")

    stored = _make_stored_upload(physical_link, sidecar_link)
    monkeypatch.setattr(document_upload, "get_vector_index", lambda: Mock())

    result = document_upload._cleanup_new_upload(stored)

    assert result is True
    assert not os.path.lexists(physical_link)
    assert not os.path.lexists(sidecar_link)


def test_cleanup_new_upload_returns_false_when_a_dangling_symlink_unlink_fails(monkeypatch, tmp_path):
    import pathlib

    import handlers.document_upload as document_upload

    physical_link = _make_dangling_symlink(tmp_path, "physical.txt")
    sidecar_link = _make_dangling_symlink(tmp_path, "sidecar.meta.json")

    real_unlink = pathlib.Path.unlink

    def failing_unlink_for_physical(self, *args, **kwargs):
        if self == physical_link:
            raise OSError("simulated disk failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", failing_unlink_for_physical)
    monkeypatch.setattr(document_upload, "get_vector_index", lambda: Mock())

    stored = _make_stored_upload(physical_link, sidecar_link)
    result = document_upload._cleanup_new_upload(stored)

    assert result is False
    assert os.path.lexists(physical_link)  # genuinely still there
    assert not os.path.lexists(sidecar_link)  # the other component still completed


def test_cleanup_new_upload_returns_false_on_qdrant_failure_with_filesystem_success(monkeypatch, tmp_path):
    """Section L.8: Qdrant delete fails but BOTH filesystem artifacts are
    genuinely removed — the overall result must still be False (every
    component must succeed, not merely a majority)."""
    import handlers.document_upload as document_upload

    physical_link = _make_dangling_symlink(tmp_path, "physical.txt")
    sidecar_link = _make_dangling_symlink(tmp_path, "sidecar.meta.json")

    qdrant_mock = Mock()
    qdrant_mock.delete_document = Mock(side_effect=RuntimeError("simulated Qdrant failure"))
    monkeypatch.setattr(document_upload, "get_vector_index", lambda: qdrant_mock)

    stored = _make_stored_upload(physical_link, sidecar_link)
    result = document_upload._cleanup_new_upload(stored)

    assert result is False
    assert not os.path.lexists(physical_link)  # filesystem cleanup genuinely succeeded
    assert not os.path.lexists(sidecar_link)
    qdrant_mock.delete_document.assert_called_once()


def test_cleanup_new_upload_attempts_all_three_components_with_dangling_symlinks_even_when_multiple_fail(monkeypatch, tmp_path):
    """Section L.9: Qdrant AND both dangling-symlink unlinks fail — every
    component must still be attempted (never short-circuited after the
    first failure)."""
    import pathlib

    import handlers.document_upload as document_upload

    physical_link = _make_dangling_symlink(tmp_path, "physical.txt")
    sidecar_link = _make_dangling_symlink(tmp_path, "sidecar.meta.json")

    real_unlink = pathlib.Path.unlink

    def failing_unlink_for_symlinks(self, *args, **kwargs):
        if self.is_symlink():
            raise OSError("simulated disk failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", failing_unlink_for_symlinks)

    qdrant_mock = Mock()
    qdrant_mock.delete_document = Mock(side_effect=RuntimeError("simulated Qdrant failure"))
    monkeypatch.setattr(document_upload, "get_vector_index", lambda: qdrant_mock)

    calls = []
    real_cleanup_file = document_upload.cleanup_file

    def spy_cleanup_file(path):
        calls.append(path)
        return real_cleanup_file(path)

    monkeypatch.setattr(document_upload, "cleanup_file", spy_cleanup_file)

    stored = _make_stored_upload(physical_link, sidecar_link)
    result = document_upload._cleanup_new_upload(stored)

    assert result is False
    assert calls == [physical_link, sidecar_link]  # both attempted despite Qdrant already failing
    qdrant_mock.delete_document.assert_called_once()
    assert os.path.lexists(physical_link) and os.path.lexists(sidecar_link)  # both attempts genuinely failed


# ===========================================================================
# Section L.10: lifecycle-level proof — a physical artifact swapped for a
# dangling symlink by something outside this process (the same threat
# shape as Blocker 1's races) must still be cleaned up correctly by the
# REAL process_document_upload() failure path, never silently reported as
# clean while the dangling symlink's own entry remains.
# ===========================================================================

def _make_document_message(user_id: int, file_name: str, file_id: str = "fid"):
    document = SimpleNamespace(file_name=file_name, mime_type="text/plain", file_id=file_id, file_size=100)
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=user_id), document=document,
    )
    return message, document


def _patch_telegram(monkeypatch, document_upload, file_bytes: bytes):
    monkeypatch.setattr(
        document_upload.bot, "get_file",
        AsyncMock(return_value=SimpleNamespace(file_path="documents/file.txt")),
    )
    monkeypatch.setattr(document_upload.bot, "download_file", AsyncMock(return_value=file_bytes))
    monkeypatch.setattr(document_upload.bot, "send_message", AsyncMock())


@pytest.mark.asyncio
async def test_lifecycle_cleans_up_a_dangling_symlink_left_by_a_swapped_physical_file(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload

    monkeypatch.setattr(document_upload, "MANAGED_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(document_upload, "get_vector_index", lambda: Mock())

    def fake_load_and_index(stored, display_name):
        # Simulate the physical file being replaced with a dangling
        # symlink by something outside this process, then indexing
        # failing — exercising the real _cleanup_new_upload()/
        # cleanup_file() path against exactly the artifact shape Codex
        # proved was falsely reported as cleaned up.
        stored.physical_path.unlink()
        outside_target = tmp_path.parent / "now_removed_target.txt"
        outside_target.write_bytes(b"temporary")
        try:
            os.symlink(outside_target, stored.physical_path)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        outside_target.unlink()
        raise RuntimeError("simulated indexing failure")

    monkeypatch.setattr(document_upload, "_load_and_index_document", fake_load_and_index)

    _patch_telegram(monkeypatch, document_upload, b"some content")
    message, document = _make_document_message(1, "notes.txt")

    await document_upload.process_document_upload(message, document)

    # Nothing lexically remains: the sidecar (regular file) AND the
    # dangling symlink standing in for the physical file are both gone.
    assert list(tmp_path.iterdir()) == []


# ===========================================================================
# Section M: conftest.py must redirect app_config.DOCUMENTS_DIR /
# app_config.MANAGED_UPLOADS_DIR to the SAME temp values rag.constants'
# copies use — not merely rag_constants' own copies.
# ===========================================================================

def test_conftest_redirects_app_config_documents_and_uploads_dirs_to_the_session_temp_tree():
    import config as app_config
    import rag.constants as rag_constants

    assert app_config.DOCUMENTS_DIR == rag_constants.DOCUMENTS_DIR
    assert app_config.MANAGED_UPLOADS_DIR == rag_constants.MANAGED_UPLOADS_DIR
    # And genuinely the temp session tree, never the real repository paths.
    real_repo_root = Path(__file__).resolve().parents[1]
    assert real_repo_root not in app_config.DOCUMENTS_DIR.parents
    assert real_repo_root not in app_config.MANAGED_UPLOADS_DIR.parents


# ===========================================================================
# Section N: main.py's shutdown_bot() deterministically closes the shared
# VectorIndex singleton, and is a safe no-op when one was never
# constructed.
# ===========================================================================

@pytest.mark.asyncio
async def test_shutdown_bot_closes_vector_index_when_one_was_constructed(monkeypatch, tmp_path):
    import rag.constants as rag_constants
    import rag.index as rag_index
    import main as main_module

    monkeypatch.setattr(rag_constants, "DATA_DIR", tmp_path)
    monkeypatch.setattr(rag_index, "_vector_index", None)
    monkeypatch.setattr(main_module.bot, "close_session", AsyncMock())

    vi = rag_index.get_vector_index()
    assert rag_index._vector_index is vi

    await main_module.shutdown_bot()

    assert rag_index._vector_index is None  # closed and reset


@pytest.mark.asyncio
async def test_shutdown_bot_is_a_safe_noop_when_vector_index_never_constructed(monkeypatch):
    import rag.index as rag_index
    import main as main_module

    monkeypatch.setattr(rag_index, "_vector_index", None)
    monkeypatch.setattr(main_module.bot, "close_session", AsyncMock())

    await main_module.shutdown_bot()  # must not raise

    assert rag_index._vector_index is None
