"""
Stage 2B-B regression tests: durable managed-upload sidecar metadata
(rag/sidecar.py) and its integration into handlers/document_upload.py's
storage/cleanup lifecycle (Section I/J/V).

All tests use temporary managed-upload directories only — nothing here
ever touches the real (gitignored) data/documents/uploads. No network,
no real Telegram/OpenAI/Qdrant calls beyond the deterministic local fake
embeddings double used for the handler-level tests.
"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rag.identity import sha256_hex, upload_document_id
from rag.sidecar import (
    PathContainmentError,
    SidecarError,
    build_sidecar,
    load_sidecar,
    resolve_managed_upload_path,
    resolve_sidecar_path,
    sidecar_path_for,
    write_sidecar_atomic,
)

_VALID_STEM = "ab" * 16  # 32 lowercase hex chars


# ---------------------------------------------------------------------------
# 1/2: valid sidecar written atomically, schema exactly validated
# ---------------------------------------------------------------------------

def test_sidecar_path_derivation():
    assert sidecar_path_for(Path("/tmp/abc123.pdf")) == Path("/tmp/abc123.meta.json")
    assert sidecar_path_for(Path("/tmp/abc123.txt")).name == "abc123.meta.json"


def test_write_sidecar_atomic_produces_valid_readable_json(tmp_path):
    stem = "a" * 32
    sidecar_path = tmp_path / f"{stem}.meta.json"
    data = build_sidecar(
        document_id=f"upload:{stem}",
        display_name="My Notes.txt",
        stored_name=f"{stem}.txt",
        content_sha256="a" * 64,
        owner_user_id=1,
    )
    write_sidecar_atomic(sidecar_path, data)

    assert sidecar_path.exists()
    loaded = load_sidecar(sidecar_path)
    assert loaded == data

    # No stray temp file left behind.
    assert list(tmp_path.iterdir()) == [sidecar_path]


def test_write_sidecar_atomic_never_leaves_a_partial_file_visible(tmp_path, monkeypatch):
    """If the write fails partway through, no half-written file may ever
    be observable at the final sidecar_path — only fully-absent or
    fully-valid are allowed outcomes."""
    import rag.sidecar as sidecar_module

    sidecar_path = tmp_path / "abc123.meta.json"

    real_replace = sidecar_module.os.replace

    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(sidecar_module.os, "replace", failing_replace)

    with pytest.raises(OSError):
        write_sidecar_atomic(sidecar_path, build_sidecar("upload:x", "n.txt", "x.txt", "b" * 64, owner_user_id=1))

    assert not sidecar_path.exists()
    # The temp file was cleaned up too — no orphaned .tmp-* artifact.
    assert list(tmp_path.iterdir()) == []


def test_load_sidecar_rejects_missing_file(tmp_path):
    with pytest.raises(SidecarError):
        load_sidecar(tmp_path / "does_not_exist.meta.json")


def test_load_sidecar_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.meta.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SidecarError):
        load_sidecar(path)


@pytest.mark.parametrize("missing_field", ["schema_version", "document_id", "display_name", "stored_name", "content_sha256", "owner_user_id"])
def test_load_sidecar_rejects_missing_required_field(tmp_path, missing_field):
    data = build_sidecar("upload:x", "n.txt", "x.txt", "c" * 64, owner_user_id=1)
    del data[missing_field]
    path = tmp_path / "incomplete.meta.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SidecarError):
        load_sidecar(path)


def test_load_sidecar_rejects_unsupported_schema_version(tmp_path):
    data = build_sidecar("upload:x", "n.txt", "x.txt", "d" * 64, owner_user_id=1)
    data["schema_version"] = 999
    path = tmp_path / "future.meta.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SidecarError):
        load_sidecar(path)


# ---------------------------------------------------------------------------
# 3/4: stored filename/document ID correspondence, sha256 matches source
# ---------------------------------------------------------------------------

def test_sidecar_document_id_and_stored_name_correspond_to_physical_upload(tmp_path):
    physical = tmp_path / "deadbeefdeadbeefdeadbeefdeadbeef.pdf"
    content = b"fake pdf bytes for sidecar correspondence test"
    physical.write_bytes(content)

    document_id = upload_document_id(physical.stem)
    content_sha256 = sha256_hex(content)
    sidecar_path = sidecar_path_for(physical)
    write_sidecar_atomic(sidecar_path, build_sidecar(document_id, "Report.pdf", physical.name, content_sha256, owner_user_id=1))

    loaded = load_sidecar(sidecar_path)
    assert loaded["document_id"] == document_id == f"upload:{physical.stem}"
    assert loaded["stored_name"] == physical.name
    assert loaded["content_sha256"] == sha256_hex(physical.read_bytes())


# ---------------------------------------------------------------------------
# 5: duplicate display filenames supported (distinct document_ids)
# ---------------------------------------------------------------------------

def test_duplicate_display_names_get_distinct_document_ids(tmp_path):
    physical_a = tmp_path / "11111111111111111111111111111111.txt"
    physical_b = tmp_path / "22222222222222222222222222222222.txt"
    physical_a.write_bytes(b"first upload content")
    physical_b.write_bytes(b"second upload content")

    for physical, content in ((physical_a, b"first upload content"), (physical_b, b"second upload content")):
        document_id = upload_document_id(physical.stem)
        write_sidecar_atomic(
            sidecar_path_for(physical),
            build_sidecar(document_id, "same_name.txt", physical.name, sha256_hex(content), owner_user_id=1),
        )

    sidecar_a = load_sidecar(sidecar_path_for(physical_a))
    sidecar_b = load_sidecar(sidecar_path_for(physical_b))
    assert sidecar_a["display_name"] == sidecar_b["display_name"] == "same_name.txt"
    assert sidecar_a["document_id"] != sidecar_b["document_id"]


# ---------------------------------------------------------------------------
# 10: sidecars never contain absolute paths. They DO now intentionally
# contain the owner's Telegram id (Stage 3A: `owner_user_id` is the durable
# ownership record — see rag/sidecar.py's module docstring) — that is a
# deliberate, structured field, not an incidental leak, so it is asserted
# present by name rather than being something this test guards against.
# ---------------------------------------------------------------------------

def test_sidecar_never_contains_absolute_path(tmp_path):
    physical = tmp_path / "33333333333333333333333333333333.txt"
    physical.write_bytes(b"content")
    data = build_sidecar(
        upload_document_id(physical.stem), "notes.txt", physical.name, sha256_hex(b"content"), owner_user_id=42,
    )
    write_sidecar_atomic(sidecar_path_for(physical), data)

    raw_text = sidecar_path_for(physical).read_text(encoding="utf-8")
    assert str(tmp_path) not in raw_text
    assert str(physical) not in raw_text
    assert set(json.loads(raw_text).keys()) == {"schema_version", "document_id", "display_name", "stored_name", "content_sha256", "owner_user_id"}


def test_sidecar_owner_user_id_round_trips_and_survives_reload(tmp_path):
    """Stage 3A requirement: ownership must be recoverable independently of
    Telegram session state — i.e. purely by reading the durable sidecar
    back off disk, with no other input."""
    physical = tmp_path / "44444444444444444444444444444444.txt"
    physical.write_bytes(b"owned content")
    data = build_sidecar(
        upload_document_id(physical.stem), "notes.txt", physical.name, sha256_hex(b"owned content"), owner_user_id=987654321,
    )
    write_sidecar_atomic(sidecar_path_for(physical), data)

    reloaded = load_sidecar(sidecar_path_for(physical))
    assert reloaded["owner_user_id"] == 987654321
    assert type(reloaded["owner_user_id"]) is int


# ---------------------------------------------------------------------------
# 6/7/8: handler-level lifecycle — sidecar creation failure cleans source;
# index failure cleans both source + sidecar; cancelled indexing leaves a
# consistent final state
# ---------------------------------------------------------------------------

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
async def test_sidecar_creation_failure_cleans_up_the_physical_file(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    import rag.sidecar as sidecar_module

    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(app_documents.document_loader, "load_document", Mock(return_value=[]))
    monkeypatch.setattr(app_documents.get_vector_index(), "add_documents", Mock())

    def failing_write_sidecar(sidecar_path, data):
        raise OSError("simulated sidecar write failure")

    monkeypatch.setattr(app_documents, "write_sidecar_atomic", failing_write_sidecar)

    _patch_telegram(monkeypatch, document_upload, b"some content")
    message, document = _make_document_message(1, "notes.txt")

    await document_upload.process_document_upload(message, document)

    # No orphan: the physical file that was created just before the
    # sidecar write failed must be cleaned up too — nothing
    # silently-unrebuildable is left behind.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_indexing_failure_cleans_up_both_physical_file_and_sidecar(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(app_documents.document_loader, "load_document", Mock(return_value=["chunk"]))
    monkeypatch.setattr(
        app_documents.get_vector_index(), "add_documents",
        Mock(side_effect=RuntimeError("simulated embedding/upsert failure")),
    )
    delete_document_mock = Mock()
    monkeypatch.setattr(app_documents.get_vector_index(), "delete_document", delete_document_mock)

    _patch_telegram(monkeypatch, document_upload, b"some content")
    message, document = _make_document_message(1, "notes.txt")

    await document_upload.process_document_upload(message, document)

    assert list(tmp_path.iterdir()) == []  # physical file AND sidecar both gone
    delete_document_mock.assert_called_once()  # defensive Qdrant cleanup attempted too


@pytest.mark.asyncio
async def test_cancelled_indexing_before_commit_leaves_no_orphan_artifacts(monkeypatch, tmp_path):
    import handlers.document_upload as document_upload
    import app.documents as app_documents
    import threading

    started = threading.Event()
    release = threading.Event()

    def fake_load_and_index(stored, display_name):
        started.set()
        assert release.wait(timeout=5), "release was never set by the test"
        raise RuntimeError("boom")

    monkeypatch.setattr(app_documents, "_load_and_index_document", fake_load_and_index)
    delete_document_mock = Mock()
    monkeypatch.setattr(app_documents.get_vector_index(), "delete_document", delete_document_mock)
    monkeypatch.setattr(app_documents, "MANAGED_UPLOADS_DIR", tmp_path)

    _patch_telegram(monkeypatch, document_upload, b"some content")
    message, document = _make_document_message(1, "notes.txt")

    task = asyncio.create_task(document_upload.process_document_upload(message, document))

    loop = asyncio.get_event_loop()
    deadline = loop.time() + 5
    while not started.is_set():
        assert loop.time() < deadline, "timed out waiting for indexing worker to start"
        await asyncio.sleep(0.01)

    # Real storage already committed a physical file + sidecar by now.
    created_before_cancel = list(tmp_path.iterdir())
    assert len(created_before_cancel) == 2

    task.cancel()
    for _ in range(20):
        await asyncio.sleep(0.01)
        assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Consistent final state: nothing orphaned — physical file, sidecar,
    # and any (defensive) Qdrant points are all cleaned up together.
    assert list(tmp_path.iterdir()) == []
    delete_document_mock.assert_called_once()


# ---------------------------------------------------------------------------
# 9: malformed sidecar rejected by rebuild — covered end-to-end in
# tests/test_stage2b_rebuild.py; here we prove the rejection at the
# sidecar-loading boundary itself.
# ---------------------------------------------------------------------------

def test_sidecar_with_wrong_field_types_is_rejected(tmp_path):
    path = tmp_path / "bad.meta.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "document_id": 12345,  # must be a string
        "display_name": "n.txt",
        "stored_name": "x.txt",
        "content_sha256": "e" * 64,
    }), encoding="utf-8")
    with pytest.raises(SidecarError):
        load_sidecar(path)


# ---------------------------------------------------------------------------
# Stage 2B-C Section G: hardened sidecar validation regression tests
# (Codex non-blocking findings fixed in this pass — the sidecar is durable
# source-of-truth metadata, so it must reject anything that doesn't exactly
# match the managed-upload identity contract this application produces).
# ---------------------------------------------------------------------------

def _valid_sidecar_data(**overrides):
    data = build_sidecar(
        document_id=f"upload:{_VALID_STEM}",
        display_name="notes.txt",
        stored_name=f"{_VALID_STEM}.txt",
        content_sha256="c" * 64,
        owner_user_id=1,
    )
    data.update(overrides)
    return data


def _write_raw_sidecar(tmp_path, data) -> Path:
    path = tmp_path / f"{_VALID_STEM}.meta.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_valid_conforming_sidecar_round_trips(tmp_path):
    """Sanity check: a fully conforming sidecar (the shape this
    application actually produces) is accepted unchanged."""
    data = _valid_sidecar_data()
    path = _write_raw_sidecar(tmp_path, data)
    assert load_sidecar(path) == data


def test_load_sidecar_rejects_unknown_extra_field(tmp_path):
    data = _valid_sidecar_data()
    data["unexpected_field"] = "sneaky"
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


def test_load_sidecar_rejects_schema_version_true(tmp_path):
    """`True == 1` and `isinstance(True, int)` are both true in Python —
    schema_version must be rejected unless it's an actual int, never a
    bool that merely compares equal to 1."""
    data = _valid_sidecar_data()
    data["schema_version"] = True
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


@pytest.mark.parametrize("bad_document_id", [
    "upload:abc123",             # too short
    "upload:" + "g" * 32,        # non-hex character
    "upload:" + "A" * 32,        # uppercase not accepted
    "ref:" + "a" * 32,           # wrong prefix
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # missing "upload:" prefix entirely
])
def test_load_sidecar_rejects_malformed_document_id(tmp_path, bad_document_id):
    data = _valid_sidecar_data(document_id=bad_document_id)
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


def test_load_sidecar_rejects_document_id_stored_name_mismatch(tmp_path):
    """document_id's UUID stem must correspond to stored_name's UUID stem
    — a sidecar hand-edited (or paired with the wrong file) to claim a
    DIFFERENT document_id than its own stored_name implies must be
    rejected, even though both are individually well-formed."""
    other_stem = "cd" * 16
    data = _valid_sidecar_data(document_id=f"upload:{other_stem}")
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


@pytest.mark.parametrize("bad_sha", [
    "a" * 63,          # too short
    "a" * 65,          # too long
    "A" * 64,          # uppercase not accepted
    "g" * 64,          # non-hex character
])
def test_load_sidecar_rejects_malformed_content_sha256(tmp_path, bad_sha):
    data = _valid_sidecar_data(content_sha256=bad_sha)
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


def test_load_sidecar_rejects_unsupported_stored_name_extension(tmp_path):
    data = _valid_sidecar_data(stored_name=f"{_VALID_STEM}.exe")
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


@pytest.mark.parametrize("bad_stored_name", [
    "../escape.txt",
    "sub/dir.txt",
    "sub\\dir.txt",
    "/etc/passwd.txt",
])
def test_load_sidecar_rejects_path_separators_in_stored_name(tmp_path, bad_stored_name):
    data = _valid_sidecar_data(stored_name=bad_stored_name)
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


# ---------------------------------------------------------------------------
# Stage 2B-C Section H: rebuild path containment for a managed-upload
# stored_name — never a path-separator/absolute escape, never a symlink
# that resolves outside the managed uploads root.
# ---------------------------------------------------------------------------

def test_resolve_managed_upload_path_accepts_a_valid_normal_file(tmp_path):
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    stored_name = f"{_VALID_STEM}.txt"
    (uploads_root / stored_name).write_bytes(b"content")

    resolved = resolve_managed_upload_path(uploads_root, stored_name)
    assert resolved == (uploads_root / stored_name).resolve()
    assert resolved.is_file()


@pytest.mark.parametrize("escaping_stored_name", [
    "../escape.txt",
    "../../etc/passwd.txt",
    "sub/dir.txt",
])
def test_resolve_managed_upload_path_rejects_relative_escape(tmp_path, escaping_stored_name):
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    with pytest.raises(PathContainmentError):
        resolve_managed_upload_path(uploads_root, escaping_stored_name)


def test_resolve_managed_upload_path_rejects_absolute_stored_name(tmp_path):
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    with pytest.raises(PathContainmentError):
        resolve_managed_upload_path(uploads_root, str(outside))


def test_resolve_managed_upload_path_rejects_symlink_escape(tmp_path):
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_bytes(b"top secret content")

    stored_name = f"{_VALID_STEM}.txt"
    link_path = uploads_root / stored_name
    try:
        os.symlink(outside, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    with pytest.raises(PathContainmentError):
        resolve_managed_upload_path(uploads_root, stored_name)


# ---------------------------------------------------------------------------
# Stage 2B-D Blocker 1 (Codex REJECTED Stage 2B-C): the SIDECAR PATH itself
# must be proven safe BEFORE load_sidecar() ever opens it — a
# `uploads/<uuid>.meta.json` implemented as a symlink was previously
# followed and its content trusted regardless of where it pointed.
# ---------------------------------------------------------------------------

def test_resolve_sidecar_path_accepts_a_normal_sidecar_inside_root(tmp_path):
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    physical = uploads_root / f"{_VALID_STEM}.txt"
    physical.write_bytes(b"content")
    sidecar_path_for(physical).write_text("{}", encoding="utf-8")

    resolved = resolve_sidecar_path(uploads_root, physical)
    assert resolved == sidecar_path_for(physical).resolve()
    assert resolved.is_file()


def test_resolve_sidecar_path_rejects_symlink_to_external_json(tmp_path):
    """The exact Codex exploit: the sidecar FILE ITSELF is a symlink to
    valid, well-formed JSON living OUTSIDE uploads_root. Must be rejected
    before that content is ever opened/parsed/trusted."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    physical = uploads_root / f"{_VALID_STEM}.txt"
    physical.write_bytes(b"content")

    outside_json = tmp_path / "outside.meta.json"
    outside_json.write_text(json.dumps(_valid_sidecar_data()), encoding="utf-8")

    sidecar_link = sidecar_path_for(physical)
    try:
        os.symlink(outside_json, sidecar_link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    with pytest.raises(PathContainmentError):
        resolve_sidecar_path(uploads_root, physical)


def test_resolve_sidecar_path_rejects_symlink_to_internal_json_too(tmp_path):
    """Policy: sidecars may not be symlinks, period — a symlink pointing
    INSIDE uploads_root (aliasing another real sidecar record) is rejected
    exactly like one pointing outside it."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    physical = uploads_root / f"{_VALID_STEM}.txt"
    physical.write_bytes(b"content")

    internal_json = uploads_root / "internal_real.meta.json"
    internal_json.write_text(json.dumps(_valid_sidecar_data()), encoding="utf-8")

    sidecar_link = sidecar_path_for(physical)
    try:
        os.symlink(internal_json, sidecar_link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    with pytest.raises(PathContainmentError):
        resolve_sidecar_path(uploads_root, physical)


def test_resolve_sidecar_path_rejects_lexical_mismatch(tmp_path):
    """Defensive self-consistency check: a caller-supplied uploads_root
    that doesn't actually match the physical file's own parent directory
    is rejected before any filesystem stat/resolve call."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    physical = other_dir / f"{_VALID_STEM}.txt"
    physical.write_bytes(b"content")
    sidecar_path_for(physical).write_text("{}", encoding="utf-8")

    with pytest.raises(PathContainmentError):
        resolve_sidecar_path(uploads_root, physical)


def test_resolve_sidecar_path_rejects_missing_sidecar(tmp_path):
    """A candidate that resolves cleanly (no symlink involved) but simply
    doesn't exist fails the final regular-file check."""
    uploads_root = tmp_path / "uploads"
    uploads_root.mkdir()
    physical = uploads_root / f"{_VALID_STEM}.txt"
    physical.write_bytes(b"content")
    # sidecar deliberately never created

    with pytest.raises(PathContainmentError):
        resolve_sidecar_path(uploads_root, physical)
