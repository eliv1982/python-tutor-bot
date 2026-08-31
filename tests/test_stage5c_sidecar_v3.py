"""
Stage 5C regression tests: sidecar schema v3 (canonical internal UUID
ownership, `owner_user_uuid`) — validation, malformed-UUID rejection,
legacy v1/v2 parse behavior, and fail-closed handling.

Entirely local filesystem/pure-function tests — no Qdrant, no network, no
PostgreSQL. Mirrors tests/test_stage2b_sidecar.py's conventions.
"""

import json
import uuid

import pytest

from rag.identity import sha256_hex, upload_document_id
from rag.sidecar import (
    SidecarError,
    build_sidecar,
    load_sidecar,
    parse_sidecar_bytes,
    sidecar_path_for,
    write_sidecar_atomic,
)

_VALID_STEM = "ab" * 16  # 32 lowercase hex chars


def _valid_v3_data(**overrides):
    data = build_sidecar(
        document_id=f"upload:{_VALID_STEM}",
        display_name="notes.txt",
        stored_name=f"{_VALID_STEM}.txt",
        content_sha256="c" * 64,
        owner_user_uuid=str(uuid.uuid4()),
    )
    data.update(overrides)
    return data


def _write_raw_sidecar(tmp_path, data):
    path = tmp_path / f"{_VALID_STEM}.meta.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# A. build_sidecar() writes v3 by default
# ---------------------------------------------------------------------------

def test_build_sidecar_defaults_to_schema_version_3():
    data = build_sidecar(
        document_id=f"upload:{_VALID_STEM}",
        display_name="notes.txt",
        stored_name=f"{_VALID_STEM}.txt",
        content_sha256="c" * 64,
        owner_user_uuid=str(uuid.uuid4()),
    )
    assert data["schema_version"] == 3
    assert "owner_user_uuid" in data
    assert "owner_user_id" not in data


def test_v3_sidecar_round_trips_through_write_and_load(tmp_path):
    physical = tmp_path / f"{_VALID_STEM}.txt"
    physical.write_bytes(b"content")
    owner = str(uuid.uuid4())
    data = build_sidecar(
        upload_document_id(_VALID_STEM), "notes.txt", physical.name, sha256_hex(b"content"), owner_user_uuid=owner,
    )
    write_sidecar_atomic(sidecar_path_for(physical), data)

    loaded = load_sidecar(sidecar_path_for(physical))
    assert loaded == data  # exact round trip — v3 carries no synthetic legacy field
    assert loaded["owner_user_uuid"] == owner
    assert "owner_user_id" not in loaded
    assert type(loaded["owner_user_uuid"]) is str


# ---------------------------------------------------------------------------
# B. Malformed owner_user_uuid rejected (fail closed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_owner", [
    "not-a-uuid",
    "A1B2C3D4-E5F6-4A5B-8C9D-1234567890AB",  # non-canonical case (uppercase hex)
    "1234567812341234123412345678901",  # 31 hex chars (one short)
    "",
    "12345678123412341234123456789012",  # 32 hex chars, no hyphens
    123,  # not even a string
    None,
])
def test_load_sidecar_rejects_malformed_owner_user_uuid(tmp_path, bad_owner):
    data = _valid_v3_data(owner_user_uuid=bad_owner)
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


def test_load_sidecar_rejects_v3_missing_owner_user_uuid_field(tmp_path):
    data = _valid_v3_data()
    del data["owner_user_uuid"]
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


def test_load_sidecar_rejects_v3_with_unknown_extra_field(tmp_path):
    data = _valid_v3_data()
    data["owner_user_id"] = 12345  # v2's field must never coexist with v3's
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)


# ---------------------------------------------------------------------------
# C. Legacy v1 (unowned) and v2 (Telegram-int-owned) still parse — migration
# input — but never silently return a v3-shaped result.
# ---------------------------------------------------------------------------

def test_parse_v1_sidecar_returns_both_owner_fields_as_none():
    legacy = {
        "schema_version": 1,
        "document_id": f"upload:{_VALID_STEM}",
        "display_name": "legacy.txt",
        "stored_name": f"{_VALID_STEM}.txt",
        "content_sha256": "c" * 64,
    }
    parsed = parse_sidecar_bytes(json.dumps(legacy).encode("utf-8"))
    assert parsed["owner_user_id"] is None
    assert parsed["owner_user_uuid"] is None


def test_parse_v2_sidecar_returns_int_owner_and_none_uuid():
    legacy = {
        "schema_version": 2,
        "document_id": f"upload:{_VALID_STEM}",
        "display_name": "legacy.txt",
        "stored_name": f"{_VALID_STEM}.txt",
        "content_sha256": "c" * 64,
        "owner_user_id": 987654321,
    }
    parsed = parse_sidecar_bytes(json.dumps(legacy).encode("utf-8"))
    assert parsed["owner_user_id"] == 987654321
    assert parsed["owner_user_uuid"] is None


def test_parse_v3_sidecar_returns_uuid_owner_with_no_synthetic_legacy_field():
    owner = str(uuid.uuid4())
    data = _valid_v3_data(owner_user_uuid=owner)
    parsed = parse_sidecar_bytes(json.dumps(data).encode("utf-8"))
    assert parsed["owner_user_uuid"] == owner
    assert "owner_user_id" not in parsed
    assert parsed == data


def test_v2_sidecar_missing_owner_user_id_field_still_rejected(tmp_path):
    """A v2 sidecar is still required to carry its own owner_user_id field
    (unchanged Stage 3A validation) — v3's introduction must not loosen v2's
    own contract."""
    malformed_v2 = {
        "schema_version": 2,
        "document_id": f"upload:{_VALID_STEM}",
        "display_name": "legacy.txt",
        "stored_name": f"{_VALID_STEM}.txt",
        "content_sha256": "c" * 64,
    }
    path = _write_raw_sidecar(tmp_path, malformed_v2)
    with pytest.raises(SidecarError):
        load_sidecar(path)


def test_unsupported_schema_version_four_rejected(tmp_path):
    data = _valid_v3_data()
    data["schema_version"] = 4
    path = _write_raw_sidecar(tmp_path, data)
    with pytest.raises(SidecarError):
        load_sidecar(path)
