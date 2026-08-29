"""
Stable, deterministic identity derivation for RAG documents and Qdrant
points (Stage 2B).

Document/point identity must never depend on Python's built-in hash()
(randomized per-process via PYTHONHASHSEED unless disabled, and never a
persistence guarantee even when stable) or on absolute filesystem paths
(which would leak deployment layout into anything derived from them, and
which also are not portable across machines/checkouts). Every id here is a
deterministic function of stable, portable inputs: a reference document's
path RELATIVE to its root, or a managed upload's opaque storage UUID, plus
a fixed namespace UUID.
"""

import hashlib
import uuid
from pathlib import PurePosixPath

# Fixed, arbitrary namespace UUIDs (generated once via uuid4(), then
# hardcoded here permanently). Any fixed UUID is valid as a uuid5()
# namespace — these do not need to be one of the RFC 4122 well-known
# namespaces. Kept separate so reference-document identity and point
# identity can never collide with each other even for coincidentally
# identical input strings.
_REFERENCE_DOCUMENT_NAMESPACE = uuid.UUID("d3f8a1c2-4b6e-4f7a-8c9d-1e2f3a4b5c6d")
_POINT_NAMESPACE = uuid.UUID("f6a2e4c8-7b1d-4a3f-9c0e-2d4f6a8b0c1e")

# "ref:" / "upload:" prefixes keep the two document_id families
# distinguishable at a glance (logs, payloads) and structurally unable to
# collide with each other regardless of what uuid5() ever produces.
_REFERENCE_PREFIX = "ref:"
_UPLOAD_PREFIX = "upload:"


def reference_document_id(relative_path: str) -> str:
    """
    Deterministic identity for a version-controlled reference document,
    derived from its path RELATIVE to the reference-document root — never
    an absolute path, never the display filename alone. Re-computing this
    for the same relative path always yields the same id, which is what
    lets index_documents_directory() recognize "this is the same logical
    document as last startup".
    """
    normalized = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
    return f"{_REFERENCE_PREFIX}{uuid.uuid5(_REFERENCE_DOCUMENT_NAMESPACE, normalized)}"


def upload_document_id(storage_uuid_hex: str) -> str:
    """
    Identity for a managed Telegram upload, derived from the opaque UUID
    stem of its exclusively-created physical storage filename (see
    handlers/document_upload.py's _store_document_exclusively()). The
    "upload:" prefix keeps this namespace disjoint from
    reference_document_id() by construction, not by coincidence.
    """
    return f"{_UPLOAD_PREFIX}{storage_uuid_hex}"


def point_id(document_id: str, chunk_index: int) -> str:
    """
    Deterministic Qdrant point id for one chunk of one logical document.
    Re-indexing the same (document_id, chunk_index) pair always yields the
    same point id — the property that makes upsert-based replacement
    (Stage 2B Section M) converge instead of accumulating duplicates on
    every restart.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


def sha256_hex(data: bytes) -> str:
    """Local, deterministic content fingerprint — never a provider call."""
    return hashlib.sha256(data).hexdigest()
