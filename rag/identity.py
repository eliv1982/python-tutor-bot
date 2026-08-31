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
import re
import uuid
from pathlib import PurePosixPath
from typing import Dict, Optional

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

# Qdrant payload `scope` values (Stage 3A). Canonical home is here (Stage
# 5C corrective pass #9) rather than rag/index.py, so rag.identity's own
# is_eligible_private_candidate() below never needs to import back from
# rag.index (which itself already imports from rag.identity) — rag/index.py
# re-exports both names via its own `from rag.identity import ...` so
# existing `from rag.index import SCOPE_PRIVATE, SCOPE_REFERENCE` call
# sites are unaffected.
SCOPE_REFERENCE = "reference"
SCOPE_PRIVATE = "private"


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


# `upload:<32 lowercase hex chars>` — the exact shape upload_document_id()
# produces from a storage UUID's `.hex`. Shared by rag/query.py's retrieval
# validation and rag/index.py's private stats counting (Stage 5C
# corrective pass #2) so both recognize the identical upload-identity
# contract from one place rather than maintaining separate copies of this
# regex.
_UPLOAD_DOCUMENT_ID_RE = re.compile(r"^upload:([0-9a-f]{32})$")


def parse_upload_document_id(document_id: object) -> Optional[uuid.UUID]:
    """
    Parse `document_id` as this application's own upload_document_id()
    shape and return the embedded storage UUID, or None if it doesn't
    match that exact shape — never guessed, never a partial match. A
    non-str input (including None) always returns None rather than
    raising, so callers can pass a raw Qdrant payload value straight
    through without a separate isinstance() guard.
    """
    if not isinstance(document_id, str):
        return None
    match = _UPLOAD_DOCUMENT_ID_RE.match(document_id)
    if not match:
        return None
    return uuid.UUID(match.group(1))


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


def is_canonical_reference_point(
    *,
    actual_point_id: object,
    document_id: object,
    chunk_index: object,
    actual_text: str,
    trusted_reference_points: Dict[str, str],
) -> bool:
    """
    THE single canonical-reference-provenance predicate (Stage 5C
    corrective pass #5, Blocker 4) — shared verbatim by rag/query.py's
    retrieval-time `_is_proven_reference()` and rag/index.py's stats-time
    `count_verified_reference_points()`, so a candidate can never qualify
    as reference for one surface and not the other. An independent audit
    reproduced exactly that drift: a point with the correct ACTUAL Qdrant
    point id and byte-identical canonical text, but missing its
    `document_id`/`chunk_index` payload fields, was excluded by retrieval
    (which required them to reconstruct/cross-check the expected point id)
    while stats counted it anyway (its own prior implementation looked the
    point up directly by id and never inspected those fields at all).

    True only if ALL of the following independently hold — never any one
    alone:
      1. `document_id` is a string and `chunk_index` is an int (a missing/
         malformed pair can never even be evaluated, let alone qualify);
      2. `rag.identity.point_id(document_id, chunk_index)` — the SAME
         deterministic derivation real indexing uses — equals
         `actual_point_id`, the point's OWN real Qdrant identity (never a
         value merely reconstructed from the candidate's own claims and
         trusted blindly);
      3. that computed point id is a genuine key of
         `trusted_reference_points` (see
         rag.loader.DocumentLoader.expected_reference_point_hashes()) —
         i.e. the version-controlled built-in corpus actually expects a
         chunk at this exact identity;
      4. `actual_text` — the exact content that would reach prompt
         construction / be counted — hashes to precisely the content hash
         the trusted manifest recorded for that expected point.

    Deliberately takes plain facts (never a Document object or a raw
    Qdrant payload dict) so both call sites can feed it whatever shape
    they already have on hand — a LangChain Document's `.metadata`/
    `.page_content` for retrieval, a raw `qdrant_client` record's
    `.payload`for stats — without either maintaining its own parallel
    copy of this logic. `scope` is never a parameter here and never
    consulted anywhere in this function — it is ordinary, mutable Qdrant
    payload metadata, never a trust anchor.
    """
    if not isinstance(document_id, str) or not isinstance(chunk_index, int):
        return False
    expected_point_id = point_id(document_id, chunk_index)
    if expected_point_id != actual_point_id:
        return False
    expected_hash = trusted_reference_points.get(expected_point_id)
    if expected_hash is None:
        return False
    return sha256_hex(actual_text.encode("utf-8")) == expected_hash


def is_canonical_uuid_str(value: object) -> bool:
    """
    True only for the exact canonical lowercase-hyphenated UUID string
    shape (`str(uuid.uuid4())`'s own output shape) — rejects uppercase,
    missing hyphens, or any other non-canonical spelling (fail closed
    rather than normalize). Shared by rag/sidecar.py's `owner_user_uuid`
    field validation and rag/index.py's `_visibility_filter()` (Stage 5C
    canonical-identity ownership) so both validate the exact same contract
    rather than maintaining two copies of this regex/round-trip check.
    """
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def is_eligible_private_candidate(
    *,
    scope: object,
    owner_user_uuid: object,
    requesting_user_uuid: str,
    document_id: object,
) -> Optional[uuid.UUID]:
    """
    THE single non-reserved private-candidate eligibility predicate (Stage
    5C corrective pass #9) — shared by rag/query.py's retrieval-time
    `_validated_similarity_search()` and rag/index.py's stats-time
    `VectorIndex.private_chunk_counts_by_document()`, so a candidate can
    never be treated as an eligible private document by one surface and not
    the other. Deliberately does NOT decide reserved-canonical-slot
    exclusion (step 0) — that remains each caller's own separate,
    already-enforced responsibility: `_validated_similarity_search()`'s
    three-way partition drops a reserved-slot candidate before this
    predicate ever runs, and `private_chunk_counts_by_document()`'s own
    `exclude_point_ids=` parameter does the equivalent exclusion at the
    Qdrant-query level. Nor does it perform the batched PostgreSQL catalog
    check — that is a separate, necessarily-batched step every caller still
    runs afterward on whatever UUID this returns.

    An independent acceptance review reproduced a release blocker where a
    NON-reserved Qdrant point with `document_id` in this application's own
    upload-identity shape, an ACTIVE PostgreSQL catalog row genuinely owned
    by the requester, but a Qdrant payload claiming `scope="reference"`
    (with `owner_user_uuid` missing or naming someone else) was returned by
    retrieval (which validated ONLY the catalog side) while statistics
    (which required Qdrant's OWN `owner_user_uuid` to equal the requester,
    via its own Qdrant-level query filter, but never checked `scope`)
    disagreed on ADJACENT scenarios — the two surfaces were each enforcing
    an incomplete, different subset of the required Qdrant-side facts.
    PostgreSQL remains the durable ownership authority, but a derived
    Qdrant point must ALSO be internally self-consistent about being
    private content before it is ever exposed as such: PostgreSQL saying
    the underlying document belongs to the requester is not sufficient on
    its own if the point's own visibility metadata claims otherwise.

    Returns the parsed upload UUID (`parse_upload_document_id(document_id)`)
    if, and only if, EVERY one of the following Qdrant-derived facts holds:
      1. `scope` is exactly SCOPE_PRIVATE — never SCOPE_REFERENCE, missing,
         or any other value. A point Qdrant itself does not mark private
         content is never an eligible private candidate, regardless of what
         its `document_id`/`owner_user_uuid` otherwise claim;
      2. `owner_user_uuid` is a genuine canonical UUID string
         (`is_canonical_uuid_str()`) — malformed or non-string values
         (including `None`, i.e. missing) are never eligible;
      3. that UUID string equals `requesting_user_uuid` exactly;
      4. `document_id` matches this application's own `upload_document_id()`
         shape (`parse_upload_document_id()` returns non-`None`).

    Any failure returns `None` ("not an eligible private candidate for this
    requester") — never raises.
    """
    if scope != SCOPE_PRIVATE:
        return None
    if not is_canonical_uuid_str(owner_user_uuid):
        return None
    if owner_user_uuid != requesting_user_uuid:
        return None
    return parse_upload_document_id(document_id)
