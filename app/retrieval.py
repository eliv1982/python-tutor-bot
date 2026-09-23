"""
Adapter-independent retrieval application layer (Stage 7A-3).

Wraps rag/query.py's search_documents() (itself a thin, generation-free
wrapper around _validated_similarity_search()) with request-boundary
validation and safe DTO construction for an authenticated HTTP caller — no
FastAPI/Telegram types, no LLM generation, ever.

A raw Qdrant hit's `document_id`/`source` metadata is never handed to a
caller as-is for a PRIVATE (non-reference) result: `source` is ordinary,
mutable Qdrant payload metadata, and the safe, current user-facing name for
a private upload is its PostgreSQL catalog `display_name` — the same
authority app/documents.py's list/detail endpoints read. Every private hit
is therefore re-resolved against db.documents (fail-closed: dropped, not
raised, if the catalog no longer agrees the document is ACTIVE and owned by
the requester) before it is ever returned. A reference hit's `document_id`
is not a UUID (see rag.identity.reference_document_id()), and its `source`
is ALSO never read from Qdrant payload metadata (Stage 7A-3 corrective pass:
`metadata["source"]` is exactly as mutable as every other payload field —
an audit showed a canonical reference hit carrying an injected filesystem
path there reaching the HTTP response). Canonical-reference proof (rag.query.
_is_proven_reference()) authenticates a reference hit's point id/document
identity/content, not that field, so the label is instead derived from the
same version-controlled manifest that proof is built from — see
_reference_source_label().
"""

import uuid
from dataclasses import dataclass
from typing import List, Optional

import db.documents as db_documents
from rag.constants import BUILTIN_REFERENCE_FILES
from rag.identity import parse_upload_document_id, reference_document_id
from rag.query import search_documents
from utils.helpers import await_worker, submit_worker
from utils.logging import logger

DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 10


class RetrievalValidationError(ValueError):
    """Invalid query/top_k — mapped to a fixed 422 by the web adapter."""


class RetrievalUnavailableError(RuntimeError):
    """The knowledge base/index could not be reached — mapped to a fixed
    503 by the web adapter. Never carries the underlying exception's text."""


@dataclass(frozen=True)
class RetrievalHit:
    document_id: str
    source: str
    chunk_index: int
    page: Optional[int]
    content: str


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise RetrievalValidationError("query must be a non-empty, non-whitespace string")
    return query


def _validate_top_k(top_k: object) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not (MIN_TOP_K <= top_k <= MAX_TOP_K):
        raise RetrievalValidationError(f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}")
    return top_k


def _resolve_private_hit(document_id: object, owner_user_id: uuid.UUID) -> Optional[tuple]:
    """Returns (safe_document_id_str, safe_source) for a private hit whose
    document_id parses as this application's own upload-identity shape AND
    the PostgreSQL catalog still agrees it is ACTIVE and owned by
    `owner_user_id` right now, or None otherwise (fails closed — a single
    bad/unreachable catalog lookup drops only that one hit, never the whole
    request, mirroring rag.query._validated_similarity_search()'s own
    per-candidate fail-closed style)."""
    upload_uuid = parse_upload_document_id(document_id)
    if upload_uuid is None:
        return None
    try:
        record = db_documents.get_sync(document_id=upload_uuid)
    except Exception as e:
        logger.warning(
            "Retrieval: catalog lookup failed while resolving a private hit's display name | error_type=%s",
            type(e).__name__,
        )
        return None
    if record is None or record.status not in db_documents.ACTIVE_STATUSES or record.owner_user_id != owner_user_id:
        return None
    return str(upload_uuid), record.display_name


def _reference_source_label(reference_document_id_str: str) -> str:
    """
    Trusted, user-facing `source` for a PROVEN canonical reference hit
    (Stage 7A-3 corrective pass) — derived only from trusted, version-
    controlled identity, never from the hit's own (mutable) Qdrant
    `source` payload field.

    rag.query._is_proven_reference() has already established, before this
    is ever called, that `reference_document_id_str` is a genuine canonical
    reference document id: rag.identity.point_id(document_id, chunk_index)
    equals the point's actual Qdrant id AND that id is a key of the trusted
    manifest (rag.loader.DocumentLoader.expected_reference_point_hashes(),
    built from rag.constants.BUILTIN_REFERENCE_FILES via exactly
    rag.identity.reference_document_id(<manifest name>)). This function
    reverses that same derivation over that same tuple: the label is the
    manifest's own filename (a bare, checked-in name like
    "python-fundamentals.md" — never a filesystem path) whose derived
    reference id equals the proven one.

    If no manifest entry matches (not expected for a genuinely proven hit,
    but never assumed), the proven reference `document_id` string itself is
    returned — safe, stable, and still independent of any Qdrant payload
    text — rather than ever falling back to untrusted metadata.
    """
    for filename in BUILTIN_REFERENCE_FILES:
        if reference_document_id(filename) == reference_document_id_str:
            return filename
    return reference_document_id_str


def _search_and_resolve_sync(query: str, owner_user_id: uuid.UUID, k: int) -> List[RetrievalHit]:
    requesting_user_uuid = str(owner_user_id)
    results = search_documents(query, requesting_user_uuid, k)

    hits: List[RetrievalHit] = []
    for doc, _score in results:
        metadata = doc.metadata
        raw_document_id = metadata.get("document_id")
        resolved_private = _resolve_private_hit(raw_document_id, owner_user_id)
        if resolved_private is not None:
            document_id_out, source_out = resolved_private
        elif parse_upload_document_id(raw_document_id) is not None:
            # A private-shaped document_id that failed the fail-closed
            # catalog re-check above (e.g. deleted between search-time
            # validation and here) — never fall back to raw metadata for
            # what was, or claims to be, a private upload.
            continue
        else:
            # Not a private-upload id shape at all — a proven reference hit
            # (see rag.query._is_proven_reference()): its document_id is
            # authenticated by that proof, but its Qdrant `source` payload
            # field is NOT — the label comes from the trusted manifest.
            document_id_out = str(raw_document_id) if raw_document_id is not None else ""
            source_out = _reference_source_label(document_id_out)
        hits.append(
            RetrievalHit(
                document_id=document_id_out,
                source=source_out,
                chunk_index=metadata.get("chunk_index"),
                page=metadata.get("page"),
                content=doc.page_content,
            )
        )
    return hits


async def search(*, owner_user_id: uuid.UUID, query: str, top_k: Optional[int] = None) -> List[RetrievalHit]:
    """Validated, generation-free document search for `owner_user_id`
    (Stage 7A-3). Raises RetrievalValidationError for a bad query/top_k
    (before any Qdrant/DB work), or RetrievalUnavailableError if the
    knowledge base itself cannot be reached."""
    validated_query = _validate_query(query)
    k = _validate_top_k(DEFAULT_TOP_K if top_k is None else top_k)
    try:
        return await await_worker(submit_worker(_search_and_resolve_sync, validated_query, owner_user_id, k))
    except Exception as e:
        logger.error("Retrieval: search failed | error_type=%s", type(e).__name__)
        raise RetrievalUnavailableError("Knowledge base unavailable") from e
