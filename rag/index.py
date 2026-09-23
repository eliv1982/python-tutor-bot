"""
Vector Index for RAG.
Creates and manages embeddings using Qdrant (local persistent mode).

Stage 2B: replaces the previous ChromaDB-backed implementation. Qdrant is
treated as fully rebuildable DERIVED state, never itself the source of
truth — that role belongs to the version-controlled reference documents
under data/documents/ and, for managed Telegram uploads, the physical file
plus its durable `.meta.json` sidecar (see rag/sidecar.py). See
scripts/rebuild_qdrant.py for the operator-facing rebuild path.
"""

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import httpx
import openai
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

import rag.constants as rag_constants
from rag.constants import BUILTIN_REFERENCE_FILES
from rag.identity import (
    SCOPE_PRIVATE,
    SCOPE_REFERENCE,
    is_canonical_reference_point,
    is_canonical_uuid_str,
    is_eligible_private_candidate,
    point_id as make_point_id,
    reference_document_id,
    sha256_hex,
)
from rag.loader import document_loader
from utils.logging import logger

# Stage 2B-D Section H: everything above is import-safe with zero
# credentials/environment configured (rag.constants is pure; rag.loader/
# rag.identity/utils.logging no longer import config.py's credential-
# validating module). `config.OPENAI_API_KEY` is imported lazily, only
# inside VectorIndex.__init__() when it actually needs to build the default
# OpenAIEmbeddings client (see below) — constructing a VectorIndex with an
# explicit `embeddings=` (as every test in this suite does) never requires
# it, and merely importing this module never constructs anything at all
# (see get_vector_index() at the bottom of this module).


# Payload fields that are safe to persist in a Qdrant point and safe to
# surface back out as LangChain Document metadata. Deliberately excludes
# `file_path` (an absolute filesystem path attached by rag/loader.py for
# internal use only) and any other non-listed metadata — see Stage 2B-B
# Section L ("no absolute path leak").
#
# `owner_user_uuid`/`scope` (Stage 3A, ownership migrated to canonical
# UUID Stage 5C): the multi-user isolation payload. `owner_user_uuid` is
# copied straight from chunk metadata like every other field here
# (present only for managed uploads — rag/loader.py never sets it for
# reference documents). `scope` is NOT read from metadata at all —
# _safe_payload() below always computes and writes it itself, purely from
# whether `owner_user_uuid` ended up present, so the two fields can never
# disagree with each other in a stored payload. There is no code path
# anywhere in this class that writes or accepts a legacy integer
# `owner_user_id` payload field — see rag_constants.QDRANT_COLLECTION_NAME
# for the fresh-collection migration strategy that keeps this a clean
# UUID-only contract rather than a mixed integer/string one.
_SAFE_PAYLOAD_FIELDS = ("source", "document_id", "chunk_index", "content_sha256", "stored_name", "page", "owner_user_uuid", "scope")

# SCOPE_REFERENCE / SCOPE_PRIVATE (Stage 3A): re-exported from rag.identity
# (their canonical home since Stage 5C corrective pass #9 — see that
# module's own comment) so existing `from rag.index import SCOPE_PRIVATE,
# SCOPE_REFERENCE` call sites keep working unchanged.


class SourceMutatedError(RuntimeError):
    """
    Raised by reconcile_document() (Stage 2B-C Section I) when a source
    file's content changed between an initial SHA-256 read and a
    re-verification read taken immediately before embedding/mutating
    Qdrant — a controlled race (the file was rewritten concurrently) that
    must never result in the NEW, unverified content being indexed under
    the OLD, already-computed hash. No Qdrant mutation occurs when this is
    raised. Deliberately a fixed, safe message — never embeds the file
    path.
    """


class VectorIndexUnavailableError(RuntimeError):
    """
    Raised (Stage 7A-3 corrective pass) when the Qdrant index itself cannot
    be opened because its local storage is held by another client instance/
    process (see VectorIndex._connect_and_ensure_collection()) — the one
    genuine availability failure the embedded (`path=`) client raises as a
    bare, otherwise-indistinguishable RuntimeError. Deliberately a fixed,
    safe message; never embeds the storage path. Recognized, together with
    the remote-client transport/service failures below, by
    is_index_unavailable_error().
    """


def is_index_unavailable_error(exc: BaseException) -> bool:
    """
    True ONLY for a genuine Qdrant/index availability failure (Stage 7A-3
    corrective pass) — so a caller (app/documents.py's ingest_document())
    can tell "the knowledge base could not be reached" apart from every
    other reason an indexing call can fail, without ever inspecting
    exception text.

    Classified as unavailable, and nothing else:
      - VectorIndexUnavailableError (embedded storage held by another
        client — see above);
      - qdrant_client.http.exceptions.ResponseHandlingException whose
        wrapped `.source` is an `httpx.TransportError` (connect/read/write
        timeout, network, protocol failures — the remote client wraps every
        transport error this way). A ResponseHandlingException wrapping
        anything else (notably a pydantic ValidationError from a
        successfully-received but malformed 200 response) is NOT
        availability and is left unclassified;
      - qdrant_client.common.client_exceptions.ResourceExhaustedResponse
        (the server's own 429 + Retry-After backpressure signal);
      - qdrant_client.http.exceptions.UnexpectedResponse with HTTP status
        429 or >= 500 (server overloaded/erroring). A 4xx client error (bad
        request, collection not found, ...) is a request/config problem,
        not availability.

    Deliberately NOT classified: parser/loader errors, OpenAI/embedding
    provider errors (a different service entirely — not the index), local
    file/storage errors, PostgreSQL catalog errors, SourceMutatedError,
    ValueError/RuntimeError raised by application code, or any other
    unrecognized exception. Classification is by exception TYPE only.
    """
    if isinstance(exc, VectorIndexUnavailableError):
        return True
    if isinstance(exc, ResponseHandlingException):
        return isinstance(exc.source, httpx.TransportError)
    if isinstance(exc, ResourceExhaustedResponse):
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and (exc.status_code == 429 or exc.status_code >= 500)
    return False


class VectorIndex:
    """Manages vector embeddings and similarity search backed by a local, persistent Qdrant collection."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embeddings=None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize vector index.

        Args:
            persist_directory: Local Qdrant storage directory (production
                default: DATA_DIR / "qdrant"). Test-injectable.
            embeddings: Embeddings instance to use (production default: the
                hardened OpenAIEmbeddings built below). Test-injectable so
                real-local-Qdrant tests can run with a deterministic fake
                embeddings double and make zero OpenAI calls.
            collection_name: Qdrant collection name (production default:
                config.QDRANT_COLLECTION_NAME). Test-injectable so parallel
                tests never share a collection identity even if they did
                share a client.
        """
        if persist_directory is None:
            # Read dynamically (module-attribute access, not a bound
            # top-level name) so a redirect of rag_constants.DATA_DIR takes
            # effect no matter when it happens relative to this module's
            # own import — this module makes no VectorIndex until something
            # explicitly calls get_vector_index() (Stage 2B-D Blocker 4).
            persist_directory = rag_constants.DATA_DIR / "qdrant"

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name or rag_constants.QDRANT_COLLECTION_NAME

        # Guards every method below that touches self.client. Local
        # persistent Qdrant does not itself serialize concurrent access
        # from multiple threads within one process — this lock is why
        # force_disable_check_same_thread=True below is safe to pass: it
        # opts out of qdrant-client's own same-thread guard specifically
        # BECAUSE this lock provides equivalent serialization instead. This
        # instance is shared across every worker thread an offloaded RAG
        # query/document-upload/startup-index call runs in. Reentrant
        # (RLock) because index_documents_directory() calls clear_index()
        # and add_documents() on itself while already holding the lock.
        self._lock = threading.RLock()

        # Initialize embeddings. model= makes the current effective default
        # (text-embedding-ada-002, 1536 dimensions) explicit rather than
        # implicit — see config.EMBEDDING_MODEL/EMBEDDING_DIMENSIONS.
        #
        # base_url is pinned explicitly: langchain-openai defaults it from
        # the OPENAI_API_BASE env var, and would otherwise pass base_url=
        # None down to the openai SDK, which itself falls back to
        # OPENAI_BASE_URL. Passing it here overrides both.
        #
        # openai_proxy=None overrides langchain-openai's own OPENAI_PROXY
        # env-var fallback (`Field(default_factory=from_env("OPENAI_PROXY",
        # ...))`) outright: an explicit kwarg always wins over the
        # default_factory, so this field is None regardless of what
        # OPENAI_PROXY is set to in the environment, now or after a later
        # .env reload.
        #
        # http_client/http_async_client are pinned to the openai SDK's own
        # public "recommended defaults" factories with trust_env=False,
        # for the same reason as services/openai_client.py: the httpx2
        # clients langchain-openai would otherwise build internally default
        # to trust_env=True, which auto-discovers a proxy from
        # HTTP_PROXY/HTTPS_PROXY/ALL_PROXY or (absent those) OS-level proxy
        # discovery (Windows Registry / macOS system config). Passing these
        # explicitly also short-circuits langchain-openai's own
        # openai_proxy-driven http_client construction in
        # validate_environment(), which would otherwise raise ValueError if
        # both openai_proxy and http_client were simultaneously non-empty.
        if embeddings is None:
            # Lazy, deliberately deferred until this exact point (Stage
            # 2B-D Section H): OPENAI_API_KEY is a genuine secret, sourced
            # from the full credential-validating config module, and is
            # only ever needed when actually constructing the DEFAULT
            # embeddings client — never merely to import this module or to
            # construct a VectorIndex with an injected `embeddings=`.
            from config import OPENAI_API_KEY
            embeddings = OpenAIEmbeddings(
                model=rag_constants.EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
                base_url=rag_constants.OFFICIAL_OPENAI_BASE_URL,
                openai_proxy=None,
                http_client=openai.DefaultHttpx2Client(trust_env=False),
                http_async_client=openai.DefaultAsyncHttpx2Client(trust_env=False),
            )
        self.embeddings = embeddings

        self.client: Optional[QdrantClient] = None
        self._connect_and_ensure_collection()

    def _connect_and_ensure_collection(self) -> None:
        with self._lock:
            # force_disable_check_same_thread=True: see self._lock's
            # docstring above — this application's own RLock is what
            # serializes cross-thread access, not qdrant-client's built-in
            # guard.
            try:
                self.client = QdrantClient(
                    path=str(self.persist_directory),
                    force_disable_check_same_thread=True,
                )
            except RuntimeError as e:
                # The embedded client raises a bare RuntimeError from this
                # constructor ONLY when its storage folder is already held
                # by another client instance/process (portalocker lock) —
                # the one genuine local availability failure it has. Only
                # this single constructor call is inside the try, so no
                # unrelated RuntimeError can be reclassified here.
                logger.error("RAG index: Qdrant storage unavailable | error_type=%s", type(e).__name__)
                raise VectorIndexUnavailableError("Vector index unavailable") from e
            self._ensure_collection()
        # persist_directory is an absolute filesystem path (can reveal the
        # deployment's OS username/layout) — never logged.
        logger.info("RAG index: Qdrant client ready")

    def _ensure_collection(self) -> None:
        """
        Create the collection if it doesn't already exist. NEVER makes a
        provider call to determine vector size: EMBEDDING_DIMENSIONS is an
        explicit constant (Stage 2B Section E), not something discovered
        via a live embed_query() probe.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=rag_constants.EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
        )
        logger.info("RAG index: created new Qdrant collection")

    # ------------------------------------------------------------------
    # Safe payload / metadata conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_payload(document: Document) -> dict:
        """Only retrieval/rebuild/attribution-safe fields ever reach a
        Qdrant payload — see _SAFE_PAYLOAD_FIELDS. In particular, the
        loader's internal `file_path` (an absolute path) never does.

        `scope` (Stage 3A) is never read from `document.metadata` — it is
        always derived here, from whether `owner_user_uuid` ended up
        present in the payload, so a stored point can never carry a
        `scope` that disagrees with its own `owner_user_uuid`."""
        meta = document.metadata
        payload = {"text": document.page_content}
        for field in _SAFE_PAYLOAD_FIELDS:
            if field in meta and meta[field] is not None:
                payload[field] = meta[field]
        payload["scope"] = SCOPE_PRIVATE if payload.get("owner_user_uuid") is not None else SCOPE_REFERENCE
        return payload

    @staticmethod
    def _document_from_payload(payload: Optional[dict], point_id: Optional[str] = None) -> Document:
        """`point_id`, when given, is the point's ACTUAL Qdrant id — never a
        payload field (payload has no id of its own; Qdrant's point id is a
        separate, non-payload identifier). Stored under the internal
        `_qdrant_point_id` metadata key (Stage 5C corrective pass #4,
        Blocker 1): outside `_SAFE_PAYLOAD_FIELDS`, so `_safe_payload()`
        never persists it even if a Document carrying it were ever re-added.
        This is what lets a caller (rag/query.py's `_is_proven_reference()`)
        bind a canonical-reference proof to the point ACTUALLY returned by
        Qdrant, rather than to a point id merely reconstructed from the
        payload's own (mutable) document_id/chunk_index fields."""
        payload = payload or {}
        text = payload.get("text", "")
        metadata = {k: v for k, v in payload.items() if k in _SAFE_PAYLOAD_FIELDS}
        if point_id is not None:
            metadata["_qdrant_point_id"] = point_id
        return Document(page_content=text, metadata=metadata)

    # ------------------------------------------------------------------
    # Multi-user visibility (Stage 3A)
    # ------------------------------------------------------------------

    @staticmethod
    def _visibility_filter(requesting_user_uuid: str) -> Filter:
        """
        The one and only Qdrant filter every normal retrieval/count call
        below is built with. A requesting user may see exactly:
          - every point with `scope="reference"` (shared, everyone);
          - a point with `scope="private"` ONLY where its own
            `owner_user_uuid` equals `requesting_user_uuid`.

        No other private point is ever returned. There is no parameter or
        code path in this class that skips this filter for a normal
        similarity_search()/similarity_search_with_score()/get_stats()
        call — see those methods below; the only thing that ever touches
        the full, unfiltered collection is index-time bookkeeping
        (list_document_ids(), _existing_points_detail()), which returns
        identifiers/hashes only, never document content, and is never
        reachable from a Telegram request.

        Raises ValueError (never silently substitutes a default, and never
        proceeds with a "search everything" fallback) if
        `requesting_user_uuid` is not a genuine canonical UUID string —
        `None`, a non-str, or any non-canonical spelling are all rejected
        the same way `rag.sidecar.parse_sidecar_bytes()` already rejects a
        malformed `owner_user_uuid`. Since this is now str-typed (Stage
        5C), the earlier `isinstance(x, bool)` special-case (bools are int
        subclasses in Python) no longer applies — a str can't be a bool.
        """
        if (
            requesting_user_uuid is None
            or not isinstance(requesting_user_uuid, str)
            or not is_canonical_uuid_str(requesting_user_uuid)
        ):
            raise ValueError(
                "requesting_user_uuid must be a real, canonical internal user UUID string — "
                "retrieval must never run without one"
            )
        return Filter(
            should=[
                FieldCondition(key="scope", match=MatchValue(value=SCOPE_REFERENCE)),
                Filter(
                    must=[
                        FieldCondition(key="scope", match=MatchValue(value=SCOPE_PRIVATE)),
                        FieldCondition(key="owner_user_uuid", match=MatchValue(value=requesting_user_uuid)),
                    ]
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Internal Qdrant helpers — every caller already holds self._lock
    # ------------------------------------------------------------------

    def _existing_points_detail(self, document_id: str) -> Dict[str, dict]:
        """Local-only lookup: every existing point id for `document_id`,
        mapped to a dict of its ACTUAL stored `text` payload (never a hash
        field) plus its stored scope/owner_user_uuid payload values (each
        None if that point has no such field). Never makes a provider call
        itself (Stage 2B Section N) — this is what reconcile_document()
        below uses to classify EXACT CURRENT / EXTRA STALE POINTS ONLY /
        MISSING-OUTDATED without embedding anything.

        `scope`/`owner_user_uuid` are included (Stage 3A pre-upgrade
        compatibility fix, ownership migrated to canonical UUID Stage 5C)
        so a point that matches on content alone but carries stale or
        missing visibility metadata — e.g. a pre-Stage-3A reference point
        indexed before `scope` existed at all, or a pre-Stage-5C point
        still carrying the legacy integer `owner_user_id` field instead of
        `owner_user_uuid` — is never misclassified as current. See
        reconcile_document()'s `expected_all_current` check, the sole
        reader of this data.

        Stage 5C corrective pass #4 (Blocker 8): the stored `content_sha256`
        PAYLOAD FIELD is ordinary, mutable Qdrant metadata — exactly like
        `document_id`/`scope`/`chunk_index` elsewhere in this codebase, it
        attests to nothing about the point's actual stored `text` unless
        independently re-verified. An independent audit reproduced Qdrant
        text corrupted in place while its `content_sha256` metadata field
        was left unchanged, with reconciliation trusting that stale field
        and reporting the point "unchanged" forever. The ACTUAL `text`
        returned here — never any stored hash field, which is no longer
        read by reconcile_document() at all — is what
        `expected_all_current` below compares directly against each
        freshly-parsed expected chunk's own page_content, so a point can be
        classified "current"/"unchanged" only if what is ACTUALLY stored
        genuinely equals the current chunk it claims to be. (Deliberately
        NOT a hash of `text` compared against the document-level
        `content_sha256` the caller recomputes for the WHOLE source file —
        those describe different things: `content_sha256` is a single
        fingerprint of the entire source file, while each point's `text` is
        only ONE chunk of it, so hashing a chunk's text could never
        legitimately equal the whole-file hash in the first place.)"""
        detail: Dict[str, dict] = {}
        offset = None
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=doc_filter,
                limit=256,
                offset=offset,
                with_payload=["text", "scope", "owner_user_uuid"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                detail[str(record.id)] = {
                    "text": payload.get("text", ""),
                    "scope": payload.get("scope"),
                    "owner_user_uuid": payload.get("owner_user_uuid"),
                }
            if offset is None:
                break
        return detail

    def _existing_point_ids(self, document_id: str) -> Set[str]:
        return set(self._existing_points_detail(document_id))

    def _replace_document_points(self, document_id: str, documents: List[Document]) -> None:
        """
        Safe replacement algorithm (Stage 2B Section M): embed the complete
        new chunk set BEFORE any destructive change, upsert it, and only
        THEN delete whatever old points for this document_id are no longer
        present in the new set. An embedding or upsert failure leaves the
        previously-indexed version of this document completely untouched
        — it never gets deleted first. Handles both "brand new document"
        (existing_ids is empty, nothing to delete) and "changed document
        with fewer chunks than before" (stale trailing points removed)
        uniformly.
        """
        texts = [document.page_content for document in documents]
        # Network call — happens first, before anything destructive.
        vectors = self.embeddings.embed_documents(texts)

        points: List[PointStruct] = []
        new_ids: Set[str] = set()
        for document, vector in zip(documents, vectors):
            pid = make_point_id(document_id, document.metadata["chunk_index"])
            new_ids.add(pid)
            points.append(PointStruct(id=pid, vector=vector, payload=self._safe_payload(document)))

        existing_ids = self._existing_point_ids(document_id)

        self.client.upsert(collection_name=self.collection_name, points=points)

        stale_ids = existing_ids - new_ids
        if stale_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=list(stale_ids)),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add (or safely replace) document chunks in the vector store.

        Every document's metadata must already carry `document_id` and
        `chunk_index` (rag/loader.py's load_document() attaches both).
        Chunks are grouped by document_id and each group is replaced via
        _replace_document_points() — this is what makes add_documents()
        idempotent/convergent for the same logical document rather than
        merely additive.

        Args:
            documents: List of document chunks
        """
        try:
            if not documents:
                logger.warning("RAG index add_documents: empty list")
                return
            grouped: "OrderedDict[str, List[Document]]" = OrderedDict()
            for document in documents:
                document_id = document.metadata["document_id"]
                grouped.setdefault(document_id, []).append(document)
            with self._lock:
                for document_id, docs in grouped.items():
                    self._replace_document_points(document_id, docs)
            logger.info("RAG index add_documents | count=%s", len(documents))
        except Exception as e:
            # This call embeds documents via OpenAIEmbeddings (a network
            # call to OpenAI) before writing to Qdrant, so the exception
            # may be a provider error — never log its raw text or a
            # traceback.
            logger.error("RAG index add_documents failed | error_type=%s", type(e).__name__)
            raise

    def reconcile_document(
        self,
        document_id: str,
        file_path: Path,
        *,
        display_name: Optional[str] = None,
        stored_name: Optional[str] = None,
        expected_content_sha256: Optional[str] = None,
        source_bytes: Optional[bytes] = None,
        owner_user_uuid: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Make Qdrant exactly reflect `file_path`'s CURRENT content under
        `document_id`, doing the least possible work to converge (Stage
        2B-C Blocker 2 — startup stale-delete retry — and Blocker 3 —
        non-destructive rebuild — share this one implementation, used by
        both index_documents_directory() and scripts/rebuild_qdrant.py).

        A matching content_sha256 on SOME/existing points is never treated
        as sufficient on its own (the Codex-proven bug this replaces): the
        exact expected point-ID SET, derived locally from
        (document_id, chunk_index) without embedding anything, is compared
        against what's actually in Qdrant for this document_id.

        Stage 3A pre-upgrade compatibility (ownership migrated to
        canonical UUID Stage 5C): content_sha256 identity alone is
        likewise never sufficient — a point is "current" only if its
        stored `scope`/`owner_user_uuid` payload fields ALSO already match
        what this call expects (`scope="reference"`+no owner when
        `owner_user_uuid` is None, else `scope="private"`+that exact
        owner). This is what makes a pre-Stage-3A point — indexed before
        `scope` existed, so it has matching id/hash but no visibility
        metadata at all — OR a pre-Stage-5C point still carrying the
        legacy integer `owner_user_id` field instead of `owner_user_uuid`
        — ineligible for "unchanged": both are instead treated exactly
        like outdated content below and safely re-embedded/upserted with
        correct current metadata, rather than being silently accepted as
        current and permanently excluded from the visibility filter
        (_visibility_filter()).

        Classifies into exactly one of four outcomes:
          - "unchanged": every expected point already exists with the
            current content_sha256 AND the expected scope/owner_user_uuid,
            and no extra stale points remain. Zero embedding, upsert, or
            delete calls. Also returned (Stage 5C corrective pass #6,
            Blocker 2) when the CURRENT source has zero chunks AND no
            points exist for this document_id yet — nothing to converge.
          - "stale_pruned": every expected point already exists with the
            current content_sha256 AND the expected scope/owner_user_uuid,
            but extra (stale) points also remain — e.g. left over from a
            previous stale-delete failure. Only those extras are deleted;
            zero embedding calls (this is what makes a deterministic retry
            converge for free).
          - "reindexed": one or more expected points are missing, outdated,
            or carry stale/missing visibility metadata — full safe
            replacement via _replace_document_points() (embed the complete
            new set, upsert, delete stale extras only after a successful
            upsert).
          - "emptied" (Stage 5C corrective pass #6, Blocker 2): the CURRENT
            source has zero chunks, but points from a PREVIOUS, non-empty
            version of this document_id still exist — Qdrant is derived
            state, so the expected point set being empty must still
            converge it to empty. Every existing point for this
            document_id is deleted; zero embedding calls. This is the fix
            for the exact defect an independent audit reproduced: an
            already-active document reconciled to zero chunks used to
            return "unchanged" WITHOUT ever inspecting/removing its old
            points, leaving stale content permanently retrievable.

        Section I hash/read consistency: `file_path`'s content is hashed
        once before loading and re-verified immediately before any
        embedding call. If the content changed in between (a concurrent
        rewrite), raises SourceMutatedError and performs NO Qdrant
        mutation — the stale, no-longer-accurate hash is never used to
        index the new, unverified content. If `expected_content_sha256` is
        given (managed uploads: the hash recorded in the durable sidecar
        at storage time), it must match the freshly computed hash too.

        Stage 2B-E Section F (plan-to-apply content integrity): when
        `source_bytes` is given (managed uploads reconciled via
        scripts/rebuild_qdrant.py's apply_plan() — see SourceDocument.
        content_bytes), those EXACT bytes — already securely read once, at
        plan-build time, via rag.safe_files.read_regular_file_secure() —
        are used for BOTH the hash check AND the parsed/embedded content:
        `file_path` is NEVER reopened. Stage 2B-F Blocker 1 (an audit
        finding against this method's own prior fix): parsing used to go
        through a private temporary snapshot file written from
        `source_bytes` and then handed to the loader BY PATHNAME — which
        the loader's own parser (PyPDFLoader/TextLoader/Docx2txtLoader)
        would reopen independently, the exact same "hash one object /
        parse another" shape this section's first paragraph closes for
        `file_path`, just moved one level down onto the new snapshot path.
        `document_loader.load_document_bytes()` closes this for good:
        `source_bytes` is parsed directly in memory (an in-memory `Blob`
        for PDF, a `BytesIO` stream for DOCX, a direct UTF-8 decode for
        TXT/MD — see its own docstring), so no pathname of any kind is
        ever written or reopened between the hash check above and parsing
        — there is nothing left for a concurrent write to swap.
        `expected_content_sha256`, if also given, still must match (a
        cheap self-consistency check — the two should always agree since
        both derive from the same bytes). Reference documents
        (source_bytes=None — Section G: version-controlled, deliberately
        not overengineered) keep the previous file_path-based behavior,
        including its second, pre-embed re-verification read below.

        owner_user_uuid (Stage 3A ownership, migrated to canonical UUID
        Stage 5C): canonical internal user UUID string of a managed
        upload's owner, threaded straight to the loader (see
        rag/loader.py's _chunk_and_tag()) so every resulting chunk's
        Qdrant payload carries `scope="private", owner_user_uuid=<this
        value>`. `None` (the default) for reference documents —
        index_documents_directory() never passes it, so reference chunks
        always get `scope="reference"`. Never inferred/guessed here;
        callers (app/documents.py, scripts/rebuild_qdrant.py) are the sole
        source of this value, and each derives it from a durable record
        (the sidecar) rather than from any Telegram session state.

        Returns (status, chunk_count) where chunk_count is the number of
        chunks `file_path` currently splits into (the size of the expected
        set) regardless of status.
        """
        file_path = Path(file_path)
        with self._lock:
            if source_bytes is not None:
                content_sha256 = sha256_hex(source_bytes)
            else:
                content_sha256 = sha256_hex(file_path.read_bytes())
            if expected_content_sha256 is not None and content_sha256 != expected_content_sha256:
                raise SourceMutatedError(document_id)

            if source_bytes is not None:
                chunks = document_loader.load_document_bytes(
                    source_bytes,
                    suffix=file_path.suffix,
                    display_name=display_name,
                    document_id=document_id,
                    content_sha256=content_sha256,
                    stored_name=stored_name,
                    owner_user_uuid=owner_user_uuid,
                )
            else:
                chunks = document_loader.load_document(
                    file_path,
                    display_name=display_name,
                    document_id=document_id,
                    content_sha256=content_sha256,
                    stored_name=stored_name,
                    owner_user_uuid=owner_user_uuid,
                )

            if not chunks:
                # Stage 5C corrective pass #6 (Blocker 2): the CURRENT
                # authoritative source parses to zero chunks, so the
                # expected derived point set for this document_id is
                # EMPTY. Qdrant is derived state (see this module's own
                # docstring) — zero expected chunks must still participate
                # in convergence, never take an early return BEFORE
                # existing points are even inspected. An independent audit
                # reproduced exactly that: an already-active document
                # whose current source was edited down to empty/
                # whitespace-only, with file/sidecar/catalog left mutually
                # consistent about the new (empty) content, left its OLD,
                # now-stale Qdrant points fully retrievable forever,
                # because this branch used to return before ever looking
                # at them.
                #
                # A document with NO existing points at all — e.g. a
                # still-'pending' upload whose very first reconciliation
                # already finds zero chunks (app.documents.EmptyDocumentError
                # / this pass's predecessor, Stage 5C corrective pass #5's
                # pending zero-chunk protection) — has nothing to remove:
                # "unchanged" exactly as before, zero Qdrant calls.
                existing_ids = self._existing_point_ids(document_id)
                if not existing_ids:
                    return ("unchanged", 0)
                # A failure here (e.g. a transient Qdrant error) raises
                # straight out of this method — never caught/swallowed —
                # so a caller can never observe a status implying
                # convergence succeeded when the stale points are actually
                # still there.
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=list(existing_ids)),
                )
                logger.info(
                    "RAG reconcile_document: current source has zero chunks, removed all previously-indexed points"
                )
                return ("emptied", 0)

            expected_ids = {make_point_id(document_id, c.metadata["chunk_index"]) for c in chunks}
            expected_text_by_id = {make_point_id(document_id, c.metadata["chunk_index"]): c.page_content for c in chunks}
            existing_detail = self._existing_points_detail(document_id)
            existing_ids = set(existing_detail)

            # Stage 3A pre-upgrade compatibility (ownership migrated to
            # canonical UUID Stage 5C): a point is only "current" if its
            # ACTUAL stored text equals the current chunk it claims to be
            # AND its stored visibility metadata already matches what this
            # call expects — `scope` equal to the scope this owner_user_uuid
            # implies, and `owner_user_uuid` equal to this call's
            # owner_user_uuid exactly (None for a reference document,
            # meaning "no owner field"). This is what stops a pre-Stage-3A
            # reference point (matching id/content, but no `scope` field at
            # all), or a pre-Stage-5C point still carrying the legacy
            # integer `owner_user_id` field, from being accepted as
            # unchanged and permanently excluded from the visibility filter
            # — see _existing_points_detail()'s docstring.
            #
            # Stage 5C corrective pass #4 (Blocker 8): compares each
            # point's ACTUAL stored text directly against the current
            # chunk's own freshly-parsed page_content — never a stored
            # content_sha256 payload field (mutable metadata that attests
            # to nothing about what is actually stored), and never the
            # document-level `content_sha256` variable (a whole-file
            # fingerprint — comparing it against one chunk's text would
            # never legitimately match at all). A point whose text was
            # corrupted in place, even with its metadata left innocently
            # unchanged, is caught here and re-embedded/replaced.
            expected_scope = SCOPE_PRIVATE if owner_user_uuid is not None else SCOPE_REFERENCE
            expected_all_current = expected_ids <= existing_ids and all(
                existing_detail[pid]["text"] == expected_text_by_id[pid]
                and existing_detail[pid]["scope"] == expected_scope
                and existing_detail[pid]["owner_user_uuid"] == owner_user_uuid
                for pid in expected_ids
            )
            if expected_all_current:
                stale_ids = existing_ids - expected_ids
                if not stale_ids:
                    return ("unchanged", len(expected_ids))
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=list(stale_ids)),
                )
                logger.info("RAG reconcile_document: pruned stale extra points, zero embedding calls")
                return ("stale_pruned", len(expected_ids))

            # MISSING/OUTDATED expected points -> full safe replacement.
            # Re-verify the content hasn't mutated since the hash above was
            # computed, immediately before the embedding call (Section I).
            # Skipped when source_bytes was given: those bytes are already
            # a fully materialized, immutable snapshot — re-hashing them
            # again would always trivially match and reads nothing new.
            if source_bytes is None and sha256_hex(file_path.read_bytes()) != content_sha256:
                raise SourceMutatedError(document_id)
            self._replace_document_points(document_id, chunks)
            return ("reindexed", len(chunks))

    def list_document_ids(self) -> Set[str]:
        """
        Every distinct document_id currently represented by at least one
        point in the collection. Used by scripts/rebuild_qdrant.py's
        apply_plan() to find orphan logical documents (indexed in Qdrant
        but no longer present in source truth) — see Stage 2B-C Blocker 3.
        """
        ids: Set[str] = set()
        offset = None
        with self._lock:
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=["document_id"],
                    with_vectors=False,
                )
                for record in records:
                    doc_id = (record.payload or {}).get("document_id")
                    if doc_id is not None:
                        ids.add(doc_id)
                if offset is None:
                    break
        return ids

    def delete_document(self, document_id: str) -> None:
        """
        Best-effort removal of every point belonging to `document_id`.
        Used as a defensive cleanup safety net (e.g. a brand-new upload
        whose indexing ultimately failed or was cancelled must leave no
        orphan Qdrant points) — not part of the normal replace flow, which
        already deletes exactly its own stale points via
        _replace_document_points().
        """
        try:
            with self._lock:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                    ),
                )
            logger.info("RAG index delete_document")
        except Exception as e:
            logger.error("RAG index delete_document failed | error_type=%s", type(e).__name__)
            raise

    def similarity_search(self, query: str, *, requesting_user_uuid: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            requesting_user_uuid: Canonical internal user UUID string of
                the user this search is being performed for (Stage 3A,
                migrated to UUID Stage 5C). Required, keyword-only, no
                default — there is deliberately no way to call this method
                and search "everything". See _visibility_filter().
            k: Number of results to return

        Returns:
            List of relevant document chunks
        """
        return [doc for doc, _ in self.similarity_search_with_score(query, requesting_user_uuid=requesting_user_uuid, k=k)]

    def similarity_search_with_score(
        self,
        query: str,
        *,
        requesting_user_uuid: str,
        k: int = 3,
        reference_candidate_point_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            requesting_user_uuid: Canonical internal user UUID string of
                the user this search is being performed for (Stage 3A,
                migrated to UUID Stage 5C). Required, keyword-only, no
                default. Every result is either `scope="reference"`
                (visible to everyone) or a `scope="private"` point owned by
                exactly this user — see _visibility_filter(), which also
                raises ValueError if this is missing/invalid rather than
                letting the call silently proceed unfiltered.
            k: Number of results to return
            reference_candidate_point_ids: Stage 5C corrective pass #6
                (Blocker 1). An optional set of DETERMINISTIC expected
                canonical reference point ids — the caller's own trust
                anchor (e.g. rag.loader.DocumentLoader.
                expected_reference_point_hashes().keys()), independent of
                anything any Qdrant point's own MUTABLE `scope`/
                `owner_user_uuid` payload claims. `None`/empty (the
                default) preserves this method's prior behavior exactly:
                one query, filtered solely by `_visibility_filter()`.

                When given and non-empty, a SECOND Qdrant query also runs,
                restricted via a `HasIdCondition` filter to EXACTLY those
                point ids — no scope/owner condition at all. This is what
                makes a genuine canonical point remain a retrieval
                CANDIDATE even when its mutable visibility metadata has
                been changed/corrupted/inconsistent (independent review
                reproduced retrieval silently excluding such a point
                before rag.query._is_proven_reference() ever got a chance
                to evaluate it, while statistics — which already retrieves
                its expected canonical point ids directly, ignoring scope —
                counted it; see count_verified_reference_points()). Candidate
                ACQUISITION here never decides canonical-reference status
                by itself — the caller's own provenance check
                (rag.identity.is_canonical_reference_point(), applied to
                every candidate this method returns) still independently
                re-verifies each one's ACTUAL point id/content against the
                trust anchor before ever treating it as reference.

                This does not weaken private isolation: the ordinary,
                `requesting_user_uuid`-scoped query is unchanged and remains
                the sole source of private candidates — this parameter only
                ever ADDS deterministic-id-restricted candidates, never
                removes or relaxes the private-scoped query's own filter.

                Results from both queries are merged by score (descending),
                deduplicated by ACTUAL Qdrant point id (a point appearing in
                both pools counts once), and truncated to the top `k`
                overall. This is not a weaker approximation of a single
                combined query: any point that would belong in the true
                global top-k necessarily belongs to the top-k of whichever
                pool it is a member of (every point queried here is a
                member of at least one of "matches the visibility filter"
                or "is a deterministic reference candidate"), so merging
                the two pools' own top-k results and re-truncating recovers
                the same global top-k a single unfiltered query would.

                Stage 5C corrective pass #8 (Blocker 2): the merge is fully
                DETERMINISTIC, independent of which backend query happens
                to return a given equal-score candidate first. Duplicates
                (the same actual point id present in both pools) are
                collapsed to one entry — keeping the higher score if the
                two copies ever disagree (the same point queried with the
                same query vector is expected to always score identically
                regardless of which filter restricted the candidate set,
                so a differing score is not expected in practice, but this
                is the documented, deterministic tie-break if it ever
                happens). The merged set is then sorted by
                `(-score, str(actual_point_id))` — descending score first,
                then ascending canonical string point id as a tie-break for
                equal scores — never Python's stable-sort/list-order
                fallback (which would depend on backend arrival order) and
                never any mutable payload field.

        Returns:
            List of (document, score) tuples, nearest-first (Qdrant's
            query_points() already returns points ordered by descending
            similarity score for COSINE distance).
        """
        query_filter = self._visibility_filter(requesting_user_uuid)
        try:
            with self._lock:
                # Network call (embeds `query`) — happens before the Qdrant
                # call(s), same ordering as before.
                vector = self.embeddings.embed_query(query)
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True,
                )
                points = list(response.points)

                if reference_candidate_point_ids:
                    reference_response = self.client.query_points(
                        collection_name=self.collection_name,
                        query=vector,
                        query_filter=Filter(
                            must=[HasIdCondition(has_id=list(reference_candidate_point_ids))]
                        ),
                        limit=k,
                        with_payload=True,
                    )
                    merged_by_id = {str(point.id): point for point in points}
                    for point in reference_response.points:
                        pid = str(point.id)
                        existing = merged_by_id.get(pid)
                        if existing is None or point.score > existing.score:
                            merged_by_id[pid] = point
                    points = list(merged_by_id.values())
                    points.sort(key=lambda point: (-point.score, str(point.id)))
                    points = points[:k]

            results = [
                (self._document_from_payload(point.payload, point_id=str(point.id)), point.score)
                for point in points
            ]
            logger.debug("RAG similarity_search_with_score | query_len=%s, k=%s, results=%s", len(query), k, len(results))
            return results
        except Exception as e:
            # similarity_search_with_score embeds `query` via
            # OpenAIEmbeddings (network call) before searching Qdrant —
            # never log raw exception text.
            logger.error("RAG similarity_search_with_score failed | error_type=%s", type(e).__name__)
            raise

    def index_documents_directory(
        self,
        directory: Optional[Path] = None,
        force_reindex: bool = False,
        reference_filenames: Optional[Sequence[str]] = BUILTIN_REFERENCE_FILES,
    ) -> int:
        """
        Index reference documents from a directory.

        Each source file is reconciled via reconcile_document() (Stage
        2B-C Blocker 2): an unchanged file costs zero embedding/Qdrant
        calls, a file with only stale leftover points gets just those
        pruned (zero embedding calls), and only a genuinely new/changed
        file is actually re-embedded. This is what makes repeated startups
        idempotent AND convergent after a previous partial failure,
        instead of merely idempotent-when-nothing-ever-failed.

        Args:
            directory: Directory containing reference documents (default:
                config.DOCUMENTS_DIR)
            force_reindex: If True, clear the existing index first, then
                index every file unconditionally (skip-if-unchanged does
                not apply, since nothing exists yet to compare against).
            reference_filenames: The exact, explicit manifest of built-in
                reference filenames to enumerate under `directory` (Stage
                2B-C Blocker 5) — default config.BUILTIN_REFERENCE_FILES,
                which is what every real (production) caller uses. A
                missing manifest file fails loudly (see
                rag.loader.MissingReferenceDocumentError) rather than
                silently indexing fewer built-in documents. Pass `None` to
                fall back to an unconstrained extension-based scan of
                `directory` instead — this bypasses the manifest gate
                entirely and exists ONLY for tests exercising generic
                directory-scan/reindex mechanics against a synthetic
                corpus; production code must never do this.

        Returns:
            Number of NEW/CHANGED chunks actually (re)embedded this call
            (unchanged documents, and documents that only had stale extra
            points pruned, do not count).
        """
        if directory is None:
            directory = rag_constants.DOCUMENTS_DIR
        directory = Path(directory)
        try:
            with self._lock:
                if force_reindex:
                    logger.info("Clearing existing index")
                    self.clear_index()

                if reference_filenames is not None:
                    source_files = document_loader.list_builtin_reference_files(directory, reference_filenames)
                else:
                    source_files = document_loader.list_source_files(directory)
                resolved_root = directory.resolve()
                total_chunks = 0

                for file_path in source_files:
                    relative = file_path.resolve().relative_to(resolved_root).as_posix()
                    document_id = reference_document_id(relative)
                    status, chunk_count = self.reconcile_document(document_id, file_path)
                    if status == "unchanged":
                        logger.debug("RAG index: reference document unchanged, skipped")
                    elif status == "stale_pruned":
                        logger.info("RAG index: reference document had only stale extra points, pruned")
                    elif status == "emptied":
                        # Stage 5C corrective pass #6 (Blocker 2): the
                        # reference document now parses to zero chunks —
                        # its previously-indexed points were just removed.
                        # chunk_count is 0, so total_chunks is unaffected
                        # either way, but this gets its own explicit log
                        # line rather than falling into the plain
                        # "reindexed" case below.
                        logger.warning("RAG index: reference document now has zero chunks, removed previously-indexed points")
                    else:
                        total_chunks += chunk_count

            logger.info("RAG index_documents_directory | chunks=%s", total_chunks)
            return total_chunks
        except Exception as e:
            logger.error("RAG index_documents_directory failed | error_type=%s", type(e).__name__)
            raise

    def clear_index(self) -> None:
        """
        Clear the entire vector store — never by removing the persistent
        storage directory (that would race the open client's own storage
        lock/files). Under self._lock the whole time.

        Deliberately NOT `delete_collection()` + `create_collection()`:
        verified empirically (Stage 2B-B) against the resolved
        qdrant-client 1.19.0 local-mode backend that this sequence is
        buggy — `delete_collection()` removes the collection from the
        local registry (`get_collections()` no longer lists it), but does
        NOT release/clear its on-disk point storage (`storage.sqlite`
        under the collection's own directory). Recreating a collection
        with the SAME name then silently resurrects the "deleted" points
        instead of starting empty — confirmed both within the same client
        session and after closing and reopening a fresh client against
        the same path. A workaround of manually removing that on-disk
        directory was also tried and rejected: on Windows the file is
        still held open by the local backend even immediately after
        `close()`, raising `PermissionError` — exactly the kind of
        "shutil.rmtree while something still holds the storage lock"
        hazard this method must never risk.

        The correct, portable fix is to delete every point via a
        match-everything filter instead, leaving the collection (and its
        schema) intact throughout — a supported, documented Qdrant API
        that does not depend on the buggy delete+recreate sequence.
        """
        try:
            with self._lock:
                self.client.delete(collection_name=self.collection_name, points_selector=Filter())
            logger.info("RAG index cleared")
        except Exception as e:
            logger.error("RAG clear_index failed | error_type=%s", type(e).__name__)
            raise

    def count_verified_reference_points(self, expected_point_hashes: Dict[str, str]) -> int:
        """
        Count exactly how many of the EXPECTED canonical reference points
        in `expected_point_hashes` (see rag.loader.DocumentLoader.
        expected_reference_point_hashes() — the caller-supplied trust
        anchor) genuinely exist in this collection with content that
        independently verifies against it (Stage 5C corrective pass #3,
        Blocker 2).

        Deliberately NEVER a `scope="reference"` Qdrant filter (the
        corrective pass #2 defect Codex reproduced: a relabelled private
        point counted as reference for another user, and could double-
        count for its own real owner too) — `scope`, like `document_id`,
        is ordinary mutable Qdrant payload metadata, never a trust
        anchor. Instead this retrieves EXACTLY the expected point IDs
        (deterministic, derived independently of anything any point's own
        payload claims — see expected_reference_point_hashes()) and
        re-verifies each one's ACTUAL `text` payload against its expected
        content hash before counting it. A point that doesn't exist at
        its expected id, or exists but has been tampered with (content no
        longer hashes to the expected value), simply isn't counted —
        fail-closed, exactly like retrieval's _is_proven_reference().

        `expected_point_hashes` empty (e.g. a test collection with no
        reference manifest configured) returns 0 without any Qdrant call.

        Stage 5C corrective pass #5 (Blocker 4): counting now goes through
        `rag.identity.is_canonical_reference_point()` — THE single
        canonical-reference predicate shared with rag.query.
        _is_proven_reference() (the retrieval surface) — rather than a
        second, narrower check of its own. The previous implementation
        here verified only the actual stored `text` against the expected
        hash, never the point's own `document_id`/`chunk_index` payload
        fields; an independent audit reproduced a point with the correct
        actual id and byte-identical canonical text, but INCOMPLETE
        reference metadata (missing document_id/chunk_index), that
        retrieval correctly excluded (it requires those fields to
        reconstruct/cross-check the expected point id) while this method
        counted it anyway — the two surfaces disagreeing on the same
        candidate. `with_payload` now also fetches `document_id`/
        `chunk_index` so the shared predicate can evaluate the identical
        contract retrieval does.
        """
        if not expected_point_hashes:
            return 0
        with self._lock:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=list(expected_point_hashes.keys()),
                with_payload=["text", "document_id", "chunk_index"],
            )
        count = 0
        for record in records:
            payload = record.payload or {}
            if is_canonical_reference_point(
                actual_point_id=str(record.id),
                document_id=payload.get("document_id"),
                chunk_index=payload.get("chunk_index"),
                actual_text=payload.get("text", ""),
                trusted_reference_points=expected_point_hashes,
            ):
                count += 1
        return count

    def private_chunk_counts_by_document(
        self, *, requesting_user_uuid: str, exclude_point_ids: Optional[Set[str]] = None
    ) -> Dict[str, int]:
        """
        Raw Qdrant-level `{document_id: chunk_count}` for every point that
        is an ELIGIBLE non-reserved private candidate for
        `requesting_user_uuid` (Stage 5C corrective pass #2, Section 2;
        tightened corrective pass #9) — per-point eligibility is decided by
        `rag.identity.is_eligible_private_candidate()`, THE SAME predicate
        rag/query.py's `_validated_similarity_search()` applies to
        retrieval, so the two surfaces can never disagree about which raw
        Qdrant facts make a point an eligible private candidate. This is
        still intentionally a raw candidate list, not a fully validated
        count: the sole caller, rag.query.get_knowledge_base_stats(),
        additionally cross-checks every returned document_id against the
        canonical PostgreSQL documents catalog (ACTIVE status + matching
        owner) before counting any of it — Qdrant's own payload, even once
        internally self-consistent, is never sufficient on its own.

        Stage 5C corrective pass #9: an independent acceptance review
        reproduced retrieval and statistics classifying the same kind of
        non-reserved point differently when its Qdrant-side `scope`/
        `owner_user_uuid` payload was incomplete or inconsistent (e.g.
        `scope="reference"` with `owner_user_uuid` naming the requester, or
        missing) — this method used to filter ONLY on `owner_user_uuid`,
        never `scope`, so such a point could still reach the batched
        catalog check and be counted, even though it is not the kind of
        point `is_eligible_private_candidate()` (or a correctly-labelled
        genuine private upload) would ever consider private. The
        `owner_user_uuid` Qdrant filter below remains as a pure performance
        prefilter (an eligible candidate must always have `owner_user_uuid`
        exactly equal to `requesting_user_uuid`, so it can never exclude a
        true positive) — every scrolled record is still independently run
        through the identical eligibility predicate retrieval uses before
        being counted at all.

        `exclude_point_ids` (Stage 5C corrective pass #4, Blocker 2): any
        point whose ACTUAL Qdrant id is a member of this set is skipped
        entirely — never counted here, regardless of what its own
        `owner_user_uuid`/`document_id` payload claims. The caller passes
        the full set of EXPECTED canonical reference point ids (every key
        of rag.loader.DocumentLoader.expected_reference_point_hashes()),
        which makes reference/private classification structurally mutually
        exclusive by construction rather than merely "coincidentally
        disjoint given how honest data is normally written": a point
        sitting at a canonical-reference point-id slot can never be double-
        counted as private even if it also carries a forged
        `owner_user_uuid` — it is either proven reference (counted via
        count_verified_reference_points(), which is itself actual-point-id
        bound via Qdrant's own id-based retrieve()) or, having failed that
        proof, simply not counted at all (never silently reclassified as
        private).
        """
        exclude_point_ids = exclude_point_ids or set()
        counts: Dict[str, int] = {}
        offset = None
        owner_filter = Filter(must=[FieldCondition(key="owner_user_uuid", match=MatchValue(value=requesting_user_uuid))])
        with self._lock:
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=owner_filter,
                    limit=256,
                    offset=offset,
                    with_payload=["document_id", "scope", "owner_user_uuid"],
                    with_vectors=False,
                )
                for record in records:
                    if str(record.id) in exclude_point_ids:
                        continue
                    payload = record.payload or {}
                    doc_uuid = is_eligible_private_candidate(
                        scope=payload.get("scope"),
                        owner_user_uuid=payload.get("owner_user_uuid"),
                        requesting_user_uuid=requesting_user_uuid,
                        document_id=payload.get("document_id"),
                    )
                    if doc_uuid is None:
                        continue
                    document_id = payload.get("document_id")
                    counts[document_id] = counts.get(document_id, 0) + 1
                if offset is None:
                    break
        return counts

    def get_stats(self, *, requesting_user_uuid: str) -> dict:
        """
        Get statistics about the vector store, scoped to what
        `requesting_user_uuid` may actually see per Qdrant's OWN
        scope/owner_user_uuid payload (Stage 3A, migrated to UUID Stage
        5C). Required, keyword-only, no default — same rationale as
        similarity_search_with_score(); see _visibility_filter(), which
        raises ValueError for a missing/invalid id rather than silently
        counting everything.

        Stage 5C corrective pass #2/#3: this raw Qdrant-level count is
        deliberately NOT what the real /stats command uses any more —
        Qdrant's own payload is never sufficient proof of private
        ownership OR reference provenance on its own (see rag/query.py's
        _validated_similarity_search()'s docstring), so
        rag.query.get_knowledge_base_stats() instead combines
        count_verified_reference_points() (manifest/content-hash verified,
        never a `scope` filter) with a catalog-validated private count
        from private_chunk_counts_by_document(). This method remains a
        legitimate raw introspection primitive in its own right — every
        test that exercises pure Qdrant reconciliation/rebuild mechanics
        against synthetic, non-catalog-backed document ids (this class's
        own extensive existing test coverage) still calls it directly.

        Returns:
            Dictionary with statistics
        """
        # Validated/built before the try block below, same as
        # similarity_search_with_score() — an invalid requesting_user_uuid
        # is a caller bug (never a Qdrant/provider failure) and must raise
        # ValueError loudly, not be swallowed into the generic
        # {"error": ...} fallback below.
        count_filter = self._visibility_filter(requesting_user_uuid)
        try:
            with self._lock:
                count = self.client.count(
                    collection_name=self.collection_name, count_filter=count_filter, exact=True
                ).count

            # No absolute filesystem path here: this dict is displayed
            # verbatim to the user by handlers/start.py's /stats command,
            # and an absolute path can reveal the deployment's OS
            # username/directory layout.
            return {
                "total_documents": count,
                "status": "ok",
            }
        except Exception as e:
            # get_stats()'s "error" field is displayed verbatim to the user
            # by handlers/start.py's /stats command, so it must never carry
            # raw exception text (which could echo Qdrant/provider internals).
            logger.error("RAG get_stats failed | error_type=%s", type(e).__name__)
            return {"error": "Не удалось получить статистику базы знаний."}

    def close(self) -> None:
        """
        Explicitly release the local persistent Qdrant client and its
        storage-path lock. Needed for deterministic cleanup in tests (a
        second QdrantClient can't open the same local path while a prior
        one is still holding it) and for any future lifecycle handling —
        never relied on garbage collection alone.
        """
        try:
            with self._lock:
                if self.client is not None:
                    self.client.close()
                    self.client = None
            logger.info("RAG index closed")
        except Exception as e:
            logger.error("RAG index close failed | error_type=%s", type(e).__name__)
            raise


# Lazy global-singleton accessors (Stage 2B-D Blocker 4).
#
# Merely importing this module must never construct Qdrant state, make a
# provider call, or write to disk — `import rag.index` (and, transitively,
# `import rag.query`) is now completely inert. A real VectorIndex is
# created only the first time something EXPLICITLY calls
# get_vector_index() — a real RAG query, a real document upload, real
# startup indexing, or a test that specifically requests it.
_vector_index: Optional["VectorIndex"] = None
_vector_index_lock = threading.Lock()


def get_vector_index() -> "VectorIndex":
    """
    Return the shared production VectorIndex singleton, constructing it on
    first call. Safe to call repeatedly/concurrently — construction is
    guarded by a lock and only ever happens once; every call after the
    first just returns the same cached instance.
    """
    global _vector_index
    if _vector_index is None:
        with _vector_index_lock:
            if _vector_index is None:
                _vector_index = VectorIndex()
    return _vector_index


def close_vector_index() -> None:
    """
    Close and release the shared singleton, then reset it so a later
    get_vector_index() call constructs a fresh instance. Safe to call even
    if no singleton was ever constructed (e.g. a test session where nothing
    happened to need one).
    """
    global _vector_index
    with _vector_index_lock:
        if _vector_index is not None:
            _vector_index.close()
            _vector_index = None
