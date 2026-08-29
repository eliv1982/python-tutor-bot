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

import openai
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

import rag.constants as rag_constants
from rag.constants import BUILTIN_REFERENCE_FILES
from rag.identity import point_id as make_point_id, reference_document_id, sha256_hex
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
_SAFE_PAYLOAD_FIELDS = ("source", "document_id", "chunk_index", "content_sha256", "stored_name", "page")


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
            self.client = QdrantClient(
                path=str(self.persist_directory),
                force_disable_check_same_thread=True,
            )
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
        loader's internal `file_path` (an absolute path) never does."""
        meta = document.metadata
        payload = {"text": document.page_content}
        for field in _SAFE_PAYLOAD_FIELDS:
            if field in meta and meta[field] is not None:
                payload[field] = meta[field]
        return payload

    @staticmethod
    def _document_from_payload(payload: Optional[dict]) -> Document:
        payload = payload or {}
        text = payload.get("text", "")
        metadata = {k: v for k, v in payload.items() if k in _SAFE_PAYLOAD_FIELDS}
        return Document(page_content=text, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal Qdrant helpers — every caller already holds self._lock
    # ------------------------------------------------------------------

    def _existing_points_detail(self, document_id: str) -> Dict[str, Optional[str]]:
        """Local-only lookup: every existing point id for `document_id`,
        mapped to its stored content_sha256 payload value (or None if that
        point somehow has no such field). Never makes a provider call
        itself (Stage 2B Section N) — this is what
        reconcile_document() below uses to classify EXACT CURRENT / EXTRA
        STALE POINTS ONLY / MISSING-OUTDATED without embedding anything."""
        detail: Dict[str, Optional[str]] = {}
        offset = None
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=doc_filter,
                limit=256,
                offset=offset,
                with_payload=["content_sha256"],
                with_vectors=False,
            )
            for record in records:
                detail[str(record.id)] = (record.payload or {}).get("content_sha256")
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

        Classifies into exactly one of three outcomes:
          - "unchanged": every expected point already exists with the
            current content_sha256, and no extra stale points remain.
            Zero embedding, upsert, or delete calls.
          - "stale_pruned": every expected point already exists with the
            current content_sha256, but extra (stale) points also remain
            — e.g. left over from a previous stale-delete failure. Only
            those extras are deleted; zero embedding calls (this is what
            makes a deterministic retry converge for free).
          - "reindexed": one or more expected points are missing or
            outdated — full safe replacement via _replace_document_points()
            (embed the complete new set, upsert, delete stale extras only
            after a successful upsert).

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
                )
            else:
                chunks = document_loader.load_document(
                    file_path,
                    display_name=display_name,
                    document_id=document_id,
                    content_sha256=content_sha256,
                    stored_name=stored_name,
                )

            if not chunks:
                # Mirrors the historical index_documents_directory()
                # behavior for an empty source: skip silently, leave
                # whatever (if anything) is already indexed untouched.
                return ("unchanged", 0)

            expected_ids = {make_point_id(document_id, c.metadata["chunk_index"]) for c in chunks}
            existing_detail = self._existing_points_detail(document_id)
            existing_ids = set(existing_detail)

            expected_all_current = expected_ids <= existing_ids and all(
                existing_detail.get(pid) == content_sha256 for pid in expected_ids
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

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant document chunks
        """
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples, nearest-first (Qdrant's
            query_points() already returns points ordered by descending
            similarity score for COSINE distance).
        """
        try:
            with self._lock:
                # Network call (embeds `query`) — happens before the Qdrant
                # call, same ordering as before.
                vector = self.embeddings.embed_query(query)
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=k,
                    with_payload=True,
                )
            results = [(self._document_from_payload(point.payload), point.score) for point in response.points]
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

    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        try:
            with self._lock:
                count = self.client.count(collection_name=self.collection_name, exact=True).count

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
