"""
Document Loader for RAG.
Loads and processes documents from various formats.
"""

import io
from pathlib import Path
from typing import List, Dict, Optional, Sequence

import docx2txt
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.document_loaders import Blob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag.constants as rag_constants
from rag.constants import (
    BUILTIN_REFERENCE_FILES,
    DOCUMENTS_DIR,
    MANAGED_UPLOADS_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)
from rag.identity import point_id as make_point_id, reference_document_id, sha256_hex
from utils.logging import logger

# SUPPORTED_EXTENSIONS is re-exported here (from rag/constants.py, the
# single source of truth) since several modules/tests do
# `from rag.loader import SUPPORTED_EXTENSIONS`.


class MissingReferenceDocumentError(FileNotFoundError):
    """
    Raised by list_builtin_reference_files() when one or more manifest
    filenames (config.BUILTIN_REFERENCE_FILES) are absent from the
    reference-document directory. Deliberately all-or-nothing (Stage 2B-C
    Blocker 5): a missing manifest file must fail loudly, never silently
    reduce built-in product knowledge to whichever files happen to exist.
    The message names only the fixed, non-secret manifest filenames — never
    an absolute path.
    """


class DocumentLoader:
    """Loads and processes documents for RAG."""

    def __init__(self):
        """Initialize document loader."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            length_function=len,
        )

    def load_document(
        self,
        file_path: Path,
        display_name: Optional[str] = None,
        document_id: Optional[str] = None,
        content_sha256: Optional[str] = None,
        stored_name: Optional[str] = None,
        owner_user_uuid: Optional[str] = None,
    ) -> List[Dict]:
        """
        Load a single document and split into chunks.

        Args:
            file_path: Path to document file on disk (the physical,
                application-generated storage path)
            display_name: Human-readable name to record as source metadata
                instead of `file_path.name`. Used when the physical filename
                is an opaque storage identity (e.g. a UUID) that must not
                leak into user-facing source attribution.
            document_id: Stable logical-document identity (see
                rag/identity.py). Attached to every chunk's metadata so
                VectorIndex can group/replace this document's points.
                Chunks with no document_id cannot be indexed via
                VectorIndex.add_documents() (Stage 2B requires it).
            content_sha256: Local content fingerprint of the whole source
                file, attached to every chunk so VectorIndex can detect an
                unchanged document without re-embedding it (Stage 2B
                Section N).
            stored_name: Physical storage filename (e.g. the opaque UUID
                name of a managed upload) — recorded in Qdrant payloads for
                managed uploads only, never an absolute path.
            owner_user_uuid: Canonical internal user UUID string (Stage 5C)
                of the managed upload's owner. `None` for reference
                documents (and for any caller with no real owner to give) —
                VectorIndex derives `scope="reference"` from that absence.
                Never guessed/inferred here; the caller (VectorIndex.
                reconcile_document()) is the sole source of this value.

        Returns:
            List of document chunks with metadata
        """
        try:
            file_path = Path(file_path)
            source_name = display_name or file_path.name

            # Select appropriate loader based on file extension
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ['.txt', '.md']:
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_path.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Load document
            documents = loader.load()

            chunks = self._chunk_and_tag(
                documents,
                source_name=source_name,
                file_path=str(file_path),
                document_id=document_id,
                content_sha256=content_sha256,
                stored_name=stored_name,
                owner_user_uuid=owner_user_uuid,
            )

            # source_name can be a user-controlled Telegram display filename
            # (see document_upload.py's display_name= usage) — log only the
            # extension and chunk count, never the name itself.
            logger.info("RAG loader load_document | extension=%s, chunks=%s", file_path.suffix.lower(), len(chunks))
            return chunks
        except Exception as e:
            # Parser errors (PyPDFLoader/TextLoader/Docx2txtLoader) can echo
            # fragments of the document's raw bytes/content in their message
            # — never log the raw exception text or a traceback here.
            logger.error("RAG loader load_document failed | extension=%s, error_type=%s", file_path.suffix.lower(), type(e).__name__)
            raise

    def load_document_bytes(
        self,
        source_bytes: bytes,
        suffix: str,
        display_name: Optional[str] = None,
        document_id: Optional[str] = None,
        content_sha256: Optional[str] = None,
        stored_name: Optional[str] = None,
        owner_user_uuid: Optional[str] = None,
    ) -> List[Dict]:
        """
        Parse `source_bytes` directly in memory and split into chunks —
        never touching any filesystem pathname (Stage 2B-F Blocker 1).

        A caller (VectorIndex.reconcile_document()) uses this exactly when
        it already hashed/verified these precise bytes for a reconciliation
        decision: the parsed/indexed content must provably be those same
        bytes, not a reopen of some pathname that could have been swapped
        after verification. PDF parsing runs against an in-memory
        `langchain_core.document_loaders.Blob` (never `Blob.from_path()`);
        DOCX parsing runs `docx2txt.process()` directly against a
        `io.BytesIO` stream (`docx2txt.process()` accepts any file-like
        object, since it only ever hands its argument to
        `zipfile.ZipFile()`); TXT/MD is UTF-8 decoded directly. None of the
        three ever writes `source_bytes` to disk, so there is no temporary
        pathname for a concurrent write to swap between verification and
        parsing, and no OS-level resource is left to clean up afterward —
        every object involved (`BytesIO`, the parsed `Document` list) is
        released by ordinary garbage collection on every path (success,
        an ordinary parser exception, or cancellation), the same guarantee
        `Blob.as_bytes_io()` and `docx2txt.process()` already provide for
        their own in-memory buffers.

        Args:
            source_bytes: The exact, already-verified bytes to parse.
            suffix: File extension (e.g. ".pdf", leading dot, any case) —
                selects the format the same way load_document()'s
                `file_path.suffix` does.
            display_name / document_id / content_sha256 / stored_name /
                owner_user_uuid: mirror load_document()'s own parameters —
                see there.

        Returns:
            List of document chunks with metadata
        """
        suffix = suffix.lower()
        source_name = display_name or f"document{suffix}"
        try:
            if suffix == '.pdf':
                blob = Blob.from_data(source_bytes, path=source_name)
                documents = list(PyPDFParser().lazy_parse(blob))
            elif suffix in ('.txt', '.md'):
                documents = [Document(page_content=source_bytes.decode('utf-8'), metadata={})]
            elif suffix == '.docx':
                text = docx2txt.process(io.BytesIO(source_bytes))
                documents = [Document(page_content=text, metadata={})]
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            chunks = self._chunk_and_tag(
                documents,
                source_name=source_name,
                file_path=None,
                document_id=document_id,
                content_sha256=content_sha256,
                stored_name=stored_name,
                owner_user_uuid=owner_user_uuid,
            )

            logger.info("RAG loader load_document_bytes | extension=%s, chunks=%s", suffix, len(chunks))
            return chunks
        except Exception as e:
            # Same rationale as load_document(): parser errors can echo
            # fragments of the document's raw content — never log the raw
            # exception text or a traceback here.
            logger.error("RAG loader load_document_bytes failed | extension=%s, error_type=%s", suffix, type(e).__name__)
            raise

    def _chunk_and_tag(
        self,
        documents: List[Document],
        *,
        source_name: str,
        file_path: Optional[str],
        document_id: Optional[str],
        content_sha256: Optional[str],
        stored_name: Optional[str],
        owner_user_uuid: Optional[str] = None,
    ) -> List[Document]:
        """
        Shared chunk-splitting + metadata tagging for both load_document()
        and load_document_bytes(). `file_path` (internal use only — nothing
        downstream may persist it into a Qdrant payload, see
        VectorIndex._safe_payload()) is omitted entirely when parsing was
        done from in-memory bytes (load_document_bytes() passes None) —
        there is no on-disk pathname to record in that case.

        `owner_user_uuid` (Stage 3A, migrated to canonical UUID Stage 5C):
        tagged onto every chunk's metadata only when given, exactly like
        document_id/content_sha256/stored_name above —
        VectorIndex._safe_payload() reads it from here to decide a chunk's
        Qdrant `scope`. Omitted (never `None`-valued) for reference
        documents, which have no owner.
        """
        chunks = self.text_splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata['source'] = source_name
            if file_path is not None:
                chunk.metadata['file_path'] = file_path
            chunk.metadata['chunk_index'] = idx
            if document_id is not None:
                chunk.metadata['document_id'] = document_id
            if content_sha256 is not None:
                chunk.metadata['content_sha256'] = content_sha256
            if stored_name is not None:
                chunk.metadata['stored_name'] = stored_name
            if owner_user_uuid is not None:
                chunk.metadata['owner_user_uuid'] = owner_user_uuid
        return chunks

    def list_source_files(self, directory: Path = DOCUMENTS_DIR) -> List[Path]:
        """
        Enumerate loader-supported source files under `directory`, excluding
        application-managed uploads (see MANAGED_UPLOADS_DIR — those are
        indexed individually at upload time, never via a directory scan).

        Sorted for deterministic iteration order (stable logging/tests),
        not because ordering has any correctness meaning here.
        """
        directory = Path(directory)
        managed_uploads_dir = MANAGED_UPLOADS_DIR.resolve()
        files = []
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if file_path.resolve().is_relative_to(managed_uploads_dir):
                # Application-managed uploads (handlers/document_upload.py)
                # are indexed once, at upload time, with their original
                # filename as source metadata. Rescanning them here would
                # re-index them a second time under their opaque UUID
                # filename. See MANAGED_UPLOADS_DIR in config.py.
                continue
            files.append(file_path)
        return sorted(files)

    def list_builtin_reference_files(
        self,
        directory: Path,
        manifest: Sequence[str] = BUILTIN_REFERENCE_FILES,
    ) -> List[Path]:
        """
        Enumerate EXACTLY the built-in reference documents named in
        `manifest` (default: config.BUILTIN_REFERENCE_FILES) under
        `directory` — never an unconstrained extension-based scan (Stage
        2B-C Blocker 5). A stray legacy `.txt` file or an arbitrary extra
        `.md`/`.pdf`/`.docx` sitting in `directory` is silently ignored: it
        is simply not one of the named files, so it is never returned.

        Raises MissingReferenceDocumentError (all-or-nothing — never a
        partial list) if any manifest filename is not a regular file under
        `directory`. The exception message names only the fixed manifest
        filenames themselves (public, checked into version control), never
        an absolute path.

        Returns Paths in manifest order (deterministic).
        """
        directory = Path(directory)
        missing = []
        found = []
        for filename in manifest:
            candidate = directory / filename
            if candidate.is_file():
                found.append(candidate)
            else:
                missing.append(filename)
        if missing:
            logger.error("RAG loader: missing built-in reference document(s) | missing=%s", sorted(missing))
            raise MissingReferenceDocumentError(
                f"missing built-in reference document(s): {sorted(missing)}"
            )
        return found

    def expected_reference_point_hashes(
        self,
        directory: Optional[Path] = None,
        manifest: Sequence[str] = BUILTIN_REFERENCE_FILES,
    ) -> Dict[str, str]:
        """
        The authoritative, independently re-derivable trust anchor for
        canonical reference provenance (Stage 5C corrective pass #3,
        Blocker 1): `{expected_qdrant_point_id: expected_content_sha256}`
        for every chunk the CURRENT version-controlled built-in reference
        corpus genuinely produces.

        Built by running the EXACT SAME pipeline real reference indexing
        uses — list_builtin_reference_files() to enumerate the manifest,
        load_document() (this class's own chunker) to split each file into
        chunks, rag.identity.reference_document_id() to derive each file's
        logical document id from its path relative to `directory` (byte-
        for-byte the same computation VectorIndex.index_documents_
        directory() performs), and rag.identity.point_id() to derive each
        chunk's deterministic Qdrant point id — never a second, subtly
        different chunking/identity implementation.

        Why this closes the Stage 5C corrective-pass #2 gap: that pass
        classified a Qdrant result as reference merely because its
        `document_id` PAYLOAD FIELD matched a canonical id — but
        `document_id` is ordinary, mutable Qdrant payload metadata an
        adversarial/corrupt point can simply copy. This manifest is keyed
        by point id (independent of anything a point's own payload
        claims) and its value is the EXPECTED CONTENT HASH — so a caller
        (rag/query.py, rag/index.py) must independently verify the
        ACTUAL, ACTUALLY-RETURNED content hashes to this exact value
        before ever trusting a result as reference. Copying `scope`,
        `document_id`, source name, chunk index, or even guessing the
        correct point id is never sufficient on its own: the returned
        content itself must be byte-identical to the real canonical chunk.

        Recomputed fresh on every call (no caching) — the built-in corpus
        is a handful of small, version-controlled files; correctness (an
        always-current trust anchor, even immediately after an operator
        edits data/documents/) outweighs the trivial repeated-parse cost.
        Raises MissingReferenceDocumentError (same as
        list_builtin_reference_files()) if the manifest itself is
        incomplete — never silently produces an incomplete trust anchor;
        callers (rag/query.py) that need "no proven references available"
        to degrade gracefully rather than fail retrieval/stats outright
        catch this themselves.

        `directory` defaults to None, read as `rag_constants.DOCUMENTS_DIR`
        FRESH inside this call (module-attribute access, never a bound
        top-level default) — mirrors VectorIndex.index_documents_
        directory()'s own `if directory is None: directory = rag_constants.
        DOCUMENTS_DIR` pattern (see its comment for the full rationale):
        a bound default (`directory: Path = DOCUMENTS_DIR`) would capture
        whatever `rag.constants.DOCUMENTS_DIR` happened to be at THIS
        MODULE's own first-import time and never see a later redirect
        (e.g. tests/conftest.py's pytest_configure(), which redirects it
        to a disposable temp path for the whole session) — this function
        must always resolve against whatever directory is CURRENT at call
        time, exactly like every other reference-scan entry point in this
        codebase.
        """
        if directory is None:
            directory = rag_constants.DOCUMENTS_DIR
        directory = Path(directory)
        resolved_root = directory.resolve()
        expected: Dict[str, str] = {}
        for file_path in self.list_builtin_reference_files(directory, manifest):
            relative = file_path.resolve().relative_to(resolved_root).as_posix()
            document_id = reference_document_id(relative)
            for chunk in self.load_document(file_path):
                pid = make_point_id(document_id, chunk.metadata["chunk_index"])
                expected[pid] = sha256_hex(chunk.page_content.encode("utf-8"))
        return expected

    def load_directory(self, directory: Path = DOCUMENTS_DIR) -> List[Dict]:
        """
        Load all documents from a directory (no document_id/content_sha256
        attached — this is a plain bulk-load utility; VectorIndex's own
        index_documents_directory() uses list_source_files() +
        load_document() directly instead, so it can attach per-file
        identity/fingerprint metadata and skip unchanged files).

        Args:
            directory: Path to directory containing documents

        Returns:
            List of all document chunks
        """
        try:
            all_chunks = []
            for file_path in self.list_source_files(directory):
                try:
                    chunks = self.load_document(file_path)
                    all_chunks.extend(chunks)
                except Exception:
                    # load_document() already logged the sanitized failure.
                    logger.warning("RAG loader: skipping file | extension=%s", file_path.suffix.lower())
            # No absolute directory path in logs (deployment layout/username).
            logger.info("RAG loader load_directory | total_chunks=%s", len(all_chunks))
            return all_chunks
        except Exception as e:
            logger.error("RAG loader load_directory failed | error_type=%s", type(e).__name__)
            raise

    def load_text(self, text: str, source: str = "manual_input") -> List[Dict]:
        """
        Load text directly and split into chunks.

        Args:
            text: Text content
            source: Source identifier

        Returns:
            List of text chunks
        """
        try:
            # Create document
            document = Document(
                page_content=text,
                metadata={"source": source}
            )

            # Split into chunks
            chunks = self.text_splitter.split_documents([document])

            logger.info(f"Created {len(chunks)} chunks from text input")
            return chunks

        except Exception as e:
            logger.error("RAG loader load_text failed | error_type=%s", type(e).__name__)
            raise


# Global loader instance
document_loader = DocumentLoader()
