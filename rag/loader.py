"""
Document Loader for RAG.
Loads and processes documents from various formats.
"""

from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCUMENTS_DIR, MANAGED_UPLOADS_DIR, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from utils.logging import logger


# Formats this loader can actually parse end-to-end. Deliberately excludes
# legacy `.doc` (Docx2txtLoader only understands the modern .docx zip
# format and cannot read it) — this is the single source of truth for what
# both Telegram uploads and the directory scan below may accept.
SUPPORTED_EXTENSIONS = frozenset({'.pdf', '.txt', '.md', '.docx'})


class DocumentLoader:
    """Loads and processes documents for RAG."""
    
    def __init__(self):
        """Initialize document loader."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_document(self, file_path: Path, display_name: Optional[str] = None) -> List[Dict]:
        """
        Load a single document and split into chunks.

        Args:
            file_path: Path to document file on disk (the physical,
                application-generated storage path)
            display_name: Human-readable name to record as source metadata
                instead of `file_path.name`. Used when the physical filename
                is an opaque storage identity (e.g. a UUID) that must not
                leak into user-facing source attribution.

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

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Add source metadata
            for chunk in chunks:
                chunk.metadata['source'] = source_name
                chunk.metadata['file_path'] = str(file_path)

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
    
    def load_directory(self, directory: Path = DOCUMENTS_DIR) -> List[Dict]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to directory containing documents
        
        Returns:
            List of all document chunks
        """
        try:
            directory = Path(directory)
            all_chunks = []
            managed_uploads_dir = MANAGED_UPLOADS_DIR.resolve()

            # Find all supported files
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
            from langchain.schema import Document
            
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

