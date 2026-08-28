"""
Vector Index for RAG.
Creates and manages embeddings using ChromaDB.
"""

from typing import List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import DATA_DIR, OPENAI_API_KEY, OFFICIAL_OPENAI_BASE_URL, DOCUMENTS_DIR
from utils.logging import logger
from rag.loader import document_loader


class VectorIndex:
    """Manages vector embeddings and similarity search."""
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """
        Initialize vector index.
        
        Args:
            persist_directory: Directory to persist embeddings
        """
        if persist_directory is None:
            persist_directory = DATA_DIR / "chroma_db"
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        # base_url is pinned explicitly: langchain-openai defaults it from
        # the OPENAI_API_BASE env var, and would otherwise pass base_url=
        # None down to the openai SDK, which itself falls back to
        # OPENAI_BASE_URL. Passing it here overrides both.
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            base_url=OFFICIAL_OPENAI_BASE_URL
        )
        
        # Initialize or load vector store
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one."""
        try:
            # Try to load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            # persist_directory is an absolute filesystem path (can reveal
            # the deployment's OS username/layout) — no need to log it.
            logger.info("RAG index: loaded existing vector store")
        except Exception as e:
            logger.warning("RAG index: could not load vectorstore, creating new | error_type=%s", type(e).__name__)
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            logger.info("RAG index: created new vector store")
    
    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document chunks
        """
        try:
            if not documents:
                logger.warning("RAG index add_documents: empty list")
                return
            self.vectorstore.add_documents(documents)
            logger.info("RAG index add_documents | count=%s", len(documents))
        except Exception as e:
            # This call embeds documents via OpenAIEmbeddings (a network call
            # to OpenAI) before writing to Chroma, so the exception may be a
            # provider error — never log its raw text or a traceback.
            logger.error("RAG index add_documents failed | error_type=%s", type(e).__name__)
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 3
    ) -> List:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of relevant document chunks
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.debug("RAG similarity_search | query_len=%s, k=%s, results=%s", len(query), k, len(results))
            return results
        except Exception as e:
            # similarity_search embeds `query` via OpenAIEmbeddings (network
            # call) before searching Chroma — never log raw exception text.
            logger.error("RAG similarity_search failed | error_type=%s", type(e).__name__)
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.debug("RAG similarity_search_with_score | k=%s, results=%s", k, len(results))
            return results
        except Exception as e:
            logger.error("RAG similarity_search_with_score failed | error_type=%s", type(e).__name__)
            raise
    
    def index_documents_directory(
        self,
        directory: Path = DOCUMENTS_DIR,
        force_reindex: bool = False
    ) -> int:
        """
        Index all documents from a directory.
        
        Args:
            directory: Directory containing documents
            force_reindex: If True, clear existing index first
        
        Returns:
            Number of documents indexed
        """
        try:
            # Clear existing index if requested
            if force_reindex:
                logger.info("Clearing existing index")
                self.clear_index()
            
            # Load documents
            documents = document_loader.load_directory(directory)
            
            if not documents:
                logger.warning("No documents found to index")
                return 0
            
            # Add to vector store
            self.add_documents(documents)
            
            logger.info("RAG index_documents_directory | chunks=%s", len(documents))
            return len(documents)
        except Exception as e:
            logger.error("RAG index_documents_directory failed | error_type=%s", type(e).__name__)
            raise
    
    def clear_index(self):
        """Clear the entire vector store."""
        try:
            # Delete and recreate
            import shutil
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
            
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._load_or_create_vectorstore()
            
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
            # ChromaDB collection stats
            collection = self.vectorstore._collection
            count = collection.count()

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
            # raw exception text (which could echo Chroma/provider internals).
            logger.error("RAG get_stats failed | error_type=%s", type(e).__name__)
            return {"error": "Не удалось получить статистику базы знаний."}


# Global index instance
vector_index = VectorIndex()

