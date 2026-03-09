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

from config import DATA_DIR, OPENAI_API_KEY, DOCUMENTS_DIR
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
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY
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
            logger.info("RAG index: loaded existing vector store | path=%s", self.persist_directory)
        except Exception as e:
            logger.warning("RAG index: could not load vectorstore, creating new | error=%s", e)
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
            logger.error("RAG index add_documents failed | error=%s", e, exc_info=True)
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
            logger.error("RAG similarity_search failed | error=%s", e, exc_info=True)
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
            logger.error("RAG similarity_search_with_score failed | error=%s", e, exc_info=True)
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
            
            logger.info("RAG index_documents_directory | directory=%s, chunks=%s", directory, len(documents))
            return len(documents)
        except Exception as e:
            logger.error("RAG index_documents_directory failed | error=%s", e, exc_info=True)
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
            logger.error("RAG clear_index failed | error=%s", e, exc_info=True)
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
            
            return {
                "total_documents": count,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error("RAG get_stats failed | error=%s", e)
            return {"error": str(e)}


# Global index instance
vector_index = VectorIndex()

