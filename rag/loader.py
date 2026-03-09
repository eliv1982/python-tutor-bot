"""
Document Loader for RAG.
Loads and processes documents from various formats.
"""

from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCUMENTS_DIR, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from utils.logging import logger


class DocumentLoader:
    """Loads and processes documents for RAG."""
    
    def __init__(self):
        """Initialize document loader."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_document(self, file_path: Path) -> List[Dict]:
        """
        Load a single document and split into chunks.
        
        Args:
            file_path: Path to document file
        
        Returns:
            List of document chunks with metadata
        """
        try:
            file_path = Path(file_path)
            
            # Select appropriate loader based on file extension
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ['.txt', '.md']:
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Load document
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add source metadata
            for chunk in chunks:
                chunk.metadata['source'] = file_path.name
                chunk.metadata['file_path'] = str(file_path)
            
            logger.info("RAG loader load_document | file=%s, chunks=%s", file_path.name, len(chunks))
            return chunks
        except Exception as e:
            logger.error("RAG loader load_document failed | file=%s, error=%s", file_path, e, exc_info=True)
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
            
            # Supported file extensions
            supported_extensions = ['.pdf', '.txt', '.md', '.docx', '.doc']
            
            # Find all supported files
            for file_path in directory.rglob('*'):
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        chunks = self.load_document(file_path)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning("RAG loader: skipping file | file=%s, error=%s", file_path.name, e)
            logger.info("RAG loader load_directory | directory=%s, total_chunks=%s", directory, len(all_chunks))
            return all_chunks
        except Exception as e:
            logger.error("RAG loader load_directory failed | directory=%s, error=%s", directory, e, exc_info=True)
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
            logger.error(f"Error loading text: {e}")
            raise


# Global loader instance
document_loader = DocumentLoader()

