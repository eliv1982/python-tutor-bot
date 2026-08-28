"""
RAG Query Handler.
Handles queries against the knowledge base with context-aware responses.
"""

from typing import List, Dict, Optional

from rag.index import vector_index
from services.openai_client import openai_client
from utils.logging import logger
from config import RAG_TOP_K


async def query_knowledge_base(
    query: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Query the knowledge base and generate response.
    
    Args:
        query: User's query
        conversation_history: Previous conversation messages
    
    Returns:
        Generated response based on retrieved context
    """
    try:
        logger.info("RAG query_knowledge_base | query_len=%s, top_k=%s", len(query), RAG_TOP_K)
        results = vector_index.similarity_search_with_score(query, k=RAG_TOP_K)
        logger.debug("RAG similarity_search | results_count=%s", len(results))
        if not results:
            logger.warning("RAG: no results, using fallback")
            return await _fallback_response(query, conversation_history)
        
        # Prepare context from retrieved documents
        context = _prepare_context(results)
        
        response = await _generate_rag_response(
            query=query,
            context=context,
            conversation_history=conversation_history
        )
        # Добавляем ссылки на источники (user-facing attribution — sent to
        # the same user who owns/uploaded these documents, not a log).
        sources = list({doc.metadata.get("source", "?") for doc, _ in results})
        sources_str = ", ".join(sources)
        response = response.rstrip() + "\n\nИсточник(и): " + sources_str
        # Source filenames can be user-controlled/confidential (Stage 1B
        # display_name) — log only a count, never the names themselves.
        logger.info("RAG query done | response_len=%s, source_count=%s", len(response), len(sources))
        return response
    except Exception as e:
        # Wraps Chroma similarity search (embeddings network call) and the
        # OpenAI chat completion — never log raw exception text.
        logger.error("RAG query_knowledge_base failed | error_type=%s", type(e).__name__)
        # Fallback to regular GPT response
        return await _fallback_response(query, conversation_history)


def _prepare_context(results: List[tuple]) -> str:
    """
    Prepare context from search results.
    
    Args:
        results: List of (document, score) tuples
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        
        context_parts.append(
            f"[Источник {i}: {source}]\n{content}\n"
        )
    
    return "\n".join(context_parts)


async def _generate_rag_response(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Generate response using RAG context.
    
    Args:
        query: User's query
        context: Retrieved context from knowledge base
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    system_prompt = """Ты — персональный тьютор по Python с доступом к базе знаний.

ПРАВИЛА:
1. Отвечай на основе предоставленного контекста.
2. Если в контексте есть ответ — используй его.
3. Если ответа нет — честно скажи и ответь из общих знаний по Python.
4. Отвечай на русском, чётко и по делу. Не используй разметку markdown — только обычный текст. Примеры кода пиши с отступом, без звёздочек и обратных кавычек.

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

Ответь на вопрос пользователя, опираясь на контекст выше."""
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(context=context)
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        # Limit history to avoid token limits
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        messages.extend(recent_history)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Generate response
    response = await openai_client.generate_text_response(messages)
    
    return response


async def _fallback_response(
    query: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Fallback to regular GPT response when RAG fails.
    
    Args:
        query: User's query
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    logger.debug("RAG fallback_response (no context)")
    
    system_message = {
        "role": "system",
        "content": """Ты — личный тьютор по Python. База знаний пуста или не содержит ответа. Ответь на основе общих знаний и предупреди, что это не из базы знаний. Не используй markdown — только обычный текст."""
    }
    
    messages = [system_message]
    
    if conversation_history:
        messages.extend(conversation_history[-6:])
    
    messages.append({
        "role": "user",
        "content": query
    })
    
    response = await openai_client.generate_text_response(messages)
    
    return f"⚠️ База знаний не содержит информации по этому вопросу.\n\n{response}"


async def add_document_to_knowledge_base(file_path: str) -> dict:
    """
    Add a document to the knowledge base.
    
    Args:
        file_path: Path to document file
    
    Returns:
        Dictionary with status and details
    """
    try:
        from pathlib import Path
        from rag.loader import document_loader
        
        # Load document
        file_path = Path(file_path)
        documents = document_loader.load_document(file_path)
        
        # Add to index
        vector_index.add_documents(documents)
        
        # file_path.name is caller-supplied and not guaranteed non-sensitive
        # (this helper is currently unused, but future callers could pass a
        # user-controlled path) — log only the chunk count.
        logger.info("RAG add_document | chunks=%s", len(documents))
        
        return {
            "success": True,
            "file": file_path.name,
            "chunks": len(documents),
            "message": f"Документ {file_path.name} успешно добавлен ({len(documents)} фрагментов)"
        }
        
    except Exception as e:
        logger.error("RAG add_document failed | error_type=%s", type(e).__name__)
        return {
            "success": False,
            "error": type(e).__name__,
            "message": "Ошибка при добавлении документа."
        }


def get_knowledge_base_stats() -> dict:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Dictionary with statistics
    """
    return vector_index.get_stats()

