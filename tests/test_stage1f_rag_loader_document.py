"""
Stage 1F-B regression test.

Stage 1F-A found that rag/loader.py's DocumentLoader.load_text() had a
deferred `from langchain.schema import Document` that no longer resolves
against the installed LangChain generation (`ModuleNotFoundError: No module
named 'langchain.schema'`). It stayed latent purely because nothing in the
existing test suite ever called load_text() - only load_document() and
load_directory() were exercised.

This test calls load_text() directly so the fixed import
(`from langchain_core.documents import Document`, now module-level in
rag/loader.py) is actually reached on every run, and a future regression
back to the obsolete import path fails the suite instead of passing
silently. Purely local text splitting - no network, no OpenAI, no Chroma.
"""

from langchain_core.documents import Document

from rag.loader import document_loader


def test_load_text_reaches_document_construction_and_returns_chunks():
    chunks = document_loader.load_text("Hello world, this is a short RAG test string.", source="unit-test")

    assert len(chunks) == 1
    assert isinstance(chunks[0], Document)
    assert chunks[0].metadata["source"] == "unit-test"
    assert "Hello world" in chunks[0].page_content


def test_load_text_default_source_is_manual_input():
    chunks = document_loader.load_text("Another short test string.")

    assert chunks[0].metadata["source"] == "manual_input"
