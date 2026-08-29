"""
Stage 1A.2 regression test: the RAG embeddings client must not be
redirectable to ProxyAPI (or any other endpoint) via ambient environment
variables.

`langchain_openai.OpenAIEmbeddings` resolves its base URL from the
`OPENAI_API_BASE` environment variable when no explicit value is passed,
and — if that field ends up None — the underlying openai SDK client falls
back further to `OPENAI_BASE_URL`. `rag/index.py` now passes
`base_url=config.OFFICIAL_OPENAI_BASE_URL` explicitly, which must take
precedence over both.

This test is isolated in its own module because importing `rag.index`
pulls in the full langchain/qdrant-client stack (slow). `tests/conftest.py`
redirects `rag.index`'s module-level `VectorIndex()` singleton to a temp
directory before this module (or anything else) can import it for the
first time, so that import never touches the real, gitignored
`data/qdrant`. The regression check below additionally constructs its
own separate `VectorIndex` against a per-test `tmp_path`, and performs no
embedding calls (no network) — only the client's resolved base_url is
inspected.
"""

import config


def test_rag_embeddings_client_ignores_ambient_openai_api_base_and_base_url(monkeypatch, tmp_path):
    # Both known ambient escape hatches, pointed at fake proxies.
    monkeypatch.setenv("OPENAI_API_BASE", "https://fake-proxy-api-base.example.invalid/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fake-proxy-base-url.example.invalid/v1")

    from rag.index import VectorIndex

    index = VectorIndex(persist_directory=tmp_path)

    assert index.embeddings.openai_api_base == config.OFFICIAL_OPENAI_BASE_URL

    resolved_sync_base_url = str(index.embeddings.client._client.base_url).rstrip("/")
    assert resolved_sync_base_url == config.OFFICIAL_OPENAI_BASE_URL
    assert "fake-proxy" not in resolved_sync_base_url

    if index.embeddings.async_client is not None:
        resolved_async_base_url = str(index.embeddings.async_client._client.base_url).rstrip("/")
        assert resolved_async_base_url == config.OFFICIAL_OPENAI_BASE_URL
        assert "fake-proxy" not in resolved_async_base_url


def test_rag_index_imports_official_base_url_from_config():
    """Source-level guard: rag/index.py must source its pin from the one
    shared constant, not redefine or hardcode its own copy.

    Stage 2B-D Section E: OFFICIAL_OPENAI_BASE_URL's canonical home moved to
    the pure rag/constants.py module (config.py re-exports it unchanged) so
    that merely importing rag/index.py never requires credential-validating
    config — rag/index.py now references it as `rag_constants.
    OFFICIAL_OPENAI_BASE_URL` rather than a bare name bound from `config`.
    """
    content = (config.BASE_DIR / "rag" / "index.py").read_text(encoding="utf-8")
    assert "OFFICIAL_OPENAI_BASE_URL" in content
    assert "base_url=rag_constants.OFFICIAL_OPENAI_BASE_URL" in content
