"""
Shared test doubles for Stage 2B Qdrant tests. NOT a test module itself
(no `test_` prefix — pytest will not collect this file).

DeterministicFakeEmbeddings replaces OpenAIEmbeddings in tests that need a
real local VectorIndex/Qdrant collection without ever calling OpenAI: same
input text always produces the same vector, and call counts are recorded
so tests can assert "zero embedding calls happened" precisely instead of
inferring it from timing or absence of network mocks.
"""

import hashlib
import math
import random
from typing import List

DEFAULT_DIMENSIONS = 1536


class DeterministicFakeEmbeddings:
    """Minimal stand-in for langchain_openai.OpenAIEmbeddings exposing
    only the two methods VectorIndex actually calls: embed_documents()
    and embed_query(). Never touches the network."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        self.dimensions = dimensions
        self.embed_documents_call_count = 0
        self.embed_documents_text_count = 0
        self.embed_query_call_count = 0

    def _vector_for(self, text: str) -> List[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embed_documents_call_count += 1
        self.embed_documents_text_count += len(texts)
        return [self._vector_for(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        self.embed_query_call_count += 1
        return self._vector_for(text)
