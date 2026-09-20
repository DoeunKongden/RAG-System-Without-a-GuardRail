import time

from src.embedding.ollama_embedder import OllamaEmbedder
from src.models import RetrievalResult, SearchQuery
from src.retrieval.searchers.base import BaseSearcher
from src.vector_store.qdrant_store import QdrantStore


class DenseSearcher(BaseSearcher):
    """Semantic retrieval using embedding vectors stored in Qdrant."""

    def __init__(self) -> None:
        self._embedder = OllamaEmbedder()
        self._store = QdrantStore()

    def search(self, query: SearchQuery) -> RetrievalResult:
        start = time.monotonic()

        embedding = self._embedder.embed(query.query)
        populated = query.model_copy(update={"query_embedding": embedding})
        chunks = self._store.search(populated)

        before = len(chunks)
        chunks = [c for c in chunks if c.relevance_score >= query.similarity_threshold]

        return RetrievalResult(
            query=query.query,
            chunks=chunks,
            total_found=len(chunks),
            threshold_filtered_count=before - len(chunks),
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def close(self) -> None:
        self._embedder.close()
        self._store.close()
