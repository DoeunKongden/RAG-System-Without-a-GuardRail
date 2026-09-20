import time

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.embedding.ollama_embedder import OllamaEmbedder
from src.models import RetrievedChunk, RetrievalResult, SearchQuery
from src.retrieval.searchers.base import BaseSearcher
from src.vector_store.qdrant_store import QdrantStore

_RRF_K = 60
_PAYLOAD_RESERVED = {"document_id", "content", "embedding_model", "content_hash"}


class HybridSearcher(BaseSearcher):
    """Dense + BM25 hybrid retrieval fused with Reciprocal Rank Fusion (RRF).

    On first search, fetches all chunks from Qdrant to build a BM25 index
    in memory. Subsequent searches reuse the cached index.
    """

    def __init__(self) -> None:
        self._embedder = OllamaEmbedder()
        self._store = QdrantStore()
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def search(self, query: SearchQuery) -> RetrievalResult:
        start = time.monotonic()
        candidate_limit = max(query.limit * 4, 20)

        # 1. Dense retrieval — fetch wider candidate pool for fusion
        embedding = self._embedder.embed(query.query)
        dense_query = query.model_copy(
            update={"query_embedding": embedding, "limit": candidate_limit}
        )
        dense_chunks = self._store.search(dense_query)

        # 2. Sparse retrieval — BM25 over the full stored corpus
        sparse_chunks = self._bm25_search(query.query, top_k=candidate_limit)

        # 3. RRF fusion
        fused = self._rrf(dense_chunks, sparse_chunks)

        # 4. Trim to requested limit and apply similarity threshold
        fused = fused[: query.limit]
        before = len(fused)
        fused = [c for c in fused if c.relevance_score >= query.similarity_threshold]

        return RetrievalResult(
            query=query.query,
            chunks=fused,
            total_found=len(fused),
            threshold_filtered_count=before - len(fused),
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            self._build_bm25_index()

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens) # type: ignore
        max_score = max(scores) if scores.max() > 0 else 1.0

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            if score <= 0:
                break
            c = self._corpus[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    content=c["content"],
                    relevance_score=round(float(score / max_score), 4),
                    metadata=c["metadata"],
                )
            )
        return results

    def _build_bm25_index(self) -> None:
        settings = get_settings()
        client = QdrantClient(url=str(settings.qdrant_url))

        all_points: list = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if offset is None:
                break
        client.close()

        self._corpus = [
            {
                "chunk_id": str(p.id),
                "document_id": p.payload.get("document_id", ""),
                "content": p.payload.get("content", ""),
                "metadata": {
                    k: v for k, v in p.payload.items() if k not in _PAYLOAD_RESERVED
                },
            }
            for p in all_points
        ]

        tokenized = [c["content"].lower().split() for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built over {len(self._corpus)} chunks")

    # ── RRF ───────────────────────────────────────────────────────────────────

    def _rrf(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        k: int = _RRF_K,
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense, 1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse, 1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map.setdefault(chunk.chunk_id, chunk)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_rrf = ranked[0][1] if ranked else 1.0

        return [
            chunk_map[cid].model_copy(
                update={"relevance_score": round(rrf_score / max_rrf, 4)}
            )
            for cid, rrf_score in ranked
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._embedder.close()
        self._store.close()
