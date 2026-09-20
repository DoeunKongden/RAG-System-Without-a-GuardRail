from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from src.config.settings import get_settings
from src.models import EmbeddedChunk, RetrievedChunk, SearchQuery
from src.vector_store.base import BaseVectorStore

_PAYLOAD_RESERVED = {"document_id", "content", "embedding_model", "content_hash"}


class QdrantStore(BaseVectorStore):
    def __init__(self) -> None:
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._client = QdrantClient(url=str(settings.qdrant_url))

    # ── Collection management ─────────────────────────────────────────────────

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def create_collection(self, vector_size: int) -> None:
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "embedding_model": chunk.embedding_model,
                    "content_hash": chunk.content_hash,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def delete(self, chunk_ids: list[str]) -> int:
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=chunk_ids),  # type: ignore[arg-type]
        )
        return len(chunk_ids)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(self, query: SearchQuery) -> list[RetrievedChunk]:
        if query.query_embedding is None:
            raise ValueError("query.query_embedding must be set before calling search")

        results = self._client.query_points(
            collection_name=self._collection,
            query=query.query_embedding,
            limit=query.limit,
            score_threshold=query.similarity_threshold,
            with_payload=True,
        ).points

        return [
            RetrievedChunk(
                chunk_id=str(r.id),
                document_id=r.payload.get("document_id", ""),
                content=r.payload.get("content", ""),
                relevance_score=r.score,
                metadata={k: v for k, v in r.payload.items() if k not in _PAYLOAD_RESERVED},
            )
            for r in results
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()
