from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    query_embedding: list[float] | None = None
    limit: int = Field(default=4, ge=1, le=100)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    filters: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_found: int = Field(..., ge=0)
    threshold_filtered_count: int = Field(default=0, ge=0)
    latency_ms: float


class RankedChunk(RetrievedChunk):
    original_score: float = Field(..., ge=0.0, le=1.0)
    reranked_score: float = Field(..., ge=0.0, le=1.0)
    rank_position: int = Field(..., ge=1)
