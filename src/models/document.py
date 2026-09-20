from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    id: str
    title: str
    content: str = Field(..., min_length=1)
    source: Literal["wikipedia", "web", "file"] = "wikipedia"
    source_url: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkedDocument(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(..., gt=0)
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str = Field(..., min_length=1)
    embedding: list[float]
    embedding_model: str
    content_hash: str
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestRequest(BaseModel):
    topics: list[str] = Field(..., min_length=1)
    max_articles: int = Field(default=15, ge=1, le=100)
    chunk_strategy: Literal["recursive", "semantic"] = "recursive"


class IngestResult(BaseModel):
    status: Literal["success", "partial", "failed"]
    documents_ingested: int = Field(..., ge=0)
    chunks_created: int = Field(..., ge=0)
    embeddings_stored: int = Field(..., ge=0)
    errors: list[str] = Field(default_factory=list)
    duration_ms: float
