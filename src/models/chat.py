from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import MessageRole


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SourceCitation(BaseModel):
    chunk_id: str
    content: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    top_k: int | None = Field(default=None, ge=1, le=100)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    guardrail_triggered: bool = False
    model: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
