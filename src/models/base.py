from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class GuardrailDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    components: dict[str, bool] = Field(default_factory=dict)
    response_time_ms: float
