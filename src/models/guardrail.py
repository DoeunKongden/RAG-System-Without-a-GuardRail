from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import GuardrailDecision


class SafetyCategory(str, Enum):
    VIOLENT_CRIMES = "violent_crimes"
    NON_VIOLENT_CRIMES = "non_violent_crimes"
    SEX_CRIMES = "sex_crimes"
    CHILD_EXPLOITATION = "child_exploitation"
    HATE = "hate"
    SELF_HARM = "self_harm"
    SEXUAL_CONTENT = "sexual_content"
    ELECTIONS = "elections"
    SPECIALIZED_ADVICE = "specialized_advice"


class GuardrailViolation(BaseModel):
    category: SafetyCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class InputGuardrailRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))


class InputGuardrailResult(BaseModel):
    request_id: str
    decision: GuardrailDecision
    violations: list[GuardrailViolation] = Field(default_factory=list)
    model: str
    latency_ms: float
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class OutputGuardrailRequest(BaseModel):
    generated_text: str = Field(..., min_length=1)
    original_query: str
    conversation_id: str
    request_id: str = Field(default_factory=lambda: str(uuid4()))


class OutputGuardrailResult(BaseModel):
    request_id: str
    decision: GuardrailDecision
    violations: list[GuardrailViolation] = Field(default_factory=list)
    model: str
    latency_ms: float
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class VerifierRequest(BaseModel):
    query: str
    generated_answer: str
    retrieved_chunks: list[str] = Field(default_factory=list)


class VerifierResult(BaseModel):
    is_grounded: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    hallucination_risk: Literal["low", "medium", "high"] = "low"
    model: str
    latency_ms: float
