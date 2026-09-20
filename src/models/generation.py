from typing import Literal

from pydantic import BaseModel, Field

from .chat import SourceCitation
from .guardrail import VerifierResult


class PromptContext(BaseModel):
    system_prompt: str
    query: str
    retrieved_chunks: list[str] = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=50, le=4096)


class LLMResponse(BaseModel):
    content: str = Field(..., min_length=1)
    model: str
    finish_reason: Literal["stop", "length", "error"] = "stop"
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    latency_ms: float


class AugmentedResponse(BaseModel):
    llm_response: LLMResponse
    verifier_result: VerifierResult
    sources: list[SourceCitation] = Field(default_factory=list)
    total_latency_ms: float
