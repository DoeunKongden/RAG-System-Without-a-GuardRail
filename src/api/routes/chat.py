import time

from fastapi import APIRouter, HTTPException

from src.generation.ollama_generator import OllamaGenerator
from src.models import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = time.monotonic()

    try:
        with OllamaGenerator() as gen:
            augmented = gen.generate_augmented(request.query)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Generation failed: {e}") from e

    return ChatResponse(
        conversation_id=request.conversation_id,
        answer=augmented.llm_response.content,
        sources=augmented.sources,
        guardrail_triggered=False,
        model=augmented.llm_response.model,
        latency_ms=(time.monotonic() - start) * 1000,
    )
