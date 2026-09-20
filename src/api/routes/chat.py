import time

from fastapi import APIRouter, HTTPException

from src.generation.ollama_generator import OllamaGenerator
from src.guardrails.input.filter import InputGuardrailFilter
from src.guardrails.output.filter import OutputGuardrailFilter
from src.models import ChatRequest, ChatResponse
from src.models.base import GuardrailDecision

router = APIRouter()

_input_guardrail = InputGuardrailFilter()
_output_guardrail = OutputGuardrailFilter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = time.monotonic()

    # ── Step 1: Input guardrail ───────────────────────────────────────────────
    # Sanitize + LlamaGuard classify before touching the RAG pipeline.
    input_result = _input_guardrail.run(request.query)

    if input_result.decision == GuardrailDecision.BLOCK:
        return ChatResponse(
            conversation_id=request.conversation_id,
            answer="I'm sorry, I can't help with that request.",
            sources=[],
            guardrail_triggered=True,
            model=input_result.model,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # ── Step 2: RAG generation (only reached for ALLOW or WARN) ──────────────
    try:
        with OllamaGenerator() as gen:
            augmented = gen.generate_augmented(request.query)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Generation failed: {e}") from e

    # ── Step 3: Output guardrail ──────────────────────────────────────────────
    # Screen the generated answer before returning it to the user.
    answer, output_result = _output_guardrail.run(
        query=request.query,
        answer=augmented.llm_response.content,
    )

    # ── Step 4: Return response ───────────────────────────────────────────────
    guardrail_triggered = (
        input_result.decision == GuardrailDecision.WARN
        or output_result.decision != GuardrailDecision.ALLOW
    )

    return ChatResponse(
        conversation_id=request.conversation_id,
        answer=answer,
        sources=augmented.sources,
        guardrail_triggered=guardrail_triggered,
        model=augmented.llm_response.model,
        latency_ms=(time.monotonic() - start) * 1000,
    )
