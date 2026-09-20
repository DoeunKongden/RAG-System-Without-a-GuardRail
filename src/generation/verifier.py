import re
import time

import ollama

from src.config.settings import get_settings
from src.models.guardrail import VerifierRequest, VerifierResult

_PROMPT_TEMPLATE = """\
You are a grounding checker. Your only job is to decide whether the given answer \
is fully supported by the provided context passages — nothing else.

CONTEXT:
{context}

QUESTION: {query}

ANSWER: {answer}

Rules:
- GROUNDED = yes  if every claim in the answer can be traced to the context.
- GROUNDED = no   if the answer adds facts not present in the context.
- CONFIDENCE is your certainty in that verdict (0.0 = unsure, 1.0 = certain).
- RISK reflects hallucination danger: low / medium / high.

Respond in EXACTLY this format, nothing else:
GROUNDED: yes or no
CONFIDENCE: 0.0 to 1.0
RISK: low, medium, or high"""


def _build_context(chunks: list[str]) -> str:
    if not chunks:
        return "(no context provided)"
    return "\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks))


def _parse(raw: str) -> tuple[bool, float, str]:
    """Extract (is_grounded, confidence, hallucination_risk) from model output."""
    grounded = False
    confidence = 0.5
    risk = "medium"

    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("grounded:"):
            value = line.split(":", 1)[1].strip().lower()
            grounded = value.startswith("yes")
        elif line.lower().startswith("confidence:"):
            m = re.search(r"-?[\d.]+", line)
            if m:
                confidence = max(0.0, min(1.0, float(m.group())))
        elif line.lower().startswith("risk:"):
            value = line.split(":", 1)[1].strip().lower()
            if value in ("low", "medium", "high"):
                risk = value

    return grounded, confidence, risk


class AnswerVerifier:
    """Calls the verifier LLM to check if the generated answer is grounded in
    the retrieved context chunks."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.verifier_model
        self._client = ollama.Client(host=str(settings.ollama_host))

    def verify(self, request: VerifierRequest) -> VerifierResult:
        prompt = _PROMPT_TEMPLATE.format(
            context=_build_context(request.retrieved_chunks),
            query=request.query,
            answer=request.generated_answer,
        )

        start = time.perf_counter()
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0.0},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        is_grounded, confidence, risk = _parse(response.response)

        return VerifierResult(
            is_grounded=is_grounded,
            confidence=confidence,
            hallucination_risk=risk,
            model=self._model,
            latency_ms=latency_ms,
        )
