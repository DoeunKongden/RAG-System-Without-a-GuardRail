import time
from uuid import uuid4

from kongden_chatbot.src.guardrails.input.sanitizer import InputSanitizer
from kongden_chatbot.src.guardrails.input.validator import LlamaGuardInputValidator
from kongden_chatbot.src.models.base import GuardrailDecision
from kongden_chatbot.src.models.guardrail import InputGuardrailRequest, InputGuardrailResult


class InputGuardrailFilter:
    """Single entry point that runs sanitisation then LLM validation.

    Flow:
      raw query
        → InputSanitizer   (rule-based, always runs)
            injection detected?  →  WARN  (skip LLM call)
        → LlamaGuardInputValidator  (LLM, runs only on clean text)
            safe?     →  ALLOW
            unsafe?   →  BLOCK  (with violation list)
    """

    def __init__(self) -> None:
        self._sanitizer = InputSanitizer()
        self._validator = LlamaGuardInputValidator()

    def run(self, query: str) -> InputGuardrailResult:
        start = time.perf_counter()

        # 1. Sanitize — always run first (cheap, rule-based)
        san_result = self._sanitizer.sanitize_full(query)

        # 2. Short-circuit: injection attempt detected by sanitizer
        if san_result.has_injection_attempt:
            latency_ms = (time.perf_counter() - start) * 1000
            return InputGuardrailResult(
                request_id=str(uuid4()),
                decision=GuardrailDecision.WARN,
                violations=[],
                model=InputSanitizer.MODEL_NAME,
                latency_ms=latency_ms,
            )

        # 3. Pass cleaned text to LLM validator
        request = InputGuardrailRequest(
            message=san_result.cleaned_text,
        )
        return self._validator.check(request)
