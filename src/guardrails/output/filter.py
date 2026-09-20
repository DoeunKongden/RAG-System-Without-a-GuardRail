from kongden_chatbot.src.guardrails.output.validator import LlamaGuardOutputValidator
from kongden_chatbot.src.models.base import GuardrailDecision
from kongden_chatbot.src.models.guardrail import OutputGuardrailRequest, OutputGuardrailResult

_SAFE_FALLBACK = "I'm sorry, I'm unable to provide that information."


class OutputGuardrailFilter:
    """Screens the generated answer before it is returned to the user.

    Flow:
      generated answer + original query
        → LlamaGuardOutputValidator
            ALLOW  →  return answer unchanged
            WARN   →  return answer unchanged, caller sets guardrail_triggered=True
            BLOCK  →  replace answer with safe fallback
    """

    def __init__(self) -> None:
        self._validator = LlamaGuardOutputValidator()

    def run(self, query: str, answer: str) -> tuple[str, OutputGuardrailResult]:
        """Returns (final_answer, result).

        final_answer is the original answer unless the decision is BLOCK,
        in which case it is replaced with a safe fallback.
        """
        request = OutputGuardrailRequest(
            generated_text=answer,
            original_query=query,
            conversation_id="",
        )
        result = self._validator.check(request)

        if result.decision == GuardrailDecision.BLOCK:
            return _SAFE_FALLBACK, result

        return answer, result
