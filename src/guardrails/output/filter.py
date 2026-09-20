# TODO: Implement `OutputGuardrailFilter` — orchestrates output screening
#       after the LLM generates a response but before it is sent to the user.
#
# Responsibilities:
# 1. Receive the generated answer and the original user query.
# 2. Run them through `LlamaGuardOutputValidator.check()`.
# 3. If decision is GuardrailDecision.BLOCK, replace the answer with a
#    safe fallback string (e.g. "I'm unable to provide that information.").
# 4. If decision is GuardrailDecision.WARN, keep the answer but flag it.
# 5. Return both the (possibly replaced) answer and the OutputGuardrailResult.
#
# The chat route (src/api/routes/chat.py) should call this filter AFTER
# OllamaGenerator.generate_augmented() and update guardrail_triggered accordingly.
#
# Interface suggestion:
#   class OutputGuardrailFilter:
#       def run(self, query: str, answer: str) -> tuple[str, OutputGuardrailResult]: ...
