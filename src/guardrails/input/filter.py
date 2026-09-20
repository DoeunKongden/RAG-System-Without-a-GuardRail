# TODO: Implement `InputGuardrailFilter` — the single entry point that
#       orchestrates sanitisation + validation for every incoming user message.
#
# Responsibilities:
# 1. Accept a raw user query string.
# 2. Run it through `InputSanitizer.sanitize()` first.
# 3. Pass the cleaned text to `LlamaGuardInputValidator.check()`.
# 4. Return the `InputGuardrailResult` to the caller.
#
# The chat route (src/api/routes/chat.py) should call this filter BEFORE
# passing the query to `OllamaGenerator.generate_augmented()`.
# If the result decision is GuardrailDecision.BLOCK, the route should return
# a safe refusal response immediately without running the RAG pipeline.
#
# Also update `guardrail_triggered` in ChatResponse to True when blocked.
#
# Interface suggestion:
#   class InputGuardrailFilter:
#       def run(self, query: str) -> InputGuardrailResult: ...
