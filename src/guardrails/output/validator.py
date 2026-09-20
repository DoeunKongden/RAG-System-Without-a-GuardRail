# TODO: Implement `LlamaGuardOutputValidator` — screens the LLM-generated answer
#       before it is returned to the user.
#
# This is structurally similar to LlamaGuardInputValidator but uses the
# multi-turn LlamaGuard prompt format that includes both the user query
# and the generated assistant response:
#
#   <|begin_of_text|><|start_header_id|>user<|end_header_id|>
#   Task: Check if the following conversation is safe.
#   ...
#   <BEGIN CONVERSATION>
#   User: {original_query}
#   Agent: {generated_text}
#   <END CONVERSATION>
#   ...
#   <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# Steps:
# 1. Inherit from `BaseOutputGuardrail`.
# 2. Build the multi-turn prompt using both request.original_query and
#    request.generated_text.
# 3. Call Ollama with model=settings.guardrail_model and parse the response
#    the same way as the input validator (safe / unsafe + category codes).
# 4. Return OutputGuardrailResult with decision, violations, model, latency_ms.
