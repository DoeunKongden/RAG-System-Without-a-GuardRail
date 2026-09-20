# TODO: Implement `LlamaGuardInputValidator` — the concrete input guardrail
#       that calls the llama-guard3:1b model via Ollama to classify user queries.
#
# Steps:
# 1. Inherit from `BaseInputGuardrail` (input/base.py).
# 2. In __init__, load settings and create an httpx.Client pointed at OLLAMA_HOST.
# 3. In `check(request)`:
#    a. Format the user message using the LlamaGuard prompt template:
#       <|begin_of_text|><|start_header_id|>user<|end_header_id|>
#       Task: Check if the following message is safe.
#       <UNSAFE_CONTENT_CATEGORIES>  ← list the SafetyCategory values
#       <BEGIN CONVERSATION>
#       User: {message}
#       <END CONVERSATION>
#       Provide your safety assessment ...
#       <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#    b. POST to /api/generate with model=settings.guardrail_model, stream=False.
#    c. Parse the response: first token is "safe" or "unsafe".
#       If "unsafe", the second line contains the violated category codes (e.g. "S1,S3").
#    d. Map category codes → SafetyCategory enum values.
#    e. Return InputGuardrailResult with:
#       - decision = GuardrailDecision.ALLOW if safe, else GuardrailDecision.BLOCK
#       - violations = list of GuardrailViolation (one per violated category)
#       - model = settings.guardrail_model
#       - latency_ms = elapsed time
#
# LlamaGuard safety category codes (S1–S9) map to SafetyCategory enum in models/guardrail.py.
# Reference: https://huggingface.co/meta-llama/Llama-Guard-3-1B
