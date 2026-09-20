import time

import ollama

from kongden_chatbot.src.config.settings import get_settings
from kongden_chatbot.src.guardrails._shared import UNSAFE_CONTENT_CATEGORIES, parse_llamaguard_response
from kongden_chatbot.src.guardrails.input.base import BaseInputGuardrail
from kongden_chatbot.src.models.guardrail import InputGuardrailRequest, InputGuardrailResult

_PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in the 'User' messages in the conversation \
according to our safety policy with the following categories:

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {message}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories.\
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".format(categories=UNSAFE_CONTENT_CATEGORIES, message="{message}")


class LlamaGuardInputValidator(BaseInputGuardrail):
    """Calls llama-guard3:1b via Ollama to classify user input for safety."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.guardrail_model
        self._client = ollama.Client(host=str(settings.ollama_host))

    def check(self, request: InputGuardrailRequest) -> InputGuardrailResult:
        prompt = _PROMPT_TEMPLATE.format(message=request.message)

        start = time.perf_counter()
        response = self._client.generate(model=self._model, prompt=prompt, stream=False)
        latency_ms = (time.perf_counter() - start) * 1000

        decision, violations = parse_llamaguard_response(response.response)

        return InputGuardrailResult(
            request_id=request.request_id,
            decision=decision,
            violations=violations,
            model=self._model,
            latency_ms=latency_ms,
        )
