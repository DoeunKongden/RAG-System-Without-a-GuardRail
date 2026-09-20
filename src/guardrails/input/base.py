# TODO: Define the abstract base class for input guardrails.
#
# Requirements:
# - Create an abstract class `BaseInputGuardrail` using Python's `abc.ABC`.
# - Define an abstract method `check(request: InputGuardrailRequest) -> InputGuardrailResult`
#   that all concrete input guardrail implementations must override.
# - Optionally add __enter__ / __exit__ lifecycle methods if your implementation
#   holds external connections (e.g. an HTTP client to Ollama).
#
# Reference models: InputGuardrailRequest, InputGuardrailResult (src/models/guardrail.py)
