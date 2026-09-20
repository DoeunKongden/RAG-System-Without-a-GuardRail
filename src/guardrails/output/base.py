# TODO: Define the abstract base class for output guardrails.
#
# Requirements:
# - Create an abstract class `BaseOutputGuardrail` using Python's `abc.ABC`.
# - Define an abstract method `check(request: OutputGuardrailRequest) -> OutputGuardrailResult`
#   that all concrete output guardrail implementations must override.
# - Mirror the same lifecycle pattern used in BaseInputGuardrail.
#
# Reference models: OutputGuardrailRequest, OutputGuardrailResult (src/models/guardrail.py)
