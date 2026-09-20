from abc import ABC, abstractmethod

from kongden_chatbot.src.models.guardrail import OutputGuardrailRequest, OutputGuardrailResult


class BaseOutputGuardrail(ABC):
    """Abstract base for all output guardrail implementations."""

    @abstractmethod
    def check(self, request: OutputGuardrailRequest) -> OutputGuardrailResult:
        """Inspect the generated response and return an allow / warn / block decision."""
        ...

    def __enter__(self) -> "BaseOutputGuardrail":
        return self

    def __exit__(self, *_args) -> None:
        pass
