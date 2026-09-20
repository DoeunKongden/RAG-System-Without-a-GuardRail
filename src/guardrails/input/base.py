from abc import ABC, abstractmethod

from kongden_chatbot.src.models.guardrail import InputGuardrailRequest, InputGuardrailResult


class BaseInputGuardrail(ABC):
    """Abstract base for all input guardrail implementations."""

    @abstractmethod
    def check(self, request: InputGuardrailRequest) -> InputGuardrailResult:
        """Inspect the request and return an allow / warn / block decision."""
        ...

    def __enter__(self) -> "BaseInputGuardrail":
        return self

    def __exit__(self, *_args) -> None:
        pass
