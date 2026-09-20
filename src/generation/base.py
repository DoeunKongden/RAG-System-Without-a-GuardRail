from abc import ABC, abstractmethod

from src.models import AugmentedResponse, LLMResponse, PromptContext


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, context: PromptContext) -> LLMResponse: ...

    @abstractmethod
    def generate_augmented(self, query: str, chunks: list[str] | None = None) -> AugmentedResponse: ...

    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseGenerator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
