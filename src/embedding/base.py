from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "BaseEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
