from abc import ABC, abstractmethod

from src.models import RetrievalResult, SearchQuery


class BaseSearcher(ABC):
    @abstractmethod
    def search(self, query: SearchQuery) -> RetrievalResult: ...

    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseSearcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
