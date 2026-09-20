from abc import ABC, abstractmethod

from src.models import EmbeddedChunk, RetrievedChunk, SearchQuery


class BaseVectorStore(ABC):
    @abstractmethod
    def collection_exists(self) -> bool: ...

    @abstractmethod
    def create_collection(self, vector_size: int) -> None: ...

    def ensure_collection(self, vector_size: int) -> None:
        if not self.collection_exists():
            self.create_collection(vector_size)

    @abstractmethod
    def upsert(self, chunks: list[EmbeddedChunk]) -> int: ...

    @abstractmethod
    def search(self, query: SearchQuery) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "BaseVectorStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
