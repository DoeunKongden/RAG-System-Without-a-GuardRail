from abc import ABC, abstractmethod

from src.models import ChunkedDocument, RawDocument


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: RawDocument) -> list[ChunkedDocument]: ...
