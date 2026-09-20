from abc import ABC, abstractmethod

from src.models import RankedChunk, RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RankedChunk]: ...
