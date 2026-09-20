from abc import ABC, abstractmethod

from src.models import RawDocument


class BaseLoader(ABC):
    @abstractmethod
    def load(self, *args, **kwargs) -> list[RawDocument]: ...
