import httpx

from src.config.settings import get_settings
from src.embedding.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.embedding_model
        self._base_url = str(settings.ollama_host).rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        response = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def close(self) -> None:
        self._client.close()
