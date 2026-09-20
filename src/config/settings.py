from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.development"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: AnyHttpUrl

    # ── Models ────────────────────────────────────────────────────────────────
    embedding_model: str
    generation_model: str
    guardrail_model: str
    verifier_model: str
    reranker_model: str

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: AnyHttpUrl
    qdrant_collection: str

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 4
    similarity_threshold: float = 0.5

    # ── Ingestion ─────────────────────────────────────────────────────────────
    wiki_topics: list[str]
    wiki_max_articles: int = 15


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
