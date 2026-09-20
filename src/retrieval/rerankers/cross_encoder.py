import httpx

from src.config.settings import get_settings
from src.models import RankedChunk, RetrievedChunk
from src.retrieval.rerankers.base import BaseReranker

# qwen3-reranker prompt format as per the model's instruction template
_RERANK_PROMPT = (
    "<|im_start|>system\n"
    "Judge whether the following Document is helpful for answering the Query. "
    "Output a single floating-point relevance score between 0 and 1. "
    "Output ONLY the number, nothing else.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Query: {query}\n\n"
    "Document: {passage}\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class CrossEncoderReranker(BaseReranker):
    """Pointwise reranker using qwen3-reranker-4b via Ollama."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.reranker_model
        self._base_url = str(settings.ollama_host).rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RankedChunk]:
        scored: list[tuple[RetrievedChunk, float]] = [
            (chunk, self._score(query, chunk.content))
            for chunk in chunks
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                relevance_score=chunk.relevance_score,
                metadata=chunk.metadata,
                original_score=chunk.relevance_score,
                reranked_score=round(rerank_score, 4),
                rank_position=i + 1,
            )
            for i, (chunk, rerank_score) in enumerate(scored)
        ]

    def _score(self, query: str, passage: str) -> float:
        prompt = _RERANK_PROMPT.format(query=query, passage=passage[:1200])
        try:
            resp = self._client.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json().get("response", "0.5").strip()
            score = float(text.split()[0])   # take first token in case of extra output
            return max(0.0, min(1.0, score))  # clamp to [0, 1]
        except Exception:
            return 0.5  # neutral fallback

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CrossEncoderReranker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
