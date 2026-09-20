import time

import httpx

from src.config.settings import get_settings
from src.generation.base import BaseGenerator
from src.generation.prompt_builder import PromptBuilder
from src.models import (
    AugmentedResponse,
    LLMResponse,
    PromptContext,
    RetrievedChunk,
    SourceCitation,
)
from src.generation.verifier import AnswerVerifier
from src.models.guardrail import VerifierRequest
from src.retrieval.rerankers.cross_encoder import CrossEncoderReranker
from src.retrieval.searchers.hybrid import HybridSearcher
from src.models import SearchQuery


class OllamaGenerator(BaseGenerator):
    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.generation_model
        self._base_url = str(settings.ollama_host).rstrip("/")
        self._client = httpx.Client(timeout=120.0)
        self._builder = PromptBuilder()
        self._searcher = HybridSearcher()
        self._reranker = CrossEncoderReranker()
        self._verifier = AnswerVerifier()
        self._settings = settings

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(self, context: PromptContext) -> LLMResponse:
        start = time.monotonic()
        prompt = self._builder.render(context)

        resp = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": context.temperature,
                    "num_predict": context.max_tokens,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            content=data.get("response", "").strip(),
            model=self._model,
            finish_reason="stop" if data.get("done") else "length",
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # ── Full RAG pipeline: retrieve → rerank → generate ───────────────────────

    def generate_augmented(self, query: str, chunks: list[str] | None = None) -> AugmentedResponse:
        start = time.monotonic()

        # 1. Retrieve
        search_query = SearchQuery(
            query=query,
            limit=self._settings.top_k,
            similarity_threshold=self._settings.similarity_threshold,
        )
        retrieval_result = self._searcher.search(search_query)
        retrieved = retrieval_result.chunks

        # 2. Rerank
        ranked = self._reranker.rerank(query, retrieved)
        top_chunks: list[RetrievedChunk] = [r for r in ranked[: self._settings.top_k]]

        # 3. Build prompt and generate
        context = self._builder.build(query, top_chunks)
        llm_response = self.generate(context)

        # 4. Build source citations from top chunks
        sources = [
            SourceCitation(
                chunk_id=c.chunk_id,
                content=c.content,
                relevance_score=c.relevance_score,
                metadata=c.metadata,
            )
            for c in top_chunks
        ]

        # 5. Verify the answer is grounded in the retrieved chunks
        verifier = self._verifier.verify(VerifierRequest(
            query=query,
            generated_answer=llm_response.content,
            retrieved_chunks=[c.content for c in top_chunks],
        ))

        return AugmentedResponse(
            llm_response=llm_response,
            verifier_result=verifier,
            sources=sources,
            total_latency_ms=(time.monotonic() - start) * 1000,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()
        self._searcher.close()
        self._reranker.close()
