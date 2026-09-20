from src.config.settings import get_settings
from src.models import PromptContext, RetrievedChunk

_SYSTEM_PROMPT = """\
You are a helpful and accurate assistant.

Rules:
- For greetings, small talk, or conversational messages (e.g. "hello", "thanks", "how are you"), respond naturally and briefly without referencing the context.
- For knowledge or factual questions, answer using ONLY the information provided in the context passages below.
- If a knowledge question cannot be answered from the context, say "I don't have enough information to answer that."
- Do not make up facts or use knowledge outside the provided context for factual questions.
- Be concise and direct.
- Cite which passage supports your answer when possible."""

_CHUNK_TEMPLATE = "[{i}] {content}"


class PromptBuilder:
    def __init__(self) -> None:
        settings = get_settings()
        self._temperature = 0.3
        self._max_tokens = 1024
        self._top_k = settings.top_k

    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> PromptContext:
        formatted_chunks = [
            _CHUNK_TEMPLATE.format(i=i + 1, content=c.content.strip())
            for i, c in enumerate(chunks)
        ]
        return PromptContext(
            system_prompt=system_prompt,
            query=query,
            retrieved_chunks=formatted_chunks,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    def render(self, context: PromptContext) -> str:
        """Render the full prompt string sent to the Ollama /api/generate endpoint."""
        separator = "\n\n---\n\n"
        context_block = separator.join(context.retrieved_chunks)

        return (
            f"{context.system_prompt}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {context.query}\n\n"
            f"Answer:"
        )
