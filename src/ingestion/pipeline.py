import hashlib

from src.config.settings import get_settings
from src.embedding.ollama_embedder import OllamaEmbedder
from src.ingestion.chunkers.recursive_chunker import RecursiveChunker
from src.ingestion.chunkers.semantic_chunker import SemanticChunker
from src.ingestion.loaders.wiki_loader import WikipediaLoader
from src.models import ChunkedDocument, EmbeddedChunk, IngestResult, RawDocument
from src.vector_store.qdrant_store import QdrantStore


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run_ingestion(chunk_strategy: str = "recursive") -> IngestResult:
    settings = get_settings()
    start = _now_ms()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("Loading Wikipedia articles...")
    loader = WikipediaLoader()
    docs: list[RawDocument] = loader.load()
    print(f"  Loaded {len(docs)} documents")

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    chunker = (
        SemanticChunker() if chunk_strategy == "semantic" else RecursiveChunker()
    )
    all_chunks: list[ChunkedDocument] = []
    for doc in docs:
        all_chunks.extend(chunker.chunk(doc))
    print(f"  Created {len(all_chunks)} chunks (strategy={chunk_strategy})")

    # ── 3. Embed + store ──────────────────────────────────────────────────────
    errors: list[str] = []
    stored = 0

    with OllamaEmbedder() as embedder, QdrantStore() as store:
        # Infer vector size from a single embedding and create collection once
        probe = embedder.embed(all_chunks[0].content)
        store.ensure_collection(vector_size=len(probe))
        print(f"  Collection '{settings.qdrant_collection}' ready (dim={len(probe)})")

        # Process in batches to avoid overwhelming the Ollama server
        batch_size = 16
        texts = [c.content for c in all_chunks]

        for i in range(0, len(texts), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            try:
                vectors = embedder.embed_batch(batch_texts)
                embedded = [
                    EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        embedding=vec,
                        embedding_model=embedder.model_name,
                        content_hash=_hash(chunk.content),
                        metadata=chunk.metadata,
                    )
                    for chunk, vec in zip(batch_chunks, vectors)
                ]
                stored += store.upsert(embedded)
                print(f"  Upserted batch {i // batch_size + 1} ({stored}/{len(all_chunks)})")
            except Exception as e:
                errors.append(f"batch {i}-{i + batch_size}: {e}")
                print(f"  ERROR in batch {i}-{i + batch_size}: {e}")

    status = "success" if not errors else ("partial" if stored > 0 else "failed")
    return IngestResult(
        status=status,
        documents_ingested=len(docs),
        chunks_created=len(all_chunks),
        embeddings_stored=stored,
        errors=errors,
        duration_ms=_now_ms() - start,
    )


def _now_ms() -> float:
    import time
    return time.monotonic() * 1000


if __name__ == "__main__":
    result = run_ingestion()
    print(f"\nResult: {result.model_dump_json(indent=2)}")
