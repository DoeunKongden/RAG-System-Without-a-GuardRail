from uuid import uuid4

from src.ingestion.chunkers.base import BaseChunker
from src.models import ChunkedDocument, RawDocument


class RecursiveChunker(BaseChunker):
    _SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._size = chunk_size
        self._overlap = chunk_overlap

    def chunk(self, document: RawDocument) -> list[ChunkedDocument]:
        texts = self._split(document.content, self._SEPARATORS)
        return [
            ChunkedDocument(
                chunk_id=str(uuid4()),
                document_id=document.id,
                chunk_index=i,
                content=t,
                chunk_size=len(t),
                metadata={
                    "title": document.title,
                    "source_url": document.source_url or "",
                },
            )
            for i, t in enumerate(texts)
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        sep, new_seps = separators[-1], []
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep, new_seps = s, separators[i + 1:]
                break

        splits = text.split(sep) if sep else list(text)

        pieces: list[str] = []
        for s in splits:
            if len(s) > self._size and new_seps:
                pieces.extend(self._split(s, new_seps))
            else:
                pieces.append(s)

        return self._merge(pieces, sep)

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            add_len = len(piece) + (len(sep) if current else 0)
            if current_len + add_len > self._size and current:
                chunks.append(sep.join(current))
                while current and current_len > self._overlap:
                    dropped = current.pop(0)
                    current_len -= len(dropped) + len(sep)
            current.append(piece)
            current_len += len(piece) + (len(sep) if len(current) > 1 else 0)

        if current:
            chunks.append(sep.join(current))

        return [c.strip() for c in chunks if c.strip()]
