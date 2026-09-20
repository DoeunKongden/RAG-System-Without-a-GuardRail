import re
from uuid import uuid4

from src.ingestion.chunkers.base import BaseChunker
from src.models import ChunkedDocument, RawDocument

# Matches Wikipedia section headers: == Title == or === Title ===
_SECTION_RE = re.compile(r"^(==+)\s*(.+?)\s*\1$", re.MULTILINE)


class SemanticChunker(BaseChunker):
    """Splits Wikipedia articles on section boundaries.

    Each section becomes one chunk. Sections that exceed max_chunk_size
    are further split on paragraph boundaries.
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self._max = max_chunk_size

    def chunk(self, document: RawDocument) -> list[ChunkedDocument]:
        sections = self._extract_sections(document.content)
        chunks: list[ChunkedDocument] = []

        for section_title, section_body in sections:
            pieces = self._split_large(section_body)
            for text in pieces:
                chunks.append(
                    ChunkedDocument(
                        chunk_id=str(uuid4()),
                        document_id=document.id,
                        chunk_index=len(chunks),
                        content=text,
                        chunk_size=len(text),
                        metadata={
                            "title": document.title,
                            "section": section_title,
                            "source_url": document.source_url or "",
                        },
                    )
                )

        return chunks

    def _extract_sections(self, text: str) -> list[tuple[str, str]]:
        boundaries = [(m.start(), m.group(2)) for m in _SECTION_RE.finditer(text)]

        if not boundaries:
            return [("main", text.strip())]

        sections: list[tuple[str, str]] = []

        # Content before the first header is the intro
        intro = text[: boundaries[0][0]].strip()
        if intro:
            sections.append(("introduction", intro))

        for i, (pos, title) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            body = _SECTION_RE.sub("", text[pos:end]).strip()
            if body:
                sections.append((title, body))

        return sections

    def _split_large(self, text: str) -> list[str]:
        if len(text) <= self._max:
            return [text]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > self._max and current:
                chunks.append(current)
                current = para
            else:
                current = (current + "\n\n" + para).strip()
        if current:
            chunks.append(current)
        return chunks
