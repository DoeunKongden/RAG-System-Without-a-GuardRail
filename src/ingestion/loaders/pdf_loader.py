from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.preprocessors.cleaner import PDFCleaner
from src.models import RawDocument


class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        self._cleaner = PDFCleaner()

    def load(self, path: str | Path) -> list[RawDocument]:
        path = Path(path)
        paths = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        return [doc for p in paths for doc in [self._load_file(p)] if doc]

    def _load_file(self, path: Path) -> RawDocument | None:
        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            content = self._cleaner.clean("\n\n".join(pages))
            if not content:
                return None
            return RawDocument(
                id=str(uuid4()),
                title=path.stem,
                content=content,
                source="file",
                source_url=str(path.resolve()),
                metadata={
                    "file_name": path.name,
                    "page_count": str(len(reader.pages)),
                },
            )
        except Exception as e:
            print(f"  PDFLoader: skipping {path.name} — {e}")
            return None
