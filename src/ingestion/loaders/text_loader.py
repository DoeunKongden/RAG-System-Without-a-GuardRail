from pathlib import Path
from uuid import uuid4

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.preprocessors.cleaner import TextCleaner
from src.models import RawDocument

_EXTENSIONS = {".txt", ".md", ".rst", ".csv"}


class TextLoader(BaseLoader):
    def __init__(self) -> None:
        self._cleaner = TextCleaner()

    def load(self, path: str | Path) -> list[RawDocument]:
        path = Path(path)
        if path.is_dir():
            paths = sorted(
                p for ext in _EXTENSIONS for p in path.glob(f"*{ext}")
            )
        else:
            paths = [path]
        return [doc for p in paths for doc in [self._load_file(p)] if doc]

    def _load_file(self, path: Path) -> RawDocument | None:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            content = self._cleaner.clean(raw)
            if not content:
                return None
            return RawDocument(
                id=str(uuid4()),
                title=path.stem,
                content=content,
                source="file",
                source_url=str(path.resolve()),
                metadata={"file_name": path.name, "extension": path.suffix},
            )
        except Exception as e:
            print(f"  TextLoader: skipping {path.name} — {e}")
            return None
