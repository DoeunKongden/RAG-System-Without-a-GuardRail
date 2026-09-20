from uuid import uuid4

import httpx
import trafilatura

from src.ingestion.loaders.base import BaseLoader
from src.ingestion.preprocessors.cleaner import WebCleaner
from src.models import RawDocument


class WebLoader(BaseLoader):
    def __init__(self, timeout: float = 15.0) -> None:
        self._cleaner = WebCleaner()
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "kongden-rag/1.0"},
        )

    def load(self, urls: list[str]) -> list[RawDocument]:
        return [doc for url in urls for doc in [self._fetch(url)] if doc]

    def _fetch(self, url: str) -> RawDocument | None:
        try:
            response = self._client.get(url)
            response.raise_for_status()

            text = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if not text:
                return None

            content = self._cleaner.clean(text)
            if not content:
                return None

            meta = trafilatura.extract_metadata(response.text)
            title = meta.title if meta and meta.title else url

            return RawDocument(
                id=str(uuid4()),
                title=title,
                content=content,
                source="web",
                source_url=url,
                metadata={"domain": str(httpx.URL(url).host)},
            )
        except Exception as e:
            print(f"  WebLoader: skipping {url} — {e}")
            return None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WebLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
