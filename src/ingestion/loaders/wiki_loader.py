import time
from uuid import uuid4

import httpx

from src.config.settings import get_settings
from src.ingestion.loaders.base import BaseLoader
from src.ingestion.preprocessors.cleaner import WikiCleaner
from src.models import RawDocument

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {
    # Wikipedia requires a descriptive User-Agent per their API policy
    "User-Agent": "kongden-rag/1.0 (educational RAG project; https://github.com/kongden)",
    "Accept": "application/json",
}


class WikipediaLoader(BaseLoader):
    def __init__(self, results_per_topic: int = 3, request_delay: float = 1.0) -> None:
        self._results_per_topic = results_per_topic
        self._delay = request_delay  # seconds between API calls to respect rate limits
        self._cleaner = WikiCleaner()
        self._client = httpx.Client(headers=_HEADERS, timeout=30.0, follow_redirects=True)

    def load(self) -> list[RawDocument]:
        settings = get_settings()
        documents: list[RawDocument] = []
        seen_titles: set[str] = set()

        for topic in settings.wiki_topics:
            titles = self._search(topic)
            for title in titles:
                if title in seen_titles:
                    continue
                doc = self._fetch_page(title, topic)
                if doc is not None:
                    documents.append(doc)
                    seen_titles.add(title)
                    print(f"  [{len(documents)}] {doc.title} ({len(doc.content)} chars)")
                if len(documents) >= settings.wiki_max_articles:
                    return documents

        return documents

    def _search(self, query: str) -> list[str]:
        time.sleep(self._delay)
        resp = self._get(_API, params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": self._results_per_topic,
            "format": "json",
        })
        if resp is None:
            return []
        hits = resp.get("query", {}).get("search", [])
        return [h["title"] for h in hits]

    def _fetch_page(self, title: str, topic: str) -> RawDocument | None:
        time.sleep(self._delay)
        resp = self._get(_API, params={
            "action": "query",
            "prop": "extracts|info",
            "explaintext": True,
            "inprop": "url",
            "titles": title,
            "redirects": 1,
            "format": "json",
        })
        if resp is None:
            return None

        pages = resp.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))

        if "missing" in page:
            return None

        content = self._cleaner.clean(page.get("extract", ""))
        if not content:
            return None

        url = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        return RawDocument(
            id=str(uuid4()),
            title=page.get("title", title),
            content=content,
            source="wikipedia",
            source_url=url,
            metadata={"topic": topic, "pageid": str(page.get("pageid", ""))},
        )

    def _get(self, url: str, params: dict) -> dict | None:
        for attempt in range(3):
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("retry-after", 10))
                    print(f"  Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"  Wikipedia API error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        return None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WikipediaLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
