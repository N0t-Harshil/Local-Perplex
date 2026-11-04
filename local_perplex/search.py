"""Web search with pluggable backends plus parallel page reading.

Backends (``Settings.search_provider``):

=========== ========= ===========================================
id          key?      notes
=========== ========= ===========================================
duckduckgo  no        default, via the ``ddgs`` package
searxng     no        self-hosted meta-search, the most private
brave       yes       BRAVE_API_KEY
tavily      yes       TAVILY_API_KEY, returns page content too
serper      yes       SERPER_API_KEY (Google results)
wikipedia   no        encyclopedic answers
arxiv       no        papers; used automatically for Academic focus
=========== ========= ===========================================
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from .config import Settings, mask_secret
from .local_perplex_core import calculate_score

logger = logging.getLogger(__name__)

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 LocalPerplex/2.0")
MAX_PAGE_BYTES = 2_000_000
MAX_PAGE_CHARS = 20_000
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form",
              "svg", "iframe", "button"]

FOCUS_SITES = {
    "Academic": "site:arxiv.org OR site:scholar.google.com OR site:researchgate.net OR site:nature.com",
    "Reddit": "site:reddit.com",
    "YouTube": "site:youtube.com",
    "Technical": "site:github.com OR site:stackoverflow.com OR site:docs.python.org OR site:developer.mozilla.org",
    "Computational": "site:wolframalpha.com OR site:wikipedia.org",
}


@dataclass(frozen=True)
class SearchBackendSpec:
    id: str
    label: str
    needs_key: bool
    env_key: str | None = None
    needs_url: bool = False
    notes: str = ""


SEARCH_BACKENDS: dict[str, SearchBackendSpec] = {s.id: s for s in [
    SearchBackendSpec("duckduckgo", "DuckDuckGo", False, notes="No key needed."),
    SearchBackendSpec("searxng", "SearXNG (self-hosted)", False, needs_url=True,
                      notes="Point at your instance, e.g. http://localhost:8888. Enable the JSON format."),
    SearchBackendSpec("brave", "Brave Search API", True, "BRAVE_API_KEY"),
    SearchBackendSpec("tavily", "Tavily", True, "TAVILY_API_KEY",
                      notes="Built for LLM agents; returns cleaned page text."),
    SearchBackendSpec("serper", "Serper (Google)", True, "SERPER_API_KEY"),
    SearchBackendSpec("wikipedia", "Wikipedia", False),
    SearchBackendSpec("arxiv", "arXiv", False, notes="Research papers only."),
]}


def search_catalog(settings: Settings) -> list[dict]:
    out = []
    for spec in SEARCH_BACKENDS.values():
        key = settings.search_value(spec.id, "api_key", spec.env_key)
        out.append({
            "id": spec.id, "label": spec.label, "needs_key": spec.needs_key,
            "env_key": spec.env_key, "needs_url": spec.needs_url, "notes": spec.notes,
            "has_key": bool(key), "key_hint": mask_secret(key),
            "base_url": settings.search_value(spec.id, "base_url", "SEARXNG_URL" if spec.id == "searxng" else None),
        })
    return out


class SearchEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT,
                                     "Accept-Language": "en-US,en;q=0.9"})
        self.last_timing: dict = {}

    @property
    def backend(self) -> str:
        return self.settings.search_provider if self.settings.search_provider in SEARCH_BACKENDS \
            else "duckduckgo"

    # ── public ───────────────────────────────────────────────────────
    def search(self, query: str, num_results: int = 12, focus_mode: str = "All") -> list[dict]:
        """Search, then read pages in parallel. Returns dicts sorted by relevance."""
        if focus_mode == "Writing":
            return []
        t0 = time.perf_counter()
        hits = self.find(query, num_results=num_results, focus_mode=focus_mode)
        t1 = time.perf_counter()
        sources = self.read_pages(hits, query)
        t2 = time.perf_counter()
        self.last_timing = {"search": round(t1 - t0, 3), "read": round(t2 - t1, 3),
                            "hits": len(hits), "read_ok": len(sources), "backend": self.backend}
        return sources

    def find(self, query: str, num_results: int = 12, focus_mode: str = "All") -> list[dict]:
        """Return raw hits: ``{"title", "url", "snippet", "content"?}``."""
        backend = self.backend
        hits: list[dict] = []
        if focus_mode == "Academic" and backend != "arxiv":
            hits.extend(self._safe(self._arxiv, query, min(5, num_results)))
        q = query
        if focus_mode in FOCUS_SITES and backend in ("duckduckgo", "brave", "serper", "searxng"):
            q = f"{query} {FOCUS_SITES[focus_mode]}"
        fn = getattr(self, f"_{backend}")
        hits.extend(self._safe(fn, q, num_results))
        if not hits and backend != "duckduckgo":
            logger.warning("%s returned nothing, falling back to DuckDuckGo", backend)
            hits.extend(self._safe(self._duckduckgo, q, num_results))
        seen, unique = set(), []
        for h in hits:
            if h.get("url") and h["url"] not in seen:
                seen.add(h["url"])
                unique.append(h)
        return unique

    def read_pages(self, hits: list[dict], query: str, max_workers: int = 12) -> list[dict]:
        sources = []
        pool = ThreadPoolExecutor(max_workers=max_workers)
        futures = [pool.submit(self._read_one, h, query) for h in hits]
        try:
            for fut in as_completed(futures, timeout=12):
                try:
                    result = fut.result()
                except Exception:  # noqa: BLE001 - one bad page never fails a search
                    result = None
                if result:
                    sources.append(result)
        except TimeoutError:
            logger.info("Page reading hit the 12s budget; using %d pages", len(sources))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        sources.sort(key=lambda x: x["relevance"], reverse=True)
        return sources

    # ── page reading ─────────────────────────────────────────────────
    def _read_one(self, hit: dict, query: str) -> dict | None:
        url = hit.get("url")
        title = hit.get("title") or url
        snippet = hit.get("snippet", "")
        text = hit.get("content") or ""
        if len(text) < 400 and not url.lower().endswith(".pdf"):
            page = self._fetch_text(url)
            if page and len(page) > len(text):
                text = page
        if len(text) < 120:
            text = f"{snippet}\n{text}".strip()
        if len(text) < 60:
            return None
        return {"title": title, "url": url, "content": text[:MAX_PAGE_CHARS],
                "snippet": snippet, "relevance": calculate_score(query, text)}

    def _fetch_text(self, url: str) -> str:
        try:
            resp = self.session.get(url, timeout=(3.05, 6), stream=True)
            if resp.status_code != 200:
                return ""
            ctype = resp.headers.get("Content-Type", "")
            if "html" not in ctype and "text" not in ctype:
                return ""
            raw = resp.raw.read(MAX_PAGE_BYTES, decode_content=True)
            encoding = resp.encoding if "charset" in ctype.lower() else "utf-8"
            html = raw.decode(encoding, errors="replace")
        except (requests.exceptions.RequestException, OSError, LookupError):
            return ""
        return html_to_text(html)

    # ── backends ─────────────────────────────────────────────────────
    def _safe(self, fn, query, n) -> list[dict]:
        try:
            return fn(query, n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Search backend %s failed: %s", fn.__name__.strip("_"), exc)
            return []

    def _duckduckgo(self, query: str, n: int) -> list[dict]:
        from ddgs import DDGS
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    rows = list(ddgs.text(query, max_results=n))
                if rows:
                    return [{"title": r.get("title", ""), "url": r.get("href", ""),
                             "snippet": r.get("body", "")} for r in rows]
            except Exception as exc:  # noqa: BLE001
                logger.warning("DuckDuckGo attempt %d failed: %s", attempt + 1, exc)
                time.sleep(0.6 * (2 ** attempt))
        return []

    def _searxng(self, query: str, n: int) -> list[dict]:
        base = self.settings.search_value("searxng", "base_url", "SEARXNG_URL").rstrip("/")
        if not base:
            raise RuntimeError("SearXNG URL not configured")
        resp = self.session.get(f"{base}/search", params={"q": query, "format": "json"}, timeout=8)
        resp.raise_for_status()
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in resp.json().get("results", [])[:n]]

    def _brave(self, query: str, n: int) -> list[dict]:
        key = self._key("brave")
        resp = self.session.get("https://api.search.brave.com/res/v1/web/search",
                                params={"q": query, "count": min(n, 20)},
                                headers={"X-Subscription-Token": key, "Accept": "application/json"},
                                timeout=8)
        resp.raise_for_status()
        rows = (resp.json().get("web") or {}).get("results", [])
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": BeautifulSoup(r.get("description", ""), "html.parser").get_text()}
                for r in rows]

    def _tavily(self, query: str, n: int) -> list[dict]:
        key = self._key("tavily")
        resp = self.session.post("https://api.tavily.com/search",
                                 json={"query": query, "max_results": min(n, 20),
                                       "search_depth": "basic", "include_raw_content": True},
                                 headers={"Authorization": f"Bearer {key}"}, timeout=15)
        resp.raise_for_status()
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("content", ""),
                 "content": r.get("raw_content") or r.get("content", "")}
                for r in resp.json().get("results", [])]

    def _serper(self, query: str, n: int) -> list[dict]:
        key = self._key("serper")
        resp = self.session.post("https://google.serper.dev/search", json={"q": query, "num": n},
                                 headers={"X-API-KEY": key}, timeout=8)
        resp.raise_for_status()
        return [{"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
                for r in resp.json().get("organic", [])]

    def _wikipedia(self, query: str, n: int) -> list[dict]:
        resp = self.session.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": min(n, 10),
            "format": "json"}, timeout=8)
        resp.raise_for_status()
        rows = resp.json().get("query", {}).get("search", [])
        return [{"title": r["title"],
                 "url": f"https://en.wikipedia.org/wiki/{quote(r['title'].replace(' ', '_'))}",
                 "snippet": BeautifulSoup(r.get("snippet", ""), "html.parser").get_text()}
                for r in rows]

    def _arxiv(self, query: str, n: int) -> list[dict]:
        resp = self.session.get("https://export.arxiv.org/api/query", params={
            "search_query": f"all:{query}", "max_results": n, "sortBy": "relevance"}, timeout=10)
        resp.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        hits = []
        for entry in root.findall("a:entry", ns):
            title = " ".join((entry.findtext("a:title", "", ns) or "").split())
            summary = " ".join((entry.findtext("a:summary", "", ns) or "").split())
            url = entry.findtext("a:id", "", ns)
            authors = ", ".join(a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)[:4])
            hits.append({"title": f"{title} (arXiv)", "url": url, "snippet": summary[:300],
                         "content": f"{title}. Authors: {authors}. Abstract: {summary}",
                         "category": "Academic"})
        return hits

    def _key(self, backend: str) -> str:
        spec = SEARCH_BACKENDS[backend]
        key = self.settings.search_value(backend, "api_key", spec.env_key)
        if not key:
            raise RuntimeError(f"{spec.label} needs an API key ({spec.env_key})")
        return key


def html_to_text(html: str) -> str:
    """Extract readable article text, dropping navigation and boilerplate."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td", "blockquote"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if len(text) >= 25 or el.name in ("h1", "h2", "h3", "pre"):
            blocks.append(text)
    text = "\n\n".join(dict.fromkeys(blocks))  # de-duplicate repeated blocks
    if len(text) < 200:  # unusual markup: fall back to all visible text
        text = " ".join(root.get_text(" ", strip=True).split())
    return text
