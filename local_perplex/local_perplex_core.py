from __future__ import annotations

import math
import re
from collections import Counter, namedtuple
from typing import Iterable
from urllib.parse import urlparse


TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
RankedSource = namedtuple("RankedSource", ["title", "url", "content", "relevance", "category"])


def tokenize(text: str) -> list[str]:
    """Tokenize text in pure Python for ranking and graph extraction."""
    return TOKEN_RE.findall((text or "").lower())


def calculate_score(query: str, content: str) -> float:
    """Calculate a normalized term-frequency relevance score."""
    query_tokens = tokenize(query)
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    content_counts = Counter(content_tokens)
    score = 0.0
    for token in set(query_tokens):
        if token in content_counts:
            score += 1.0 + math.log(content_counts[token])

    return min(score / math.log(1.0 + len(content_tokens)), 1.0)


def _category_for_url(url: str) -> str:
    host = (urlparse(url or "").netloc or "").lower()
    if host.endswith(".edu") or "arxiv.org" in host or "scholar.google" in host:
        return "Academic"
    if host.endswith(".gov"):
        return "Government"
    if "github.com" in host or "stackoverflow.com" in host:
        return "Technical"
    if "youtube.com" in host or "youtu.be" in host:
        return "Video"
    if "reddit.com" in host or "news.ycombinator.com" in host:
        return "Community"
    if "wikipedia.org" in host:
        return "Reference"
    return "General"


def _source_get(source, key: str, default: str = ""):
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


class ResearchEngine:
    """Ranks candidate sources against a query.

    Scoring blends three signals:

    * **BM25** over the candidate set (title counted twice so it carries weight),
    * **query coverage**: the share of distinct query terms a source contains,
    * a small **authority prior** for academic and government domains.

    Duplicate URLs are collapsed. ``relevance`` is normalised to 0..1.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def rank_sources(self, query: str, sources: Iterable[dict]) -> list[RankedSource]:
        from .retrieval import BM25, analyze

        items, seen = [], set()
        for source in sources:
            url = _source_get(source, "url", "")
            key = _normalize_url(url)
            if key and key in seen:
                continue
            seen.add(key)
            title = _source_get(source, "title", "Unknown") or "Unknown"
            content = _source_get(source, "content", "") or ""
            category = _source_get(source, "category", "") or _category_for_url(url)
            items.append((title, url, content, category))
        if not items:
            return []

        q_terms = analyze(query)
        docs = [analyze(f"{t} {t} {c[:20000]}") for t, _, c, _ in items]
        bm25 = BM25(docs, k1=self.k1, b=self.b)
        raw = [bm25.score(q_terms, i) for i in range(len(items))]
        top = max(raw) or 1.0
        q_set = set(q_terms)

        ranked: list[RankedSource] = []
        for (title, url, content, category), score, doc in zip(items, raw, docs):
            coverage = (len(q_set & set(doc)) / len(q_set)) if q_set else 0.0
            relevance = 0.65 * (score / top) + 0.35 * coverage
            relevance *= _AUTHORITY.get(category, 1.0)
            ranked.append(RankedSource(title=title, url=url, content=content,
                                       relevance=round(min(relevance, 1.0), 4),
                                       category=category))
        return sorted(ranked, key=lambda item: item.relevance, reverse=True)


_AUTHORITY = {"Academic": 1.15, "Government": 1.1, "Reference": 1.05}


def _normalize_url(url: str) -> str:
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower().removeprefix("www.")
    return f"{host}{parsed.path.rstrip('/')}" if host else (url or "")
