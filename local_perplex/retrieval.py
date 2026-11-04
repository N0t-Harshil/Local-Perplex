"""Lexical retrieval: Okapi BM25 ranking and query-focused passage selection.

Pure Python, no dependencies. Used for three things:

* ranking scraped web pages against the query,
* picking the most relevant passages of each page for the LLM context
  (instead of blindly sending the first N characters), and
* searching the private document vault chunk by chunk.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from typing import Iterable, Sequence

from .local_perplex_core import tokenize

# Short English stopword list: dropping these sharpens BM25 noticeably on
# natural-language questions ("what is the ..."), without hurting recall.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
""".split())

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def analyze(text: str) -> list[str]:
    """Tokenize, lowercase, drop stopwords and apply a light suffix stemmer."""
    return [(_stem(t) if t[-1] in "sgd" else t) for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS]


@lru_cache(maxsize=262_144)
def _stem(token: str) -> str:
    # A tiny, conservative stemmer: enough to match "models"/"model",
    # "learning"/"learn", "studies"/"study" without a dependency.
    if len(token) <= 4 or token.isdigit():
        return token
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "es" and not token.endswith(("ses", "xes", "zes", "ches", "shes")):
                continue
            if suffix == "s" and token.endswith("ss"):
                continue
            return token[: -len(suffix)] + repl
    return token


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.n = len(corpus)
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        # Lucene-style IDF, always positive.
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: Sequence[str], index: int) -> float:
        freqs = self.doc_freqs[index]
        dl = self.doc_lens[index] or 1
        norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
        total = 0.0
        for term in set(query):
            tf = freqs.get(term)
            if tf:
                total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + norm)
        return total

    def scores(self, query: Sequence[str]) -> list[float]:
        return [self.score(query, i) for i in range(self.n)]


# ── passages ─────────────────────────────────────────────────────────

def split_passages(text: str, target_chars: int = 600, overlap_sentences: int = 1) -> list[str]:
    """Split text into ~``target_chars`` passages on sentence boundaries."""
    text = " ".join((text or "").split()) if "\n\n" not in (text or "") else (text or "")
    sentences = [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]
    passages: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        # Hard-wrap pathological "sentences" (tables, minified text).
        while len(sent) > target_chars * 2:
            head, sent = sent[:target_chars], sent[target_chars:]
            if current:
                passages.append(" ".join(current))
                current, size = [], 0
            passages.append(head)
        current.append(sent)
        size += len(sent) + 1
        if size >= target_chars:
            passages.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(s) + 1 for s in current)
    if current:
        tail = " ".join(current)
        if not passages or tail not in passages[-1]:
            passages.append(tail)
    return passages


def best_passages(query: str, text: str, max_chars: int = 1800, target_chars: int = 600) -> str:
    """Return the passages of ``text`` most relevant to ``query``, in reading order."""
    passages = split_passages(text, target_chars=target_chars)
    if not passages:
        return ""
    if sum(len(p) for p in passages) <= max_chars:
        return " ".join(passages)
    q = analyze(query)
    bm25 = BM25([analyze(p) for p in passages])
    scored = sorted(range(len(passages)), key=lambda i: bm25.score(q, i), reverse=True)
    chosen, used = [], 0
    for i in scored:
        if used + len(passages[i]) > max_chars and chosen:
            continue
        chosen.append(i)
        used += len(passages[i])
        if used >= max_chars:
            break
    if not chosen:
        chosen = [0]
    return " … ".join(passages[i] for i in sorted(chosen))


@dataclass
class Chunk:
    source: str
    text: str
    score: float = 0.0


class ChunkIndex:
    """BM25 index over passages from many documents (the document vault)."""

    def __init__(self, documents: Iterable[tuple[str, str]], target_chars: int = 700):
        self.chunks: list[Chunk] = []
        for name, text in documents:
            for passage in split_passages(text, target_chars=target_chars):
                self.chunks.append(Chunk(source=name, text=passage))
        self._bm25 = BM25([analyze(c.text) for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 6, per_source: int = 3) -> list[Chunk]:
        q = analyze(query)
        if not q or not self.chunks:
            return []
        scored = [(self._bm25.score(q, i), i) for i in range(len(self.chunks))]
        scored.sort(reverse=True)
        results, counts = [], Counter()
        for score, i in scored:
            if score <= 0:
                break
            chunk = self.chunks[i]
            if counts[chunk.source] >= per_source:
                continue
            counts[chunk.source] += 1
            results.append(Chunk(chunk.source, chunk.text, score))
            if len(results) >= k:
                break
        return results
