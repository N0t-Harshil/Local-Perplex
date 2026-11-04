"""The research pipeline: route → search → read → rank → retrieve → synthesize.

``LocalPerplex`` is shared by the CLI and the web UI. Every stage is timed so
the UI can show where the seconds went, and every LLM call goes through a
provider adapter so the same pipeline runs on Ollama, LM Studio, OpenAI,
Claude, Gemini, Groq and 20 other backends.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Iterator

import requests

from .config import Settings, load_settings, save_settings
from .documents import DocumentManager
from .graph import KnowledgeGraph
from .history import HistoryManager
from .local_perplex_core import RankedSource, ResearchEngine
from .providers import PROVIDERS, LLMProvider, OllamaProvider, build_provider
from .retrieval import best_passages
from .search import SearchEngine

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "llama3.2:3b"
SMALL_FALLBACK_MODEL = "llama3.2:1b"
MAX_CONTEXT_CHARS = 14000
LOCAL_CONTEXT_CHARS = 6000
STATUS_TTL = 20.0  # seconds between remote provider health checks

GREETING_RE = re.compile(
    r"^(hi+|hey+|hello+|howdy|sup|yo|what'?s? ?up|good\s*(morning|afternoon|evening)|thanks?( you)?|thx)"
    r"\s*[.!?]*\s*$", re.IGNORECASE)
RECENCY_RE = re.compile(
    r"\b(latest|today|tonight|yesterday|this (week|month|year)|current(ly)?|right now|news|price|"
    r"stock|weather|score|released?|announce[sd]?|update[sd]?|20[2-3]\d|recent(ly)?|upcoming|"
    r"who won|election|deadline|schedule)\b", re.IGNORECASE)
LLM_ERRORS = (RuntimeError, requests.exceptions.RequestException)


def is_greeting(text: str) -> bool:
    return bool(GREETING_RE.match((text or "").strip()))


class LocalPerplex:
    def __init__(self, model: str | None = None, provider: str | None = None,
                 settings: Settings | None = None, autostart: bool | None = None):
        self.settings = settings or load_settings()
        if provider:
            self.settings.provider = provider
        if self.settings.provider not in PROVIDERS:
            logger.warning("Unknown provider %r, using ollama", self.settings.provider)
            self.settings.provider = "ollama"

        self.search_engine = SearchEngine(self.settings)
        self.ranking_engine = ResearchEngine()
        self.history = HistoryManager()
        self.docs = DocumentManager()
        self.graph = KnowledgeGraph()

        self.llm: LLMProvider = build_provider(self.settings.provider, self.settings)
        self.ollama = self.llm if isinstance(self.llm, OllamaProvider) else build_provider("ollama", self.settings)
        self.model = model or self.settings.model or self.llm.spec.default_model or FALLBACK_MODEL
        self.llm_online = False
        self.available_models: list[str] = []
        self.ollama_process = None
        self.model_warm = False
        self.last_metrics: dict = {}
        self._status_checked = 0.0
        self._research_count = 0

        if autostart is None:
            autostart = os.getenv("LOCAL_PERPLEX_NO_AUTOSTART", "") not in ("1", "true", "yes")
        self._validate_connection()
        if not self.llm_online and self.provider_id == "ollama" and autostart:
            self._auto_start_ollama()
        if self.llm_online and self.provider_id == "ollama" and autostart:
            threading.Thread(target=self._warm_model, daemon=True).start()

    # ── compatibility aliases ────────────────────────────────────────
    @property
    def ollama_online(self) -> bool:
        return self.llm_online

    @ollama_online.setter
    def ollama_online(self, value: bool) -> None:
        self.llm_online = value

    @property
    def provider_id(self) -> str:
        return self.llm.spec.id

    @property
    def provider_label(self) -> str:
        return self.llm.spec.label

    @property
    def is_local(self) -> bool:
        return self.llm.spec.local

    # ── LLM helpers ──────────────────────────────────────────────────
    def _chat(self, messages: list[dict], timeout: int = 120, **params) -> str:
        params.setdefault("temperature", self.settings.temperature)
        return self.llm.chat(self.model, messages, timeout=timeout, **params)

    def _warm_model(self):
        try:
            self.llm.chat(self.model, [{"role": "user", "content": "hi"}], timeout=600,
                          max_tokens=1)
            self.model_warm = True
        except LLM_ERRORS as exc:
            logger.warning("Model warm-up skipped: %s", exc)

    def _supports_vision(self) -> bool:
        name = self.model.lower()
        if not self.is_local:
            return any(k in name for k in ("gpt-4o", "gpt-4.1", "gpt-5", "claude", "gemini",
                                           "vision", "pixtral", "llama-4"))
        return any(k in name for k in ("vision", "llava", "vl", "gemma3", "minicpm-v", "moondream",
                                       "qwen2.5vl", "llama4"))

    # ── routing ──────────────────────────────────────────────────────
    def plan_query(self, question: str, mode: str = "industry_standard",
                   focus_mode: str = "All") -> tuple[bool, str]:
        """Decide whether to search the web and with which query.

        Cheap rules settle most questions; only ambiguous ones cost a single
        LLM call (which also rewrites the query), down from two calls before.
        """
        q = question.strip()
        if is_greeting(q) or focus_mode == "Writing":
            return False, q
        if not self.llm_online:
            return False, q
        if focus_mode != "All" or mode in ("all_references", "pro") or RECENCY_RE.search(q):
            return True, _clean_query(q)
        try:
            reply = self._chat([
                {"role": "system", "content": (
                    "You route questions for a research assistant. If answering well needs fresh or "
                    "specific facts from the web (news, prices, statistics, niche facts, products, "
                    "people, events, anything after your training data), reply exactly:\n"
                    "SEARCH: <web search query, max 10 words>\n"
                    "If it is casual chat, an opinion, creative writing, math, coding help or stable "
                    "general knowledge, reply exactly:\nANSWER")},
                {"role": "user", "content": q},
            ], timeout=60, temperature=0, max_tokens=24)
        except LLM_ERRORS:
            return True, _clean_query(q)
        m = re.search(r"SEARCH:\s*(.+)", reply, re.IGNORECASE)
        if m:
            query = m.group(1).strip().strip('"\'').splitlines()[0]
            return True, query if len(query) > 3 else _clean_query(q)
        if "ANSWER" in reply.upper():
            return False, q
        return True, _clean_query(q)

    def should_search_web(self, question: str) -> bool:
        return self.plan_query(question)[0]

    def improve_search_query(self, question: str) -> str:
        return self.plan_query(question, mode="all_references")[1]

    # ── research ─────────────────────────────────────────────────────
    def research_events(self, question: str, mode: str = "industry_standard",
                        image_b64: str | None = None, focus_mode: str = "All") -> Iterator[tuple]:
        """Run retrieval, yielding ``("status", str)``, ``("sources", list)`` and
        finally ``("result", (question, selected_sources, context))``."""
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        if image_b64 and self._supports_vision():
            yield "status", "Looking at your image…"
            try:
                vision = self._chat([{
                    "role": "user", "images": [image_b64],
                    "content": (f"Describe this image in the context of: '{question}'. List the key "
                                "objects and entities, transcribe and translate any visible text to "
                                "English, and state what should be researched.")}], timeout=300)
                question = f"{question}\n\n[Image analysis: {vision.strip()}]"
            except LLM_ERRORS as exc:
                logger.warning("Vision step failed: %s", exc)
            timings["vision"] = time.perf_counter() - t_start

        t = time.perf_counter()
        if mode == "pro":
            yield "status", "Planning research steps…"
            queries = self._plan_deep_queries(question)
            needs_search = True
        else:
            yield "status", "Understanding the question…"
            needs_search, query = self.plan_query(question, mode, focus_mode)
            queries = [query]
        timings["route"] = time.perf_counter() - t

        selected: list[RankedSource] = []
        if needs_search:
            t = time.perf_counter()
            label = queries[0] if len(queries) == 1 else f"{len(queries)} queries in parallel"
            yield "status", f"Searching the web for {label}…"
            raw = self._search_many(queries, focus_mode)
            timings["search"] = time.perf_counter() - t
            yield "status", f"Read {len(raw)} pages. Ranking…"
            t = time.perf_counter()
            ranked = self.ranking_engine.rank_sources(question if mode == "pro" else queries[0], [
                {"title": s["title"], "url": s["url"], "content": s["content"],
                 "category": s.get("category", "")} for s in raw])
            limit = {"industry_standard": 6, "all_references": 10, "pro": 10}.get(mode, 6)
            selected = ranked[:limit]
            timings["rank"] = time.perf_counter() - t
            if selected:
                yield "sources", selected

        t = time.perf_counter()
        context = self.build_context(question, selected)
        timings["retrieve"] = time.perf_counter() - t
        timings["research_total"] = time.perf_counter() - t_start
        self.last_metrics = {"timings": {k: round(v, 3) for k, v in timings.items()},
                             "queries": queries if needs_search else [],
                             "sources": len(selected),
                             "searched": needs_search,
                             "search_backend": self.search_engine.backend}
        yield "result", (question, selected, context)

    def research_step(self, question: str, mode: str = "industry_standard",
                      image_b64: str | None = None, focus_mode: str = "All"):
        """Blocking version of :meth:`research_events` (used by the CLI)."""
        result = (question, [], "")
        for kind, payload in self.research_events(question, mode, image_b64, focus_mode):
            if kind == "result":
                result = payload
        return result

    def deep_research_iterative(self, question: str, mode: str = "pro", focus_mode: str = "All"):
        """Back-compat wrapper: deep research as status + result events."""
        for kind, payload in self.research_events(question, "pro", None, focus_mode):
            if kind in ("status", "result"):
                yield kind, payload

    def _plan_deep_queries(self, question: str) -> list[str]:
        try:
            raw = self._chat([
                {"role": "system", "content": (
                    "You are a research planner. Write 3 distinct web search queries that together "
                    "cover the question from different angles (definition/background, specifics/data, "
                    "recent developments or criticism). Output only the queries, one per line.")},
                {"role": "user", "content": question}], timeout=90, max_tokens=90)
            queries = [re.sub(r"^[\s\-*\d.)]+", "", line).strip().strip('"')
                       for line in raw.splitlines()]
            queries = [q for q in queries if len(q) > 3][:3]
        except LLM_ERRORS:
            queries = []
        return queries or [_clean_query(question)]

    def _search_many(self, queries: list[str], focus_mode: str) -> list[dict]:
        if len(queries) == 1:
            return self.search_engine.search(queries[0], focus_mode=focus_mode)
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            batches = list(pool.map(lambda q: self.search_engine.search(q, num_results=8,
                                                                        focus_mode=focus_mode), queries))
        seen, merged = set(), []
        for batch in batches:
            for s in batch:
                if s["url"] not in seen:
                    seen.add(s["url"])
                    merged.append(s)
        return merged

    def build_context(self, question: str, sources: list) -> str:
        """Assemble the prompt context within the character budget.

        Web sources are numbered ``[1]..[n]`` in the same order the UI shows
        them, so citations in the answer always point at the right card.
        """
        budget = int(self.settings.max_context_chars or
                     (LOCAL_CONTEXT_CHARS if self.is_local else MAX_CONTEXT_CHARS))
        parts: list[str] = []

        graph_ctx = self.graph.context_for_query(question)
        if graph_ctx:
            parts.append(graph_ctx[:budget // 8])
        doc_ctx = self.docs.get_local_context(question, max_chars=budget // 4)
        if doc_ctx:
            parts.append(f"--- YOUR DOCUMENTS ---\n{doc_ctx}")

        if sources:
            remaining = max(budget - sum(len(p) for p in parts), budget // 2)
            per_source = max(400, min(2400, remaining // len(sources)))
            blocks = []
            for i, s in enumerate(sources, 1):
                content = getattr(s, "content", "") or ""
                passage = best_passages(question, content, max_chars=per_source)
                blocks.append(f"[{i}] {getattr(s, 'title', '')} ({getattr(s, 'url', '')})\n{passage}")
            parts.append("--- WEB SOURCES ---\n" + "\n\n".join(blocks))

        context = "\n\n".join(parts)
        if len(context) > budget:
            context = context[:budget] + "\n[Context truncated]"
        return context

    # ── synthesis ────────────────────────────────────────────────────
    def system_prompt(self, has_context: bool, sources: list | None, mode: str, followup: bool) -> str:
        lines = [
            f"You are Local Perplex, a precise research assistant. Today is {date.today():%B %d, %Y}.",
            "Answer directly in the first sentence, then add the detail that matters. "
            "Use Markdown: short paragraphs, bullet lists or tables when they help, and **bold** for key terms.",
        ]
        if sources:
            lines.append(
                "Ground every factual claim in the numbered WEB SOURCES and cite them inline like [1] or [2][3], "
                "placed right after the claim. Only cite numbers that exist. If the sources disagree or don't "
                "cover something, say so instead of guessing.")
        elif has_context:
            lines.append("Use the provided context where relevant; mention which document a fact came from.")
        if mode in ("all_references", "pro"):
            lines.append("Be thorough: cover background, key facts, nuances and open questions, using headings.")
        if followup:
            lines.append("This continues an earlier conversation; build on it naturally.")
        lines.append("If the question or image contains another language, translate and explain it.")
        return "\n".join(lines)

    def ask_stream(self, question: str, context: str, history: list | None = None,
                   mode: str = "industry_standard", sources: list | None = None) -> Iterator[str]:
        if not self.llm_online:
            yield (f"\n[{self.provider_label} is not reachable. "
                   + ("Start it with `ollama serve` and try again.]" if self.provider_id == "ollama"
                      else "Check the API key and base URL in Settings.]"))
            return
        has_context = bool(context and context.strip())
        messages = [{"role": "system", "content": self.system_prompt(has_context, sources, mode, bool(history))}]
        for h in (history or [])[-6:]:
            messages.append({"role": "user", "content": h.get("question", "")})
            messages.append({"role": "assistant", "content": h.get("answer", "")})
        if has_context:
            # Small local models follow the instruction nearest the question far more
            # reliably than one in the system prompt, so restate the citation rule here.
            reminder = ("\n\nAnswer from the sources above. Put the source number in square brackets, "
                        "like [1] or [2][3], right after each fact you take from it." if sources else "")
            messages.append({"role": "user", "content": f"{context}\n\n---\nQuestion: {question}{reminder}"})
        else:
            messages.append({"role": "user", "content": question})
        yield from self.llm.chat(self.model, messages, stream=True, timeout=900,
                                 temperature=self.settings.temperature)
        self.last_metrics["generation"] = self.llm.last_stats.as_dict()

    def ask(self, question: str, context: str, history: list | None = None,
            mode: str = "industry_standard", sources: list | None = None) -> str:
        return "".join(self.ask_stream(question, context, history, mode, sources))

    def generate_chat_title(self, question: str) -> str:
        fallback = question if len(question) <= 50 else question[:47].rstrip() + "…"
        if not self.llm_online:
            return fallback
        try:
            title = self._chat([
                {"role": "system", "content": "Write a 2-6 word title for a chat that starts with this "
                                              "question. Output only the title, no quotes."},
                {"role": "user", "content": question[:500]}], timeout=60, max_tokens=16).strip().strip('"\'*#')
            title = title.splitlines()[0] if title else ""
            return title if 2 < len(title) <= 80 else fallback
        except LLM_ERRORS:
            return fallback

    # ── after the answer ─────────────────────────────────────────────
    def finalize_research(self, question: str, answer: str, selected: list,
                          tag: str = "General", background: bool = True) -> list[str]:
        """Persist the session, grow the knowledge graph, return follow-ups.

        Graph concept extraction runs in a background thread so the UI gets
        its "done" event as soon as follow-up questions are ready.
        """
        if tag != "[INCOGNITO]":
            payloads = [self._source_payload(s) for s in selected]
            try:
                self.history.save_session(question, answer, payloads, tag=tag,
                                          metrics=self.last_metrics)
            except OSError as exc:
                logger.warning("Could not save history: %s", exc)
            job = threading.Thread(target=self._record_graph, args=(question, answer, selected, tag),
                                   daemon=True)
            job.start()
            if not background:
                job.join()

        if not self.llm_online or not answer.strip():
            return []
        try:
            raw = self._chat([
                {"role": "system", "content": "Suggest 3 short, specific follow-up questions a curious "
                                              "reader would ask next. One per line, no numbering."},
                {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer[:3000]}"}],
                timeout=90, max_tokens=110)
        except LLM_ERRORS as exc:
            logger.warning("Follow-up generation failed: %s", exc)
            return []
        out = []
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-*\d.)]+", "", line).strip().strip('"')
            if len(line) > 8 and line not in out:
                out.append(line)
        return out[:3]

    def _record_graph(self, question, answer, selected, tag):
        concepts = None
        if self.llm_online and answer:
            try:
                concepts = self.graph.extract_concepts(question, answer, lambda p: self._chat([
                    {"role": "system", "content": "You extract semantic concepts from research. "
                                                  "Follow the format exactly."},
                    {"role": "user", "content": p}], timeout=90, max_tokens=220))
            except LLM_ERRORS:
                concepts = None
        try:
            self.graph.record_research(question, answer, selected, tag=tag, concepts=concepts)
            self._research_count += 1
            if self._research_count % 10 == 0:
                self.graph.decay_weights()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph update failed: %s", exc)

    @staticmethod
    def _source_payload(source) -> dict:
        get = source.get if isinstance(source, dict) else lambda k, d=None: getattr(source, k, d)
        return {
            "title": get("title", "Unknown"),
            "url": get("url", ""),
            "relevance": get("relevance", 0.0),
            "category": get("category", "General"),
            "snippet": (get("content", "") or get("snippet", "") or "")[:300],
        }

    # ── provider management ──────────────────────────────────────────
    def switch_provider(self, provider_id: str, model: str | None = None, persist: bool = True) -> dict:
        self.settings.provider = provider_id
        self.llm = build_provider(provider_id, self.settings)
        if isinstance(self.llm, OllamaProvider):
            self.ollama = self.llm
        self.model = model or self.llm.spec.default_model or self.model
        self.settings.model = self.model
        self._status_checked = 0.0
        self._validate_connection()
        if persist:
            save_settings(self.settings)
        return self.refresh_status(force=False)

    def _validate_connection(self, timeout: float = 3.0) -> None:
        self._status_checked = time.monotonic()
        auth_failed = False
        try:
            models = self.llm.list_models(timeout=timeout)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            auth_failed = status in (401, 403)
            logger.warning("%s rejected the request: %s", self.provider_label, exc)
            models = []
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s not reachable: %s", self.provider_label, exc)
            models = []
        if isinstance(self.llm, OllamaProvider):
            chat_models = [m for m in models if OllamaProvider.is_chat_model(m)]
            self.available_models = chat_models
            self.llm_online = bool(chat_models)
            if self.llm_online:
                self._ensure_model()
            elif models or self.llm.health(timeout):
                logger.warning("Ollama has no chat models; pulling %s", SMALL_FALLBACK_MODEL)
                threading.Thread(target=self._download_small_model, daemon=True).start()
        else:
            self.available_models = models
            # Hosted APIs: online when a key is configured and the endpoint answered,
            # or when it's a keyless local server that listed its models.
            self.llm_online = bool(models) or (self.llm.configured and not auth_failed
                                               and not self.llm.spec.local
                                               and self._remote_reachable(timeout))
            if models and self.model not in models and self.llm.spec.local:
                self.model = models[0]

    def _remote_reachable(self, timeout: float) -> bool:
        try:
            requests.head(self.llm.base_url, timeout=timeout)
            return True
        except requests.exceptions.RequestException:
            return False

    _validate_ollama_connection = _validate_connection  # back-compat

    def _ensure_model(self) -> None:
        if self.model in self.available_models:
            return
        if f"{self.model}:latest" in self.available_models:
            self.model = f"{self.model}:latest"
            return
        logger.warning("Model %r not installed. Available: %s", self.model, ", ".join(self.available_models))
        sized = [(m, OllamaProvider.estimate_model_size(m)) for m in self.available_models]
        # Prefer the smallest capable model (3-8B) for CPU speed, then any sized model.
        for lo, hi in ((2.5, 8.5), (0.1, 1000)):
            options = sorted((s, m) for m, s in sized if lo <= s <= hi)
            if options:
                self.model = options[0][1]
                return
        self.model = self.available_models[0]

    def _download_small_model(self) -> None:
        if self.ollama.pull_model(SMALL_FALLBACK_MODEL):
            self.model = SMALL_FALLBACK_MODEL
            self._validate_connection()

    @staticmethod
    def _find_ollama():
        path = shutil.which("ollama")
        if path:
            return path
        for cp in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                   Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe",
                   Path("/usr/local/bin/ollama"), Path("/opt/homebrew/bin/ollama")):
            if cp.exists():
                return str(cp)
        return None

    def _auto_start_ollama(self) -> None:
        ollama_bin = self._find_ollama()
        if not ollama_bin:
            logger.warning("Ollama not found. Install it from https://ollama.com or pick another provider.")
            return
        try:
            kwargs = {}
            if os.name == "nt":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                kwargs = {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}
            self.ollama_process = subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.DEVNULL, **kwargs)
            for _ in range(30):
                time.sleep(1)
                self._validate_connection()
                if self.llm_online:
                    logger.info("Ollama started")
                    return
            logger.warning("Ollama started but is not responding after 30s")
        except OSError as exc:
            logger.warning("Failed to start Ollama: %s", exc)

    def refresh_status(self, force: bool = True) -> dict:
        """Re-check the backend. Local servers are cheap to ping; hosted APIs are
        re-checked at most every ``STATUS_TTL`` seconds even when forced."""
        age = time.monotonic() - self._status_checked
        if (force and self.is_local) or age > (2.0 if self.is_local else STATUS_TTL):
            self._validate_connection()
        return {
            "ollama_online": self.llm_online,
            "llm_online": self.llm_online,
            "provider": self.provider_id,
            "provider_label": self.provider_label,
            "local": self.is_local,
            "available_models": self.available_models,
            "current_model": self.model,
            "search_backend": self.search_engine.backend,
        }

    def graph_stats(self) -> dict:
        return self.graph.stats()

    def index_document(self, filename: str) -> None:
        content = self.docs.extract_document_text(self.docs.doc_dir / filename)
        if content:
            self.graph.record_document(filename, content)


_FILLER_RE = re.compile(r"^(please\s+)?(can|could|would) you (please\s+)?(tell me|explain|describe|"
                        r"show me|find|search( for)?)\s+|^(tell me|explain|describe)\s+", re.IGNORECASE)


def _clean_query(question: str) -> str:
    q = question.strip().split("\n\n[Image analysis:")[0]
    q = _FILLER_RE.sub("", q).strip().rstrip("?").strip()
    return q[:200] or question[:200]
