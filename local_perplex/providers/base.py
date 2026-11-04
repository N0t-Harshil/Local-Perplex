"""Common interface every LLM backend implements."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised for any provider failure. Subclasses RuntimeError so callers can
    keep catching ``RuntimeError`` as they did with the original Ollama client."""


@dataclass
class GenerationStats:
    """Timing for the most recent generation, surfaced in the UI and benchmarks."""

    ttft: float = 0.0          # seconds until the first token arrived
    total: float = 0.0         # seconds for the whole generation
    tokens: int = 0            # output tokens (exact when the API reports it)
    tokens_per_sec: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ttft": round(self.ttft, 3),
            "total": round(self.total, 3),
            "tokens": self.tokens,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
        }


@dataclass
class ProviderSpec:
    """Static description of a provider preset (shown in Settings)."""

    id: str
    label: str
    kind: str                     # "ollama" | "openai" | "anthropic"
    base_url: str
    local: bool
    env_key: str | None = None    # env var holding the API key
    default_model: str = ""
    needs_key: bool = True
    editable_url: bool = False    # only local / custom endpoints may change URL
    docs: str = ""
    notes: str = ""
    models: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "base_url": self.base_url,
            "local": self.local,
            "env_key": self.env_key,
            "default_model": self.default_model,
            "needs_key": self.needs_key,
            "editable_url": self.editable_url,
            "docs": self.docs,
            "notes": self.notes,
        }


class LLMProvider:
    """Base class. Subclasses implement ``_complete`` and ``_stream``."""

    spec: ProviderSpec

    def __init__(self, spec: ProviderSpec, api_key: str = "", base_url: str = "",
                 max_retries: int = 2, base_delay: float = 0.8):
        self.spec = spec
        self.api_key = api_key or ""
        self.base_url = (base_url or spec.base_url).rstrip("/")
        self.session = requests.Session()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.last_stats = GenerationStats()
        self.last_latency = 0.0

    # ── public API ───────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or not self.spec.needs_key

    def chat(self, model: str, messages: list[dict], stream: bool = False,
             timeout: int = 600, **params):
        """Return the full reply (``stream=False``) or an iterator of text chunks."""
        if stream:
            return self._stream_wrapper(model, messages, timeout, params)
        start = time.perf_counter()
        try:
            text = self._complete(model, messages, timeout, params)
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"{self.spec.label} error: {exc}") from exc
        self.last_latency = time.perf_counter() - start
        return text

    def list_models(self, timeout: float = 5.0) -> list[str]:
        return list(self.spec.models)

    def health(self, timeout: float = 3.0) -> bool:
        try:
            return bool(self.list_models(timeout=timeout)) or self.configured
        except Exception:  # noqa: BLE001 - health checks never raise
            return False

    # ── subclass hooks ───────────────────────────────────────────────
    def _complete(self, model, messages, timeout, params) -> str:  # pragma: no cover
        raise NotImplementedError

    def _stream(self, model, messages, timeout, params) -> Iterator[tuple[str, int | None]]:  # pragma: no cover
        """Yield ``(text, reported_token_count_or_None)``."""
        raise NotImplementedError

    # ── helpers ──────────────────────────────────────────────────────
    def _stream_wrapper(self, model, messages, timeout, params) -> Iterator[str]:
        start = time.perf_counter()
        first = None
        chunks = 0
        reported = None
        try:
            for text, tokens in self._stream(model, messages, timeout, params):
                if tokens is not None:
                    reported = tokens
                if not text:
                    continue
                if first is None:
                    first = time.perf_counter() - start
                chunks += 1
                yield text
        except (requests.exceptions.RequestException, json.JSONDecodeError, ProviderError) as exc:
            logger.error("%s stream error: %s", self.spec.label, exc)
            yield f"\n[Error from {self.spec.label}: {exc}]"
        finally:
            total = time.perf_counter() - start
            tokens = reported if reported else chunks
            gen_time = max(total - (first or 0.0), 1e-6)
            self.last_stats = GenerationStats(
                ttft=first or total, total=total, tokens=tokens,
                tokens_per_sec=tokens / gen_time if tokens else 0.0,
            )
            self.last_latency = total

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, requests.exceptions.Timeout):
            return False
        if isinstance(exc, requests.exceptions.HTTPError):
            status = exc.response.status_code if exc.response is not None else 0
            return status == 429 or status >= 500
        return isinstance(exc, (requests.exceptions.ConnectionError,
                                requests.exceptions.ChunkedEncodingError))

    def _post(self, url: str, payload: dict, headers: dict | None = None,
              timeout: float = 600, stream: bool = False) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, headers=headers or {},
                                         timeout=timeout, stream=stream)
                if resp.status_code >= 400:
                    detail = _error_detail(resp)
                    err = requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code}: {detail}", response=resp)
                    raise err
                return resp
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries and self._should_retry(exc):
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.4)
                    logger.warning("%s attempt %d failed (%s); retrying in %.1fs",
                                   self.spec.label, attempt + 1, exc, delay)
                    time.sleep(delay)
                    continue
                break
        assert last_exc is not None
        raise last_exc


def _error_detail(resp: requests.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return (resp.text or "")[:200]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)[:200]
    if err:
        return str(err)[:200]
    return str(data)[:200]


def iter_sse(resp: requests.Response) -> Iterator[str]:
    """Yield the ``data:`` payloads of a server-sent-events response."""
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            return
        yield data
