"""Native Ollama API (``/api/chat``), the default fully-local backend."""
from __future__ import annotations

import json
import re
from typing import Iterator

from .base import LLMProvider

# Ollama's historical default context is 2048 tokens, which silently truncates
# retrieval-augmented prompts. 4096 fits the ~6k-character local budget plus the
# answer, without the slower prompt processing of a bigger window on CPU.
DEFAULT_NUM_CTX = 4096
KEEP_ALIVE = "30m"   # keep weights resident between questions


class OllamaProvider(LLMProvider):
    def _payload(self, model, messages, stream, params) -> dict:
        options = {"num_ctx": params.pop("num_ctx", DEFAULT_NUM_CTX)}
        if "max_tokens" in params:
            options["num_predict"] = params.pop("max_tokens")
        options.update(params)
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "keep_alive": KEEP_ALIVE,
        }

    def _complete(self, model, messages, timeout, params) -> str:
        resp = self._post(f"{self.base_url}/api/chat",
                          self._payload(model, messages, False, dict(params)), timeout=timeout)
        return (resp.json().get("message") or {}).get("content", "")

    def _stream(self, model, messages, timeout, params) -> Iterator[tuple[str, int | None]]:
        resp = self._post(f"{self.base_url}/api/chat",
                          self._payload(model, messages, True, dict(params)),
                          timeout=timeout, stream=True)
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                yield f"\n[Ollama error: {chunk['error']}]", None
                return
            text = (chunk.get("message") or {}).get("content", "")
            tokens = chunk.get("eval_count") if chunk.get("done") else None
            yield text, tokens
            if chunk.get("done"):
                return

    def list_models(self, timeout: float = 3.0) -> list[str]:
        resp = self.session.get(f"{self.base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]

    def health(self, timeout: float = 3.0) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def pull_model(self, model_name: str, timeout: int = 600) -> bool:
        try:
            resp = self.session.post(f"{self.base_url}/api/pull", json={"model": model_name},
                                     stream=True, timeout=timeout)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and json.loads(line).get("status") == "success":
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    @staticmethod
    def estimate_model_size(model_name: str) -> float:
        """Parameter count in billions guessed from the tag, e.g. ``llama3.2:3b`` -> 3."""
        m = re.search(r"[:\-.](\d+(?:\.\d+)?)\s*b\b", model_name.lower())
        return float(m.group(1)) if m else 0.0

    @staticmethod
    def is_chat_model(model_name: str) -> bool:
        name = model_name.lower()
        return not any(k in name for k in ("embed", "bge-", "minilm", "rerank"))
