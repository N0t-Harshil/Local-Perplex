"""Any server that speaks the OpenAI Chat Completions API.

This one adapter covers OpenAI itself plus Groq, Mistral, DeepSeek, OpenRouter,
Together, Fireworks, xAI, Perplexity, Cerebras, Gemini, Cohere, Hugging Face,
NVIDIA NIM, SambaNova, DeepInfra and local servers such as LM Studio,
llama.cpp, vLLM, LocalAI, Jan, KoboldCpp and text-generation-webui.
"""
from __future__ import annotations

import json
from typing import Iterator

from .base import LLMProvider, ProviderError, iter_sse


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        images = m.get("images") or []
        if not images:
            out.append({"role": m["role"], "content": m.get("content", "")})
            continue
        parts = [{"type": "text", "text": m.get("content", "")}]
        for b64 in images:
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{b64}"}})
        out.append({"role": m["role"], "content": parts})
    return out


class OpenAICompatibleProvider(LLMProvider):
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.spec.id == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/N0t-Harshil/Local-Perplex"
            headers["X-Title"] = "Local Perplex"
        return headers

    def _payload(self, model, messages, stream, params) -> dict:
        payload = {"model": model, "messages": _to_openai_messages(messages), "stream": stream}
        if "temperature" in params:
            payload["temperature"] = params["temperature"]
        if "max_tokens" in params:
            payload["max_tokens"] = params["max_tokens"]
        if stream and not self.spec.local:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _complete(self, model, messages, timeout, params) -> str:
        if not self.configured:
            raise ProviderError(f"{self.spec.label} needs an API key ({self.spec.env_key}).")
        resp = self._post(f"{self.base_url}/chat/completions",
                          self._payload(model, messages, False, params),
                          headers=self._headers(), timeout=timeout)
        data = resp.json()
        try:
            return data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected response from {self.spec.label}: {data}") from exc

    def _stream(self, model, messages, timeout, params) -> Iterator[tuple[str, int | None]]:
        if not self.configured:
            raise ProviderError(f"{self.spec.label} needs an API key ({self.spec.env_key}).")
        resp = self._post(f"{self.base_url}/chat/completions",
                          self._payload(model, messages, True, params),
                          headers=self._headers(), timeout=timeout, stream=True)
        for data in iter_sse(resp):
            chunk = json.loads(data)
            usage = chunk.get("usage") or {}
            tokens = usage.get("completion_tokens")
            choices = chunk.get("choices") or []
            text = ""
            if choices:
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
            yield text, tokens

    def list_models(self, timeout: float = 5.0) -> list[str]:
        if not self.configured:
            return []
        resp = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", data if isinstance(data, list) else [])
        names = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name")
                if name:
                    names.append(str(name).removeprefix("models/"))
        return sorted(set(names))
