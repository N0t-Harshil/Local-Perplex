"""Anthropic Messages API (Claude models)."""
from __future__ import annotations

import json
from typing import Iterator

from .base import LLMProvider, ProviderError, iter_sse

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts, out = [], []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m.get("content", ""))
            continue
        images = m.get("images") or []
        if images:
            content = [{"type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64}}
                       for b64 in images]
            content.append({"type": "text", "text": m.get("content", "")})
        else:
            content = m.get("content", "")
        # The API requires alternating roles; merge consecutive same-role turns.
        if out and out[-1]["role"] == m["role"] and isinstance(content, str) \
                and isinstance(out[-1]["content"], str):
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": m["role"], "content": content})
    return "\n\n".join(p for p in system_parts if p), out


class AnthropicProvider(LLMProvider):
    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _payload(self, model, messages, stream, params) -> dict:
        system, msgs = _split_system(messages)
        payload = {
            "model": model,
            "max_tokens": params.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": msgs,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if "temperature" in params:
            payload["temperature"] = params["temperature"]
        return payload

    def _complete(self, model, messages, timeout, params) -> str:
        if not self.api_key:
            raise ProviderError("Anthropic needs an API key (ANTHROPIC_API_KEY).")
        resp = self._post(f"{self.base_url}/v1/messages",
                          self._payload(model, messages, False, params),
                          headers=self._headers(), timeout=timeout)
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def _stream(self, model, messages, timeout, params) -> Iterator[tuple[str, int | None]]:
        if not self.api_key:
            raise ProviderError("Anthropic needs an API key (ANTHROPIC_API_KEY).")
        resp = self._post(f"{self.base_url}/v1/messages",
                          self._payload(model, messages, True, params),
                          headers=self._headers(), timeout=timeout, stream=True)
        for data in iter_sse(resp):
            event = json.loads(data)
            etype = event.get("type")
            if etype == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield delta.get("text", ""), None
            elif etype == "message_delta":
                yield "", (event.get("usage") or {}).get("output_tokens")
            elif etype == "error":
                raise ProviderError(str(event.get("error")))

    def list_models(self, timeout: float = 5.0) -> list[str]:
        if not self.api_key:
            return []
        resp = self.session.get(f"{self.base_url}/v1/models", headers=self._headers(),
                                timeout=timeout)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", []) if m.get("id")]
