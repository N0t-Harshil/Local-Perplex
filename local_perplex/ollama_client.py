"""Backward-compatible import path for the Ollama client.

The implementation now lives in :mod:`local_perplex.providers.ollama`.
"""
from __future__ import annotations

from .providers import PROVIDERS
from .providers.ollama import OllamaProvider


class OllamaClient(OllamaProvider):
    def __init__(self, base_url: str = "http://localhost:11434", max_retries: int = 3,
                 base_delay: float = 1.0):
        super().__init__(PROVIDERS["ollama"], base_url=base_url,
                         max_retries=max_retries, base_delay=base_delay)
