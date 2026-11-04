"""User configuration for Local Perplex.

Settings are resolved in this order (highest priority first):

1. Explicit arguments (CLI flags such as ``--provider`` / ``--model``)
2. Environment variables (``LOCAL_PERPLEX_PROVIDER``, ``OPENAI_API_KEY``, ...)
3. The JSON config file (``~/.local_perplex/config.json`` or ``$LOCAL_PERPLEX_CONFIG``)
4. Built-in defaults (Ollama + DuckDuckGo, fully local, no keys needed)

API keys are stored only in the user's home directory, never in the project
tree, and are never returned in full by the web API.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_PROVIDER = "ollama"
DEFAULT_SEARCH_PROVIDER = "duckduckgo"


def config_path() -> Path:
    override = os.getenv("LOCAL_PERPLEX_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".local_perplex" / "config.json"


@dataclass
class Settings:
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    # Per-provider overrides: {"openai": {"api_key": "...", "base_url": "..."}}
    providers: dict = field(default_factory=dict)
    search_provider: str = DEFAULT_SEARCH_PROVIDER
    # Per-search-provider overrides: {"brave": {"api_key": "..."}, "searxng": {"base_url": "..."}}
    search: dict = field(default_factory=dict)
    temperature: float = 0.3
    # 0 = automatic: ~6k characters for local models (fast prompt processing on
    # CPU), ~14k for hosted APIs with large context windows.
    max_context_chars: int = 0

    # ── provider credentials ──────────────────────────────────────────
    def provider_value(self, provider_id: str, key: str, env_var: str | None = None) -> str:
        if env_var and os.getenv(env_var):
            return os.getenv(env_var, "")
        return (self.providers.get(provider_id) or {}).get(key, "") or ""

    def set_provider_value(self, provider_id: str, key: str, value: str) -> None:
        self.providers.setdefault(provider_id, {})[key] = value

    def search_value(self, provider_id: str, key: str, env_var: str | None = None) -> str:
        if env_var and os.getenv(env_var):
            return os.getenv(env_var, "")
        return (self.search.get(provider_id) or {}).get(key, "") or ""

    def set_search_value(self, provider_id: str, key: str, value: str) -> None:
        self.search.setdefault(provider_id, {})[key] = value


_lock = threading.Lock()


def load_settings() -> Settings:
    path = config_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    known = {k: v for k, v in data.items() if k in Settings.__dataclass_fields__}
    settings = Settings(**known)
    settings.provider = os.getenv("LOCAL_PERPLEX_PROVIDER", settings.provider) or DEFAULT_PROVIDER
    settings.model = os.getenv("LOCAL_PERPLEX_MODEL", settings.model) or settings.model
    settings.search_provider = (
        os.getenv("LOCAL_PERPLEX_SEARCH", settings.search_provider) or DEFAULT_SEARCH_PROVIDER
    )
    return settings


def save_settings(settings: Settings) -> Path:
    path = config_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path


def mask_secret(value: str) -> str:
    """Show only enough of a key to recognise it: ``sk-…a1b2``."""
    if not value:
        return ""
    if len(value) <= 8:
        return "••••"
    return f"{value[:3]}…{value[-4:]}"
