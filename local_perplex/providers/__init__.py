"""LLM provider registry.

26 presets: 8 local runtimes, 17 hosted APIs and a custom OpenAI-compatible
endpoint. Three adapters (Ollama, OpenAI-compatible, Anthropic) cover them all.
"""
from __future__ import annotations

from ..config import Settings
from .anthropic import AnthropicProvider
from .base import GenerationStats, LLMProvider, ProviderError, ProviderSpec
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatibleProvider

__all__ = [
    "PROVIDERS", "GenerationStats", "LLMProvider", "ProviderError", "ProviderSpec",
    "OllamaProvider", "OpenAICompatibleProvider", "AnthropicProvider",
    "build_provider", "get_spec", "provider_catalog",
]


def _local(id, label, url, model="", kind="openai", notes="", docs=""):
    return ProviderSpec(id=id, label=label, kind=kind, base_url=url, local=True,
                        needs_key=False, editable_url=True, default_model=model,
                        notes=notes, docs=docs)


def _cloud(id, label, url, env, model, kind="openai", notes="", docs=""):
    return ProviderSpec(id=id, label=label, kind=kind, base_url=url, local=False,
                        env_key=env, default_model=model, notes=notes, docs=docs)


_SPECS: list[ProviderSpec] = [
    # ── local runtimes: nothing leaves the machine ────────────────────
    _local("ollama", "Ollama", "http://localhost:11434", "llama3.2:3b", kind="ollama",
           notes="Default. Auto-started if installed.", docs="https://ollama.com"),
    _local("lmstudio", "LM Studio", "http://localhost:1234/v1",
           notes="Start the local server from the Developer tab.", docs="https://lmstudio.ai"),
    _local("llamacpp", "llama.cpp server", "http://localhost:8080/v1",
           notes="llama-server -m model.gguf --port 8080",
           docs="https://github.com/ggml-org/llama.cpp"),
    _local("vllm", "vLLM", "http://localhost:8001/v1",
           notes="vllm serve <model> --port 8001", docs="https://docs.vllm.ai"),
    _local("localai", "LocalAI", "http://localhost:8080/v1", docs="https://localai.io"),
    _local("jan", "Jan", "http://localhost:1337/v1", docs="https://jan.ai"),
    _local("koboldcpp", "KoboldCpp", "http://localhost:5001/v1",
           docs="https://github.com/LostRuins/koboldcpp"),
    _local("textgen", "text-generation-webui", "http://localhost:5000/v1",
           notes="Launch with --api.", docs="https://github.com/oobabooga/text-generation-webui"),
    # ── hosted APIs ───────────────────────────────────────────────────
    _cloud("openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini",
           docs="https://platform.openai.com"),
    _cloud("anthropic", "Anthropic Claude", "https://api.anthropic.com", "ANTHROPIC_API_KEY",
           "claude-haiku-5-5", kind="anthropic", docs="https://console.anthropic.com"),
    _cloud("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
           "GEMINI_API_KEY", "gemini-2.5-flash", docs="https://aistudio.google.com"),
    _cloud("groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
           "llama-3.3-70b-versatile", docs="https://console.groq.com"),
    _cloud("mistral", "Mistral AI", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
           "mistral-small-latest", docs="https://console.mistral.ai"),
    _cloud("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY",
           "deepseek-chat", docs="https://platform.deepseek.com"),
    _cloud("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
           "openrouter/auto", notes="One key, hundreds of models.", docs="https://openrouter.ai"),
    _cloud("together", "Together AI", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
           "meta-llama/Llama-3.3-70B-Instruct-Turbo", docs="https://api.together.ai"),
    _cloud("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1",
           "FIREWORKS_API_KEY", "accounts/fireworks/models/llama-v3p3-70b-instruct",
           docs="https://fireworks.ai"),
    _cloud("xai", "xAI Grok", "https://api.x.ai/v1", "XAI_API_KEY", "grok-3-mini",
           docs="https://console.x.ai"),
    _cloud("perplexity", "Perplexity Sonar", "https://api.perplexity.ai", "PERPLEXITY_API_KEY",
           "sonar", docs="https://docs.perplexity.ai"),
    _cloud("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
           "llama-3.3-70b", docs="https://cloud.cerebras.ai"),
    _cloud("cohere", "Cohere", "https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY",
           "command-a-03-2025", docs="https://dashboard.cohere.com"),
    _cloud("huggingface", "Hugging Face Inference", "https://router.huggingface.co/v1",
           "HF_TOKEN", "meta-llama/Llama-3.1-8B-Instruct", docs="https://huggingface.co/settings/tokens"),
    _cloud("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
           "meta/llama-3.3-70b-instruct", docs="https://build.nvidia.com"),
    _cloud("sambanova", "SambaNova Cloud", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY",
           "Meta-Llama-3.3-70B-Instruct", docs="https://cloud.sambanova.ai"),
    _cloud("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY",
           "meta-llama/Llama-3.3-70B-Instruct", docs="https://deepinfra.com"),
    # ── anything else that speaks the OpenAI API ─────────────────────
    ProviderSpec(id="custom", label="Custom (OpenAI-compatible)", kind="openai",
                 base_url="http://localhost:8000/v1", local=False, needs_key=False,
                 editable_url=True, env_key="CUSTOM_OPENAI_API_KEY",
                 notes="Any /v1/chat/completions endpoint, e.g. Azure, a gateway or a proxy."),
]

PROVIDERS: dict[str, ProviderSpec] = {s.id: s for s in _SPECS}
_ADAPTERS = {"ollama": OllamaProvider, "openai": OpenAICompatibleProvider,
             "anthropic": AnthropicProvider}


def get_spec(provider_id: str) -> ProviderSpec:
    try:
        return PROVIDERS[provider_id]
    except KeyError as exc:
        raise ProviderError(f"Unknown provider '{provider_id}'. "
                            f"Choose one of: {', '.join(PROVIDERS)}") from exc


def build_provider(provider_id: str, settings: Settings | None = None,
                   api_key: str | None = None, base_url: str | None = None) -> LLMProvider:
    spec = get_spec(provider_id)
    settings = settings or Settings()
    key = api_key if api_key is not None else settings.provider_value(spec.id, "api_key", spec.env_key)
    url = base_url or ""
    if not url and spec.editable_url:
        url = settings.provider_value(spec.id, "base_url",
                                      "OLLAMA_HOST" if spec.id == "ollama" else None)
        if spec.id == "ollama" and url:
            if not url.startswith("http"):
                url = f"http://{url}"
            url = url.replace("://0.0.0.0", "://127.0.0.1")  # a bind address, not a destination
    return _ADAPTERS[spec.kind](spec, api_key=key, base_url=url)


def provider_catalog(settings: Settings | None = None) -> list[dict]:
    from ..config import mask_secret
    settings = settings or Settings()
    out = []
    for spec in _SPECS:
        key = settings.provider_value(spec.id, "api_key", spec.env_key)
        item = spec.public()
        item["has_key"] = bool(key)
        item["key_hint"] = mask_secret(key)
        item["base_url"] = (settings.provider_value(spec.id, "base_url") if spec.editable_url
                            else "") or spec.base_url
        out.append(item)
    return out
