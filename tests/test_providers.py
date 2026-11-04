import json

import pytest

from local_perplex.config import Settings, load_settings, mask_secret, save_settings
from local_perplex.providers import PROVIDERS, ProviderError, build_provider, provider_catalog
from local_perplex.providers.anthropic import _split_system
from local_perplex.providers.ollama import OllamaProvider


class FakeResponse:
    def __init__(self, payload=None, lines=None, status=200):
        self.payload, self.lines, self.status_code, self.text = payload, lines or [], status, json.dumps(payload)

    def json(self):
        return self.payload

    def iter_lines(self, decode_unicode=False):
        for line in self.lines:
            yield line if decode_unicode else line.encode()

    def raise_for_status(self):
        pass


def capture(provider, response):
    sent = {}

    def post(url, json=None, headers=None, timeout=None, stream=False):
        sent.update(url=url, json=json, headers=headers or {})
        return response
    provider.session.post = post
    return sent


def test_catalog_has_local_and_hosted_presets():
    assert len(PROVIDERS) >= 25
    assert sum(p.local for p in PROVIDERS.values()) >= 8
    for pid in ("ollama", "lmstudio", "openai", "anthropic", "gemini", "groq", "openrouter", "deepseek"):
        assert pid in PROVIDERS


def test_openai_compatible_chat_and_stream():
    llm = build_provider("groq", api_key="k")
    sent = capture(llm, FakeResponse({"choices": [{"message": {"content": "hello"}}]}))
    assert llm.chat("m", [{"role": "user", "content": "hi"}]) == "hello"
    assert sent["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer k"

    lines = ['data: {"choices":[{"delta":{"content":"Hel"}}]}', 'data: {"choices":[{"delta":{"content":"lo"}}]}',
             'data: {"choices":[],"usage":{"completion_tokens":2}}', "data: [DONE]"]
    capture(llm, FakeResponse(lines=lines))
    assert "".join(llm.chat("m", [{"role": "user", "content": "hi"}], stream=True)) == "Hello"
    assert llm.last_stats.tokens == 2


def test_openai_images_become_content_parts():
    llm = build_provider("openai", api_key="k")
    sent = capture(llm, FakeResponse({"choices": [{"message": {"content": "a cat"}}]}))
    llm.chat("gpt-4o-mini", [{"role": "user", "content": "what?", "images": ["AAAA"]}])
    parts = sent["json"]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,AAAA")


def test_missing_key_raises_provider_error():
    with pytest.raises(ProviderError):
        build_provider("openai", api_key="").chat("m", [{"role": "user", "content": "x"}])


def test_anthropic_payload_and_stream():
    llm = build_provider("anthropic", api_key="sk-ant")
    lines = ['data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
             'data: {"type":"message_delta","usage":{"output_tokens":1}}']
    sent = capture(llm, FakeResponse(lines=lines))
    out = "".join(llm.chat("claude-x", [{"role": "system", "content": "be brief"},
                                        {"role": "user", "content": "hello"}], stream=True))
    assert out == "Hi"
    assert sent["json"]["system"] == "be brief"
    assert sent["headers"]["x-api-key"] == "sk-ant"
    assert all(m["role"] != "system" for m in sent["json"]["messages"])


def test_anthropic_merges_consecutive_roles():
    _, msgs = _split_system([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert msgs == [{"role": "user", "content": "a\n\nb"}]


def test_ollama_requests_large_context_and_reports_tokens():
    llm = build_provider("ollama")
    lines = [json.dumps({"message": {"content": "ok"}, "done": False}),
             json.dumps({"message": {"content": ""}, "done": True, "eval_count": 7})]
    sent = capture(llm, FakeResponse(lines=lines))
    assert "".join(llm.chat("llama3.2:3b", [{"role": "user", "content": "x"}], stream=True)) == "ok"
    assert sent["json"]["options"]["num_ctx"] >= 4096
    assert llm.last_stats.tokens == 7


def test_model_size_parsing():
    assert OllamaProvider.estimate_model_size("llama3.2:3b") == 3
    assert OllamaProvider.estimate_model_size("qwen3.5:0.8b") == 0.8
    assert OllamaProvider.estimate_model_size("qwen2.5-coder:14b") == 14
    assert not OllamaProvider.is_chat_model("nomic-embed-text:latest")


def test_settings_roundtrip_and_env_override(isolated, monkeypatch):
    s = Settings(provider="groq")
    s.set_provider_value("groq", "api_key", "abc12345678")
    save_settings(s)
    assert load_settings().provider_value("groq", "api_key") == "abc12345678"
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert load_settings().provider_value("groq", "api_key", "GROQ_API_KEY") == "from-env"
    monkeypatch.setenv("LOCAL_PERPLEX_PROVIDER", "mistral")
    assert load_settings().provider == "mistral"


def test_catalog_masks_keys(isolated):
    s = Settings()
    s.set_provider_value("openai", "api_key", "sk-proj-abcdefghijklmnop")
    entry = next(p for p in provider_catalog(s) if p["id"] == "openai")
    assert entry["key_hint"] == mask_secret("sk-proj-abcdefghijklmnop") == "sk-…mnop"
    assert "abcdefgh" not in json.dumps(provider_catalog(s))


def test_hosted_urls_cannot_be_overridden(isolated):
    s = Settings()
    s.set_provider_value("openai", "base_url", "https://evil.example/v1")
    assert build_provider("openai", s).base_url == "https://api.openai.com/v1"
    s.set_provider_value("lmstudio", "base_url", "http://10.0.0.5:1234/v1")
    assert build_provider("lmstudio", s).base_url == "http://10.0.0.5:1234/v1"
