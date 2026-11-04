import importlib
import json

import pytest

from local_perplex.providers import PROVIDERS
from local_perplex.providers.base import LLMProvider


class FakeLLM(LLMProvider):
    """Deterministic provider: replies are chosen by inspecting the system prompt."""

    def __init__(self, answer="Python was created by Guido van Rossum [1]. It is popular [2].",
                 route="SEARCH: python history"):
        super().__init__(PROVIDERS["ollama"])
        self.answer, self.route, self.calls = answer, route, []

    def list_models(self, timeout=5):
        return ["fake-model:3b"]

    def _reply(self, messages):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        self.calls.append(system[:40])
        if "route questions" in system:
            return self.route
        if "research planner" in system:
            return "python origins\npython adoption statistics\npython criticism"
        if "follow-up" in system:
            return "Who maintains Python today?\nWhat is Python 3.13?\nWhy is Python slow?"
        if "title" in system:
            return "Python history"
        if "extract semantic concepts" in system:
            return "Python language:is_a->Programming language\nGuido van Rossum"
        return self.answer

    def _complete(self, model, messages, timeout, params):
        return self._reply(messages)

    def _stream(self, model, messages, timeout, params):
        words = self._reply(messages).split(" ")
        for i, w in enumerate(words):
            yield (w if i == 0 else " " + w), None
        yield "", len(words)


FAKE_PAGES = [
    {"title": "History of Python", "url": "https://en.wikipedia.org/wiki/History_of_Python",
     "content": "Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands. "
                "Its implementation began in December 1989. " * 5, "relevance": 0.5},
    {"title": "Python popularity", "url": "https://example.com/popularity",
     "content": "Python is one of the most popular programming languages according to many surveys. " * 5,
     "relevance": 0.4},
    {"title": "Bread recipes", "url": "https://example.com/bread",
     "content": "Knead the dough and bake the bread for forty minutes. " * 5, "relevance": 0.1},
]


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_PERPLEX_HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setenv("LOCAL_PERPLEX_DOCUMENT_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("LOCAL_PERPLEX_GRAPH_DIR", str(tmp_path / "graph"))
    monkeypatch.setenv("LOCAL_PERPLEX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LOCAL_PERPLEX_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("LOCAL_PERPLEX_NO_AUTOSTART", "1")
    for var in ("LOCAL_PERPLEX_MODEL", "LOCAL_PERPLEX_SEARCH", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # A keyless hosted provider never touches the network, so importing the web
    # app (which builds a real engine) is instant and works without Ollama.
    monkeypatch.setenv("LOCAL_PERPLEX_PROVIDER", "openai")
    return tmp_path


def make_engine(online=True, llm=None):
    from local_perplex.engine import LocalPerplex
    engine = LocalPerplex.__new__(LocalPerplex)
    from local_perplex.config import Settings
    from local_perplex.documents import DocumentManager
    from local_perplex.graph import KnowledgeGraph
    from local_perplex.history import HistoryManager
    from local_perplex.local_perplex_core import ResearchEngine
    from local_perplex.search import SearchEngine

    engine.settings = Settings()
    engine.search_engine = SearchEngine(engine.settings)
    engine.search_engine.search = lambda q, num_results=12, focus_mode="All": [dict(p) for p in FAKE_PAGES]
    engine.ranking_engine = ResearchEngine()
    engine.history, engine.docs, engine.graph = HistoryManager(), DocumentManager(), KnowledgeGraph()
    engine.llm = llm or FakeLLM()
    engine.ollama = engine.llm
    engine.model = "fake-model:3b"
    engine.llm_online = online
    engine.available_models = ["fake-model:3b"]
    engine.ollama_process = None
    engine.model_warm = True
    engine.last_metrics = {}
    engine._status_checked = 0.0
    engine._research_count = 0
    engine._validate_connection = lambda timeout=3.0: None
    return engine


@pytest.fixture
def engine(isolated):
    return make_engine()


@pytest.fixture
def web(isolated):
    """The FastAPI module with its engine replaced by a fake, fully offline."""
    module = importlib.import_module("local_perplex.ui.web.app")
    module = importlib.reload(module)
    module.engine = make_engine()
    return module


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
