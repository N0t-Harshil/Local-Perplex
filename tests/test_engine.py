from conftest import FakeLLM, make_engine

from local_perplex.engine import _clean_query, is_greeting
from local_perplex.search import SearchEngine, html_to_text


def test_greetings():
    assert is_greeting("hii!") and is_greeting("Good morning") and is_greeting("thanks")
    assert not is_greeting("hello world program in rust")


def test_plan_query_rules_skip_the_llm(engine):
    assert engine.plan_query("latest iPhone price") == (True, "latest iPhone price")
    assert engine.plan_query("write a poem", focus_mode="Writing")[0] is False
    assert engine.plan_query("explain monads", mode="all_references")[0] is True
    assert engine.llm.calls == []  # none of these needed a routing call


def test_plan_query_uses_one_llm_call(engine):
    assert engine.plan_query("Who designed the Python language?") == (True, "python history")
    assert len(engine.llm.calls) == 1
    no_search = make_engine(llm=FakeLLM(route="ANSWER"))
    assert no_search.plan_query("What is 2 + 2?")[0] is False


def test_research_events_and_numbered_context(engine):
    events = list(engine.research_events("Who designed the Python language?"))
    kinds = [k for k, _ in events]
    assert kinds[0] == "status" and kinds[-1] == "result" and "sources" in kinds
    _, selected, context = events[-1][1]
    assert selected[0].title == "History of Python"
    assert "[1] History of Python" in context and "[2]" in context
    assert engine.last_metrics["searched"] and engine.last_metrics["sources"] == len(selected)


def test_deep_research_runs_planned_queries(engine):
    seen = []
    engine.search_engine.search = lambda q, num_results=12, focus_mode="All": seen.append(q) or []
    list(engine.research_events("Is Python good?", mode="pro"))
    assert len(seen) == 3


def test_answer_streams_with_stats(engine):
    answer = engine.ask("q", "[1] ctx", sources=[object()])
    assert "[1]" in answer
    assert engine.llm.last_stats.tokens > 0


def test_finalize_saves_history_graph_and_followups(engine):
    _, selected, ctx = engine.research_step("Who designed the Python language?")
    related = engine.finalize_research("Who designed Python?", "Guido van Rossum [1].", selected,
                                       background=False)
    assert len(related) == 3
    assert engine.history.get_all_sessions()[0]["question"] == "Who designed Python?"
    assert engine.graph.stats()["kinds"].get("concept", 0) >= 1


def test_incognito_saves_nothing(engine):
    engine.finalize_research("secret", "answer", [], tag="[INCOGNITO]", background=False)
    assert engine.history.get_all_sessions() == []
    assert engine.graph.stats()["nodes"] == 0


def test_offline_engine_explains_itself(isolated):
    off = make_engine(online=False)
    assert "not reachable" in off.ask("q", "")
    assert off.plan_query("anything")[0] is False


def test_clean_query():
    assert _clean_query("Can you tell me how vaccines work?") == "how vaccines work"


def test_html_to_text_drops_boilerplate():
    html = """<html><body><nav>Home About Login</nav><article><h1>Title</h1>
    <p>This is the real article paragraph with enough words to count.</p></article>
    <footer>Copyright footer links</footer><script>evil()</script></body></html>"""
    text = html_to_text(html)
    assert "real article paragraph" in text
    assert "Login" not in text and "evil" not in text and "Copyright" not in text


def test_search_backend_fallback_and_dedupe(isolated):
    eng = SearchEngine()
    eng.settings.search_provider = "brave"   # no key configured -> falls back to DuckDuckGo
    eng._duckduckgo = lambda q, n: [{"title": "A", "url": "https://a.example", "snippet": "x"},
                                    {"title": "A again", "url": "https://a.example", "snippet": "x"}]
    hits = eng.find("anything")
    assert [h["url"] for h in hits] == ["https://a.example"]


def test_writing_focus_never_searches(isolated):
    assert SearchEngine().search("poem", focus_mode="Writing") == []
