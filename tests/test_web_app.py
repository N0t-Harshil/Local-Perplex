import re

from fastapi.testclient import TestClient

from conftest import sse_events


def test_pages_render(web):
    client = TestClient(web.app)
    for path in ("/", "/graph", "/vault", "/settings", "/privacy", "/history", "/healthz"):
        assert client.get(path).status_code == 200, path


def test_status_reports_provider(web):
    data = TestClient(web.app).get("/api/status").json()
    assert data["llm_online"] is True
    assert data["ollama_online"] is True
    assert "graph" in data and "provider_label" in data


def test_ask_offline_returns_page_not_422(web):
    web.engine.llm_online = False
    response = TestClient(web.app).post("/ask", data={"question": "hello"})
    assert response.status_code == 200
    assert "reachable" in response.text and "Ollama" in response.text


def test_ask_then_stream_full_research_flow(web):
    client = TestClient(web.app)
    page = client.post("/ask", data={"question": "Who created Python?", "focus_mode": "All"})
    assert page.status_code == 200
    job = re.search(r"/stream-answer\?job=([0-9a-f]+)", page.text).group(1)

    events = sse_events(client.get(f"/stream-answer?job={job}").text)
    statuses = [e["status"] for e in events if "status" in e]
    sources = next(e["update_sources"] for e in events if "update_sources" in e)
    answer = "".join(e["chunk"] for e in events if "chunk" in e)
    done = events[-1]

    assert any("Searching" in s for s in statuses)
    assert sources[0]["n"] == 1 and "wikipedia" in sources[0]["domain"]
    assert "Bread" not in sources[0]["title"]
    assert "[1]" in answer
    assert done["done"] is True and len(done["related"]) == 3
    assert done["metrics"]["generation"]["tokens"] > 0
    assert "route" in done["metrics"]["timings"]

    # the conversation is persisted and listed in the sidebar
    sessions = client.get("/api/sessions").json()["sessions"]
    assert sessions and sessions[0]["message_count"] == 1
    assert client.get(f"/chat/{done['session_id']}").status_code == 200
    # a job can only be streamed once
    assert "expired" in client.get(f"/stream-answer?job={job}").text


def test_incognito_is_not_listed_or_saved(web):
    client = TestClient(web.app)
    page = client.post("/ask", data={"question": "secret topic", "privacy_mode": "true"})
    job = re.search(r"job=([0-9a-f]+)", page.text).group(1)
    sse_events(client.get(f"/stream-answer?job={job}").text)
    assert client.get("/api/sessions").json()["sessions"] == []
    assert web.engine.history.get_all_sessions() == []


def test_greeting_skips_research(web):
    client = TestClient(web.app)
    job = re.search(r"job=([0-9a-f]+)", client.post("/ask", data={"question": "hi"}).text).group(1)
    events = sse_events(client.get(f"/stream-answer?job={job}").text)
    assert not any("update_sources" in e for e in events)
    assert events[-1]["done"]


def test_cross_origin_writes_are_blocked(web):
    client = TestClient(web.app)
    evil = client.post("/api/settings", json={"provider": "openai", "base_url": "https://evil.example"},
                       headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    same = client.post("/api/clear-graph", headers={"Origin": "http://testserver"})
    assert same.status_code == 200


def test_settings_api_saves_key_without_echoing_it(web, monkeypatch):
    client = TestClient(web.app)
    monkeypatch.setattr(web.engine, "switch_provider",
                        lambda pid, model=None, persist=True: {"provider": pid, "llm_online": False})
    r = client.post("/api/settings", json={"provider": "groq", "model": "llama-3.3-70b-versatile",
                                           "api_key": "gsk_supersecretvalue1234"})
    assert r.status_code == 200
    catalog = client.get("/api/providers").json()["providers"]
    groq = next(p for p in catalog if p["id"] == "groq")
    assert groq["has_key"] and groq["key_hint"].endswith("1234")
    assert "supersecret" not in client.get("/api/providers").text


def test_settings_rejects_bad_input(web):
    client = TestClient(web.app)
    assert client.post("/api/settings", json={"provider": "nope"}).status_code == 400
    assert client.post("/api/settings", json={"provider": "lmstudio", "base_url": "file:///etc"}).status_code == 400


def test_upload_and_search_documents(web):
    client = TestClient(web.app)
    files = [("files", ("notes.md", b"Quantum annealing is used for optimisation problems.", "text/markdown")),
             ("files", ("evil.exe", b"MZ", "application/octet-stream"))]
    assert client.post("/upload-docs", files=files, follow_redirects=False).status_code == 303
    names = [f.name for f in web.engine.docs.files()]
    assert names == ["notes.md"]
    assert "Quantum annealing" in web.engine.docs.get_local_context("what is quantum annealing")


def test_api_ask_json(web):
    data = TestClient(web.app).post("/api/ask", json={"question": "Who created Python?"}).json()
    assert data["answer"].startswith("Python was created")
    assert len(data["sources"]) >= 2
