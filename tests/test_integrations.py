"""Search backends, document parsing, mind map and CLI, all offline."""
from click.testing import CliRunner

from conftest import make_engine

from local_perplex.config import Settings
from local_perplex.documents import DocumentManager
from local_perplex.mindmap import MindMapGenerator
from local_perplex.search import SearchEngine


class Resp:
    def __init__(self, payload=None, content=b""):
        self.payload, self.content, self.status_code = payload, content, 200

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


def engine_with(backend, **values):
    s = Settings(search_provider=backend)
    for k, v in values.items():
        s.set_search_value(backend, k, v)
    return SearchEngine(s)


def test_brave_parsing(isolated):
    eng = engine_with("brave", api_key="b")
    seen = {}
    eng.session.get = lambda url, params=None, headers=None, timeout=None: seen.update(h=headers) or Resp(
        {"web": {"results": [{"title": "T", "url": "https://t.example", "description": "<b>bold</b> text"}]}})
    hits = eng._brave("q", 5)
    assert hits == [{"title": "T", "url": "https://t.example", "snippet": "bold text"}]
    assert seen["h"]["X-Subscription-Token"] == "b"


def test_tavily_serper_searxng_wikipedia(isolated):
    tav = engine_with("tavily", api_key="t")
    tav.session.post = lambda url, json=None, headers=None, timeout=None: Resp(
        {"results": [{"title": "A", "url": "https://a.example", "content": "c", "raw_content": "full page"}]})
    assert tav._tavily("q", 3)[0]["content"] == "full page"

    ser = engine_with("serper", api_key="s")
    ser.session.post = lambda url, json=None, headers=None, timeout=None: Resp(
        {"organic": [{"title": "G", "link": "https://g.example", "snippet": "s"}]})
    assert ser._serper("q", 3)[0]["url"] == "https://g.example"

    sx = engine_with("searxng", base_url="http://localhost:8888")
    sx.session.get = lambda url, params=None, timeout=None: Resp(
        {"results": [{"title": "X", "url": "https://x.example", "content": "c"}]})
    assert sx._searxng("q", 3)[0]["title"] == "X"

    wk = engine_with("wikipedia")
    wk.session.get = lambda url, params=None, timeout=None: Resp(
        {"query": {"search": [{"title": "Alan Turing", "snippet": "<span>math</span>"}]}})
    assert wk._wikipedia("q", 3)[0]["url"].endswith("/wiki/Alan_Turing")


def test_arxiv_atom_parsing(isolated):
    atom = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/1706.03762</id><title>Attention Is All You Need</title>
      <summary>The dominant sequence transduction models...</summary>
      <author><name>Ashish Vaswani</name></author></entry></feed>"""
    eng = engine_with("arxiv")
    eng.session.get = lambda url, params=None, timeout=None: Resp(content=atom)
    hit = eng._arxiv("transformers", 2)[0]
    assert hit["category"] == "Academic" and "Vaswani" in hit["content"]


def test_read_pages_scores_and_drops_thin_pages(isolated):
    eng = SearchEngine()
    eng._fetch_text = lambda url: "Python is a programming language. " * 30 if "good" in url else ""
    out = eng.read_pages([{"title": "Good", "url": "https://good.example", "snippet": ""},
                          {"title": "Thin", "url": "https://thin.example", "snippet": "x"}], "python language")
    assert [s["title"] for s in out] == ["Good"]
    assert out[0]["relevance"] > 0


def test_document_formats(isolated, tmp_path):
    import docx
    import openpyxl
    dm = DocumentManager(tmp_path / "docs")
    d = docx.Document()
    d.add_paragraph("Mitochondria are the powerhouse of the cell.")
    d.save(dm.doc_dir / "bio.docx")
    wb = openpyxl.Workbook()
    wb.active.append(["metric", "value"])
    wb.active.append(["revenue", 4200])
    wb.save(dm.doc_dir / "sales.xlsx")
    (dm.doc_dir / "page.html").write_text("<html><body><article><p>Volcanic soils are unusually fertile "
                                          "for agriculture.</p></article></body></html>", encoding="utf-8")
    assert "powerhouse" in dm.extract_document_text(dm.doc_dir / "bio.docx")
    assert "revenue, 4200" in dm.extract_document_text(dm.doc_dir / "sales.xlsx")
    assert "powerhouse" in dm.get_local_context("what do mitochondria do")
    assert "Volcanic" in dm.get_local_context("why are volcanic soils fertile")
    assert dm.stats()["files"] == 3
    # cached parse: same object returned while the file is unchanged
    first = dm.index()
    assert dm.index() is first


def test_mindmap_renders_graph(engine):
    engine.finalize_research("How do vaccines work?", "Vaccines train the immune system.", [],
                             background=False)
    mm = MindMapGenerator()
    data = mm.load_from_graph(engine.graph)
    assert data["stats"]["nodes"] > 0
    assert mm.to_mermaid_mindmap(data).startswith("mindmap")
    assert "graph" in mm.to_mermaid_graph(data)


def test_cli_end_to_end(isolated, monkeypatch):
    import local_perplex.ui.cli as cli
    monkeypatch.setattr(cli, "LocalPerplex", lambda model=None, provider=None: make_engine())
    result = CliRunner().invoke(cli.main, ["Who", "created", "Python?", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert "Guido" in result.output and "sources" in result.output


def test_cli_lists_providers_and_handles_offline(isolated, monkeypatch):
    import local_perplex.ui.cli as cli
    out = CliRunner().invoke(cli.main, ["--list-providers"]).output
    assert "anthropic" in out and "searxng" in out
    monkeypatch.setattr(cli, "LocalPerplex", lambda model=None, provider=None: make_engine(online=False))
    assert CliRunner().invoke(cli.main, ["hello world question"]).exit_code == 1
