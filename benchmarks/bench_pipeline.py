"""Pipeline micro-benchmarks, plus an optional live run against a real model.

    python benchmarks/bench_pipeline.py              # offline, deterministic
    python benchmarks/bench_pipeline.py --live       # + real web search and LLM
    python benchmarks/bench_pipeline.py --live --provider groq --model llama-3.3-70b-versatile

Writes benchmarks/results/pipeline.json (and live.json with --live).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import string
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from local_perplex.local_perplex_core import ResearchEngine, calculate_score  # noqa: E402
from local_perplex.retrieval import ChunkIndex, best_passages  # noqa: E402

random.seed(7)
WORDS = ["model", "graph", "memory", "python", "search", "vector", "latency", "token", "quantum",
         "energy", "battery", "protein", "market", "climate", "neural", "compiler", "storage"]


def text(n_words: int, inject: str = "") -> str:
    words = [random.choice(WORDS) if random.random() < 0.25 else
             "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9))) for _ in range(n_words)]
    if inject:
        words.insert(random.randint(0, len(words)), inject)
    sentences, out = [], []
    for i, w in enumerate(words, 1):
        out.append(w)
        if i % 14 == 0:
            sentences.append(" ".join(out).capitalize() + ".")
            out = []
    return " ".join(sentences + [" ".join(out)])


def timeit(fn, repeat=5):
    runs = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(runs), 2)


def offline() -> dict:
    res = {}
    query = "python graph memory latency"

    # 1) Ranking a realistic search result set: 12 pages of ~3,000 words.
    pages = [{"title": f"Page {i}", "url": f"https://site{i}.example/a", "content": text(3000)} for i in range(12)]
    ranker = ResearchEngine()
    res["rank_12_pages_ms"] = timeit(lambda: ranker.rank_sources(query, pages))

    # 2) Ranking throughput at scale: 10,000 documents of 500 words.
    big = [{"title": f"Doc {i}", "url": f"https://e.example/{i}", "content": text(500)} for i in range(10_000)]
    res["rank_10k_docs_ms"] = timeit(lambda: ranker.rank_sources(query, big), repeat=2)
    res["v1_rank_10k_docs_ms"] = timeit(lambda: [calculate_score(query, d["content"]) for d in big], repeat=2)

    # 3) Passage selection: needle in a 20,000-character page.
    hay = text(3200, inject="NEEDLE the eiffel tower is 330 metres tall")
    res["passage_select_20k_chars_ms"] = timeit(lambda: best_passages("how tall is the eiffel tower", hay, 1800))
    res["passage_finds_needle"] = "330 metres" in best_passages("how tall is the eiffel tower", hay, 1800)

    # 4) Document vault: ~1 MB of text across 40 files.
    docs = [(f"doc{i}.md", text(4000)) for i in range(40)]
    t = time.perf_counter()
    idx = ChunkIndex(docs)
    res["vault_index_build_ms"] = round((time.perf_counter() - t) * 1000, 1)
    res["vault_chars"] = sum(len(d) for _, d in docs)
    res["vault_chunks"] = len(idx)
    res["vault_query_ms"] = timeit(lambda: idx.search("neural compiler storage", k=6), repeat=20)

    # 5) Knowledge-graph recall after 200 recorded research sessions.
    os.environ["LOCAL_PERPLEX_GRAPH_DIR"] = tempfile.mkdtemp()
    from local_perplex.graph import KnowledgeGraph
    g = KnowledgeGraph()
    t = time.perf_counter()
    for i in range(200):
        topic = " ".join(random.sample(WORDS, 3))
        g.record_research(f"What about {topic} {i}?", text(120), [
            {"title": f"Source {i}-{j}", "url": f"https://s{i}.example/{j}", "content": text(80)} for j in range(4)])
    res["graph_record_session_ms"] = round((time.perf_counter() - t) * 1000 / 200, 1)
    res["graph_nodes"] = g.stats()["nodes"]
    res["graph_edges"] = g.stats()["edges"]
    res["graph_recall_ms"] = timeit(lambda: g.context_for_query("python graph memory"), repeat=10)
    return res


V1_PRE_RENDER_PROMPTS = [
    "Does this question NEED a web search to answer? Only say 'yes' if it asks about current events, real-time "
    "data, specific statistics, niche facts, or news. Say 'no' for opinions, casual chat, general knowledge, "
    "personal questions, hypotheticals, definitions, explanations, advice, or anything the model can answer from "
    "what it already knows. Reply ONLY 'yes' or 'no'.",
    "Rewrite the user's question into a concise, effective web search query (max 10 words). Output ONLY the "
    "query, nothing else.",
    "Generate a short title (max 6 words) for a chat conversation that starts with this question. Output ONLY "
    "the title.",
]

LIVE_QUESTIONS = [
    "What are the main new features in Python 3.13?",
    "How do solid-state batteries differ from lithium-ion batteries?",
    "What is retrieval-augmented generation?",
    "Who won the 2022 FIFA World Cup and where was it held?",
    "What causes the northern lights?",
]


def live(provider: str | None, model: str | None, always_search: bool = False) -> dict:
    from local_perplex.engine import LocalPerplex, _clean_query
    tmp = Path(tempfile.mkdtemp())
    for k in ("HISTORY", "DOCUMENT", "GRAPH"):
        os.environ[f"LOCAL_PERPLEX_{k}_DIR"] = str(tmp / k.lower())
    eng = LocalPerplex(model=model, provider=provider)
    if not eng.llm_online:
        raise SystemExit(f"{eng.provider_label} is not reachable")
    if always_search:  # measure grounded answers on every question, not just routed ones
        eng.plan_query = lambda q, mode="industry_standard", focus_mode="All": (True, _clean_query(q))
    eng.ask("warm up", "")  # load weights so the first question isn't penalised
    rows = []
    for q in LIVE_QUESTIONS:
        # v1 ran these three uncapped calls, one after another, before the answer page rendered.
        t = time.perf_counter()
        for system in ([] if always_search else V1_PRE_RENDER_PROMPTS):
            eng.llm.chat(eng.model, [{"role": "system", "content": system}, {"role": "user", "content": q}],
                         timeout=300)
        v1_llm_s = time.perf_counter() - t
        t0 = time.perf_counter()
        _, sources, ctx = eng.research_step(q)
        m = dict(eng.last_metrics)
        answer = eng.ask(q, ctx, sources=sources)
        gen = eng.llm.last_stats
        total = time.perf_counter() - t0
        cites = sum(answer.count(f"[{i}]") for i in range(1, len(sources) + 1))
        row = {"question": q, "searched": m["searched"], "sources": len(sources),
               "route_s": m["timings"].get("route", 0), "search_read_s": m["timings"].get("search", 0),
               "research_s": m["timings"]["research_total"], "ttft_s": round(gen.ttft, 2),
               "tokens": gen.tokens, "tokens_per_s": round(gen.tokens_per_sec, 1),
               "time_to_first_token_s": round(m["timings"]["research_total"] + gen.ttft, 2),
               "total_s": round(total, 2), "context_chars": len(ctx), "citations": cites,
               "v1_pre_render_llm_s": round(v1_llm_s, 2), "v2_route_llm_calls": 1 if m["timings"].get("route", 0) > 0.05 else 0}
        rows.append(row)
        print(f"  {q[:48]:<48} {row['sources']:>2} src  research {row['research_s']:5.1f}s  "
              f"first token {row['time_to_first_token_s']:5.1f}s  {row['tokens_per_s']:5.1f} tok/s  "
              f"{cites} citations  (v1 pre-render LLM {v1_llm_s:4.1f}s)")
    med = lambda k: round(statistics.median(r[k] for r in rows), 2)  # noqa: E731
    cited = [r for r in rows if r["sources"]]
    return {"provider": eng.provider_label, "model": eng.model, "always_search": always_search,
            "answers_with_citations_pct": round(100 * sum(1 for r in cited if r["citations"]) / len(cited))
            if cited else None,
            "questions": rows,
            "median": {k: med(k) for k in ("research_s", "route_s", "search_read_s", "time_to_first_token_s",
                                           "tokens_per_s", "total_s", "sources", "citations",
                                           "v1_pre_render_llm_s")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--always-search", action="store_true", help="force web search on every question")
    ap.add_argument("--skip-offline", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    machine = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}",
               "cpu": platform.processor() or platform.machine(), "cores": os.cpu_count()}
    if not args.skip_offline:
        res = {"machine": machine, **offline()}
        print(json.dumps(res, indent=2))
        (out_dir / "pipeline.json").write_text(json.dumps(res, indent=2))
    if args.live:
        print("\nLive run:")
        lv = {"machine": machine, **live(args.provider, args.model, args.always_search)}
        print(json.dumps(lv["median"], indent=2), "citation rate:", lv["answers_with_citations_pct"])
        name = "live_search.json" if args.always_search else "live.json"
        (out_dir / name).write_text(json.dumps(lv, indent=2))


if __name__ == "__main__":
    main()
