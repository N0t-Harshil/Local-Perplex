"""Retrieval quality on BEIR SciFact (5,183 abstracts, 300 test queries).

Compares the original v1 scorer (normalised term frequency) with the v2
BM25 ranker used by Local Perplex, using the standard BEIR metrics.

    python benchmarks/bench_retrieval.py            # downloads ~2.7 MB once

Writes benchmarks/results/retrieval.json.
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from local_perplex.local_perplex_core import calculate_score, tokenize  # noqa: E402
from local_perplex.retrieval import BM25, analyze  # noqa: E402

DATA = ROOT / ".data" / "scifact"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download() -> None:
    if (DATA / "corpus.jsonl").exists():
        return
    print("Downloading SciFact from the BEIR mirror…")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            # extract only the three data files, never arbitrary paths
            base = Path(name).name
            if base in ("corpus.jsonl", "queries.jsonl", "test.tsv"):
                (DATA / base).parent.mkdir(parents=True, exist_ok=True)
                (DATA / base).write_bytes(zf.read(name))


def load():
    corpus = {}
    for line in (DATA / "corpus.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        corpus[d["_id"]] = f"{d.get('title', '')} {d.get('text', '')}"
    queries = {}
    for line in (DATA / "queries.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        queries[d["_id"]] = d["text"]
    qrels: dict[str, dict[str, int]] = {}
    for i, line in enumerate((DATA / "test.tsv").open(encoding="utf-8")):
        if i == 0:
            continue
        qid, did, score = line.strip().split("\t")
        qrels.setdefault(qid, {})[did] = int(score)
    return corpus, {q: queries[q] for q in qrels}, qrels


def ndcg_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(name, rank_fn, queries, qrels):
    metrics = {"ndcg@10": 0.0, "recall@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}
    t0 = time.perf_counter()
    for qid, text in queries.items():
        ranked = rank_fn(text)
        rel = qrels[qid]
        relevant = {d for d, s in rel.items() if s > 0}
        metrics["ndcg@10"] += ndcg_at(ranked, rel, 10)
        metrics["recall@10"] += len(relevant & set(ranked[:10])) / len(relevant)
        metrics["recall@100"] += len(relevant & set(ranked[:100])) / len(relevant)
        metrics["mrr@10"] += next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in relevant), 0.0)
    n = len(queries)
    out = {k: round(v / n, 4) for k, v in metrics.items()}
    out["ms_per_query"] = round((time.perf_counter() - t0) * 1000 / n, 1)
    print(f"{name:<28} nDCG@10 {out['ndcg@10']:.3f}  R@100 {out['recall@100']:.3f}  "
          f"MRR@10 {out['mrr@10']:.3f}  {out['ms_per_query']} ms/query")
    return out


def main() -> dict:
    download()
    corpus, queries, qrels = load()
    ids = list(corpus)
    print(f"SciFact: {len(ids)} documents, {len(queries)} queries\n")

    # v1: score every document with the original term-frequency formula
    def v1(q):
        scores = [(calculate_score(q, corpus[d]), d) for d in ids]
        scores.sort(reverse=True)
        return [d for _, d in scores[:100]]

    # v1 with pre-tokenised docs, so the comparison isn't just tokenisation cost
    t0 = time.perf_counter()
    index_tokens = [analyze(corpus[d]) for d in ids]
    bm25 = BM25(index_tokens)
    build_s = time.perf_counter() - t0

    def v2(q):
        qt = analyze(q)
        scores = bm25.scores(qt)
        top = sorted(range(len(ids)), key=scores.__getitem__, reverse=True)[:100]
        return [ids[i] for i in top]

    results = {
        "dataset": "BEIR SciFact (test)", "documents": len(ids), "queries": len(queries),
        "v1_term_frequency": evaluate("v1 term-frequency scorer", v1, queries, qrels),
        "v2_bm25": evaluate("v2 BM25 + stemming", v2, queries, qrels),
        "v2_index_build_s": round(build_s, 2),
        "tokens_indexed": sum(len(t) for t in index_tokens),
        "_unused_tokenizer_check": len(tokenize("ok")),
    }
    del results["_unused_tokenizer_check"]
    a, b = results["v1_term_frequency"]["ndcg@10"], results["v2_bm25"]["ndcg@10"]
    results["ndcg_relative_gain_pct"] = round((b - a) / a * 100, 1) if a else None
    print(f"\nnDCG@10 improvement: {results['ndcg_relative_gain_pct']}%")
    out = ROOT / "results" / "retrieval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
