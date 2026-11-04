from local_perplex.local_perplex_core import ResearchEngine
from local_perplex.retrieval import BM25, ChunkIndex, analyze, best_passages, split_passages


def test_analyze_drops_stopwords_and_stems():
    assert analyze("What are the models learning?") == ["model", "learn"]


def test_bm25_prefers_rare_matching_terms():
    docs = [analyze(t) for t in ["the cat sat on the mat", "quantum computing with qubits",
                                 "the dog sat on the log"]]
    bm25 = BM25(docs)
    scores = bm25.scores(analyze("qubits quantum"))
    assert scores.index(max(scores)) == 1
    assert scores[0] == 0


def test_bm25_length_normalisation():
    short = analyze("transformers attention")
    long = analyze("transformers attention " + "filler words here " * 50)
    bm25 = BM25([short, long])
    assert bm25.score(analyze("attention"), 0) > bm25.score(analyze("attention"), 1)


def test_split_passages_respects_target_size():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))
    passages = split_passages(text, target_chars=400)
    assert len(passages) > 10
    assert all(len(p) < 1000 for p in passages)


def test_best_passages_finds_the_needle():
    hay = " ".join(f"Filler sentence {i} about gardening and soil." for i in range(300))
    text = hay + " The Eiffel Tower is 330 metres tall. " + hay
    picked = best_passages("How tall is the Eiffel Tower?", text, max_chars=700)
    assert "330 metres" in picked
    assert len(picked) <= 1500


def test_chunk_index_limits_chunks_per_source():
    docs = [("a.md", "Neural networks learn representations. " * 80),
            ("b.md", "Neural networks are trained with backpropagation. " * 5)]
    hits = ChunkIndex(docs, target_chars=300).search("neural networks", k=6, per_source=2)
    assert {h.source for h in hits} == {"a.md", "b.md"}
    assert sum(1 for h in hits if h.source == "a.md") <= 2


def test_ranker_dedupes_and_orders():
    ranked = ResearchEngine().rank_sources("python graph memory", [
        {"title": "Cooking", "url": "https://example.com/food", "content": "bread and soup"},
        {"title": "Python Graph Memory", "url": "https://www.example.com/python/", "content": "python graph memory"},
        {"title": "dup", "url": "https://example.com/python", "content": "python graph memory"},
    ])
    assert len(ranked) == 2
    assert ranked[0].title == "Python Graph Memory"
    assert 0 < ranked[0].relevance <= 1
    assert ranked[1].relevance < ranked[0].relevance
