from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .local_perplex_core import tokenize


STOPWORDS = {
    "about","after","again","against","also","and","any","are","because","been",
    "before","being","between","both","but","can","could","did","does","doing",
    "for","from","had","has","have","how","into","its","more","most","not","now",
    "off","only","other","our","out","over","own","same","should","such","than",
    "that","the","their","then","there","these","they","this","those","through",
    "too","under","very","was","were","what","when","where","which","while","who",
    "why","will","with","you","your",
}

# Semantic relation types that mirror how humans connect ideas
RELATION_TYPES = {
    "is_a": "hyponym/hypernym relationship",
    "part_of": "meronym/holonym relationship",
    "related_to": "general semantic association",
    "causes": "causal relationship",
    "leads_to": "temporal/causal chain",
    "opposes": "contradictory or opposing concepts",
    "supports": "supporting or reinforcing evidence",
    "example_of": "instance or example relationship",
    "used_in": "instrumental relationship",
    "informs": "one concept informs another",
    "follows_from": "logical consequence",
    "co_occurs_with": "frequently appears together",
    "elaborates": "provides detail or elaboration",
    "contrasts_with": "contrasting but not opposing",
}

# How much weight to add for different interaction types
WEIGHT_BOOSTS = {
    "direct_query": 2.0,
    "follow_up": 1.5,
    "co_occurrence": 0.3,
    "transitive": 0.15,
    "user_confirmed": 1.0,
    "decay_per_day": 0.02,
}

CONCEPT_EXTRACTION_PROMPT = """Extract 3-6 key concepts from the following research Q&A. Each concept should be a meaningful phrase (2-5 words). For each concept, also identify its relationship to other concepts if applicable.

Output format: one concept per line as "Concept Name:relation->Related Concept"
Use these relation types: is_a, part_of, related_to, causes, leads_to, opposes, supports, example_of, used_in, informs, follows_from, elaborates, contrasts_with

If a concept has no clear relation, just output "Concept Name"

Question: {question}
Answer: {answer}"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json(data: dict | None) -> str:
    return json.dumps(data or {}, ensure_ascii=True, sort_keys=True)


def _load_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _clip(value: str, limit: int) -> str:
    value = " ".join((value or "").split())
    return value[:limit]


def _item_get(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class KnowledgeGraph:
    """A growing, self-organizing knowledge graph inspired by human memory.

    Features:
    - Semantic concept nodes (not just words) extracted via LLM
    - Rich relationship types that mirror human thinking
    - Hebbian strengthening: concepts that fire together wire together
    - Spreading activation retrieval
    - Transitive inference: A→B→C creates A→C
    - Weight decay: unused connections slowly fade
    - Co-occurrence tracking for implicit relationships
    """

    def __init__(self, graph_dir: str | Path | None = None, db_path: str | Path | None = None):
        if db_path is None:
            if graph_dir is None:
                graph_dir = os.getenv("LOCAL_PERPLEX_GRAPH_DIR")
                if graph_dir is None:
                    project_root = Path(__file__).parent.parent
                    graph_dir = project_root / "graph"
            graph_dir = Path(graph_dir)
            graph_dir.mkdir(parents=True, exist_ok=True)
            db_path = graph_dir / "knowledge_graph.sqlite3"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL")  # durable enough under WAL, far fewer fsyncs
        return conn

    @contextmanager
    def _connect(self):
        """Yield a connection: the open batch transaction if there is one,
        otherwise a short-lived connection that commits and closes."""
        shared = getattr(self._local, "conn", None)
        if shared is not None:
            yield shared
            return
        conn = self._open()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def batch(self):
        """Group many writes into one transaction (one commit instead of hundreds)."""
        if getattr(self._local, "conn", None) is not None:
            yield
            return
        conn = self._open()
        self._local.conn = conn
        try:
            with conn:
                yield
        finally:
            self._local.conn = None
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding TEXT DEFAULT '',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1,
                    evidence TEXT NOT NULL DEFAULT '',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id, relation),
                    FOREIGN KEY (source_id) REFERENCES nodes(id),
                    FOREIGN KEY (target_id) REFERENCES nodes(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS co_occurrences (
                    concept_a TEXT NOT NULL,
                    concept_b TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    last_seen TEXT NOT NULL,
                    PRIMARY KEY (concept_a, concept_b)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)")

    @staticmethod
    def node_id(kind: str, label: str) -> str:
        normalized = " ".join((label or "").lower().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return f"{kind}:{digest}"

    @staticmethod
    def extract_terms(text: str, limit: int = 24) -> list[str]:
        counts = Counter(
            token
            for token in tokenize(text)
            if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
        )
        return [term for term, _ in counts.most_common(limit)]

    # ─── LLM-powered concept extraction ───

    def extract_concepts(self, question: str, answer: str, llm_func: Callable) -> list[dict]:
        """Use the LLM to extract semantic concepts and their relationships."""
        try:
            prompt = CONCEPT_EXTRACTION_PROMPT.format(question=question[:500], answer=answer[:2000])
            result = llm_func(prompt)
            return self._parse_concept_output(result, question, answer)
        except Exception:
            return []

    def _parse_concept_output(self, text: str, question: str, answer: str) -> list[dict]:
        concepts = []
        seen = set()
        for line in text.strip().split("\n"):
            line = line.strip().strip("- ").strip()
            if not line or ":" not in line:
                if line and len(line) > 2 and line.lower() not in seen:
                    concepts.append({"name": line, "relations": []})
                    seen.add(line.lower())
                continue
            parts = line.split(":", 1)
            name = parts[0].strip()
            if not name or len(name) < 3 or name.lower() in seen:
                continue
            seen.add(name.lower())
            relations = []
            rest = parts[1].strip()
            if "->" in rest:
                rel_parts = rest.split("->", 1)
                rel_type = rel_parts[0].strip()
                target = rel_parts[1].strip()
                if rel_type in RELATION_TYPES:
                    relations.append({"type": rel_type, "target": target})
            elif ">" in rest:
                rel_parts = rest.split(">", 1)
                rel_type = rel_parts[0].strip()
                target = rel_parts[1].strip()
                # Try common misspellings
                if rel_type in RELATION_TYPES:
                    relations.append({"type": rel_type, "target": target})
            concepts.append({"name": name, "relations": relations})
        return concepts[:8]

    # ─── Node operations ───

    def upsert_node(
        self,
        kind: str,
        label: str,
        metadata: dict | None = None,
        weight: float = 1.0,
        node_id: str | None = None,
    ) -> str:
        node_id = node_id or self.node_id(kind, label)
        now = _now()
        metadata_json = _json(metadata)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO nodes (id, kind, label, weight, metadata, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    weight = MIN(nodes.weight + excluded.weight, 100),
                    label = excluded.label,
                    metadata = CASE
                        WHEN excluded.metadata != '{}' THEN excluded.metadata
                        ELSE nodes.metadata
                    END,
                    last_seen = excluded.last_seen
            """, (node_id, kind, label, weight, metadata_json, now, now))
        return node_id

    def get_node(self, node_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row:
            return dict(row)
        return None

    # ─── Edge operations ───

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        evidence: str = "",
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO edges (source_id, target_id, relation, weight, evidence, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                    weight = MIN(edges.weight + excluded.weight, 50),
                    evidence = excluded.evidence,
                    last_seen = excluded.last_seen
            """, (source_id, target_id, relation, weight, _clip(evidence, 500), now, now))

    # ─── Co-occurrence tracking (Hebbian learning) ───

    def record_co_occurrence(self, concept_a: str, concept_b: str) -> None:
        """Record that two concepts appeared together. Hebbian: fire together, wire together."""
        if concept_a == concept_b:
            return
        a, b = sorted([concept_a.lower(), concept_b.lower()])
        now = _now()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO co_occurrences (concept_a, concept_b, count, last_seen)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(concept_a, concept_b) DO UPDATE SET
                    count = MIN(co_occurrences.count + 1, 100),
                    last_seen = excluded.last_seen
            """, (a, b, now))

    def get_co_occurrences(self, concept: str, limit: int = 10) -> list[tuple[str, int]]:
        concept = concept.lower()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT concept_a, concept_b, count FROM co_occurrences
                WHERE concept_a = ? OR concept_b = ?
                ORDER BY count DESC LIMIT ?
            """, (concept, concept, limit)).fetchall()
        results = []
        for row in rows:
            other = row["concept_b"] if row["concept_a"] == concept else row["concept_a"]
            results.append((other, row["count"]))
        return results

    # ─── Transitive inference ───

    def infer_transitive(self, node_id: str, max_depth: int = 2) -> int:
        """Create inferred connections through transitive relationships.
        If A→B and B→C, strengthen the A→C path.
        """
        inferred = 0
        with self._connect() as conn:
            # Find nodes reachable in 2 steps
            rows = conn.execute("""
                SELECT e1.target_id AS mid, e2.target_id AS target, e1.relation AS r1, e2.relation AS r2,
                       e1.weight AS w1, e2.weight AS w2
                FROM edges e1
                JOIN edges e2 ON e1.target_id = e2.source_id AND e1.relation = e2.relation
                WHERE e1.source_id = ? AND e1.target_id != e2.target_id
                  AND e1.relation IN ('related_to', 'is_a', 'part_of', 'informs', 'leads_to')
            """, (node_id,)).fetchall()

            for row in rows:
                target = row["target"]
                mid = row["mid"]
                r1 = row["r1"]
                w = min(float(row["w1"]), float(row["w2"])) * 0.3

                # Check if edge already exists
                existing = conn.execute(
                    "SELECT weight FROM edges WHERE source_id = ? AND target_id = ? AND relation = ?",
                    (node_id, target, r1)
                ).fetchone()

                if not existing:
                    conn.execute("""
                        INSERT INTO edges (source_id, target_id, relation, weight, evidence, first_seen, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (node_id, target, r1, w,
                          f"Inferred via {mid}", _now(), _now()))
                    inferred += 1
                elif existing["weight"] < w:
                    conn.execute("""
                        UPDATE edges SET weight = MIN(weight + ?, 50), last_seen = ?
                        WHERE source_id = ? AND target_id = ? AND relation = ?
                    """, (w * 0.5, _now(), node_id, target, r1))

        return inferred

    # ─── Weight decay ───

    def decay_weights(self, factor: float = 0.995, min_weight: float = 0.1) -> None:
        """Slowly decay weights of all edges. Like forgetting unused memories."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE edges SET weight = MAX(?, weight * ?)
                WHERE last_seen < datetime('now', '-7 days')
            """, (min_weight, factor))
            conn.execute("""
                UPDATE nodes SET weight = MAX(?, weight * ?)
                WHERE last_seen < datetime('now', '-7 days')
            """, (min_weight, factor * 0.5))

    # ─── Enhanced research recording ───

    def record_research(self, question: str, answer: str, sources: Iterable, tag: str = "General",
                        concepts: list[dict] | None = None) -> str:
        """Record research with optional semantic concepts for brain-like connections."""
        with self.batch():
            return self._record_research(question, answer, sources, tag, concepts)

    def _record_research(self, question, answer, sources, tag, concepts) -> str:
        timestamp = _now()

        # Always create the research session node
        session_id = self.upsert_node(
            "research",
            _clip(question, 160) or "Untitled research",
            {
                "question": question,
                "answer": _clip(answer, 3000),
                "tag": tag,
                "timestamp": timestamp,
            },
            node_id=self.node_id("research", f"{timestamp}:{question}"),
        )

        question_id = self.upsert_node("question", question, {"text": question})
        tag_id = self.upsert_node("tag", tag or "General")
        self.upsert_edge(session_id, question_id, "asked", weight=1.0, evidence=question)
        self.upsert_edge(session_id, tag_id, "tagged", weight=1.0)

        if concepts:
            self._record_semantic_concepts(session_id, question_id, concepts, question, answer, sources)
        else:
            self._record_term_based(session_id, question_id, question, answer, sources)

        # Transitive inference on the new nodes
        self.infer_transitive(question_id)

        return session_id

    def _record_semantic_concepts(self, session_id: str, question_id: str,
                                   concepts: list[dict], question: str, answer: str,
                                   sources: Iterable) -> None:
        """Record LLM-extracted concepts with rich relationships."""
        concept_ids = []

        for concept in concepts:
            name = concept["name"]
            cid = self.upsert_node("concept", name,
                                   {"extracted_from": _clip(question, 100),
                                    "description": _clip(name, 200)})
            concept_ids.append(cid)

            self.upsert_edge(question_id, cid, "mentions", weight=WEIGHT_BOOSTS["direct_query"])
            self.upsert_edge(session_id, cid, "explored", weight=WEIGHT_BOOSTS["direct_query"])

            # Record relationships between concepts
            for rel in concept.get("relations", []):
                target_name = rel.get("target", "")
                if not target_name:
                    continue
                target_cid = self.upsert_node("concept", target_name,
                                              {"description": _clip(target_name, 200)})
                self.upsert_edge(question_id, target_cid, "mentions", weight=WEIGHT_BOOSTS["co_occurrence"])
                rel_type = rel.get("type", "related_to")
                self.upsert_edge(cid, target_cid, rel_type, weight=WEIGHT_BOOSTS["direct_query"],
                                 evidence=f"{name} {rel_type} {target_name}")
                concept_ids.append(target_cid)

        # Hebbian co-occurrence between all concepts in this session
        for i, cid_a in enumerate(concept_ids):
            node_a = self.get_node(cid_a)
            label_a = node_a["label"] if node_a else ""
            for cid_b in concept_ids[i + 1:]:
                node_b = self.get_node(cid_b)
                label_b = node_b["label"] if node_b else ""
                if label_a and label_b:
                    self.record_co_occurrence(label_a, label_b)
                self.upsert_edge(cid_a, cid_b, "co_occurs_with",
                                 weight=WEIGHT_BOOSTS["co_occurrence"])

        # Also record sources
        for source in sources or []:
            title = _item_get(source, "title", "Untitled source") or "Untitled source"
            url = _item_get(source, "url", "") or ""
            content = _item_get(source, "content", "") or _item_get(source, "snippet", "") or ""
            label = title if title != "Untitled source" else url
            source_id = self.upsert_node(
                "source", label or "Untitled source",
                {"title": title, "url": url, "snippet": _clip(content, 1200)},
            )
            self.upsert_edge(session_id, source_id, "used_source", evidence=url)
            self.upsert_edge(question_id, source_id, "returned_source", evidence=url)

            # Connect source to relevant concepts
            for cid in concept_ids:
                node_c = self.get_node(cid)
                if node_c and node_c["label"].lower() in f"{title} {content}".lower():
                    self.upsert_edge(source_id, cid, "discusses", weight=0.5)

            # Fallback: term-based source indexing
            for term in self.extract_terms(f"{title}\n{content}", limit=8):
                term_id = self.upsert_node("term", term)
                self.upsert_edge(source_id, term_id, "contains", weight=0.3, evidence=title)

    def _record_term_based(self, session_id: str, question_id: str,
                            question: str, answer: str, sources: Iterable) -> None:
        """Fallback: traditional term-based recording."""
        for term in self.extract_terms(f"{question}\n{answer}", limit=20):
            term_id = self.upsert_node("term", term)
            self.upsert_edge(question_id, term_id, "mentions", weight=0.5, evidence=question)
            self.upsert_edge(session_id, term_id, "mentions", weight=0.5, evidence=answer)

        for source in sources or []:
            title = _item_get(source, "title", "Untitled source") or "Untitled source"
            url = _item_get(source, "url", "") or ""
            content = _item_get(source, "content", "") or _item_get(source, "snippet", "") or ""
            label = title if title != "Untitled source" else url
            source_id = self.upsert_node(
                "source", label or "Untitled source",
                {"title": title, "url": url, "snippet": _clip(content, 1200)},
            )
            self.upsert_edge(session_id, source_id, "used_source", evidence=url)
            self.upsert_edge(question_id, source_id, "returned_source", evidence=url)
            for term in self.extract_terms(f"{title}\n{content}", limit=8):
                term_id = self.upsert_node("term", term)
                self.upsert_edge(source_id, term_id, "contains", weight=0.3, evidence=title)

    def record_document(self, name: str, content: str) -> str:
        with self.batch():
            return self._record_document(name, content)

    def _record_document(self, name: str, content: str) -> str:
        doc_id = self.upsert_node(
            "document", name,
            {"name": name, "snippet": _clip(content, 2000), "indexed_at": _now()},
        )
        for term in self.extract_terms(content, limit=24):
            term_id = self.upsert_node("term", term)
            self.upsert_edge(doc_id, term_id, "contains", weight=0.3, evidence=name)
        return doc_id

    # ─── Spreading activation retrieval ───

    def context_for_query(self, query: str, limit: int = 8) -> str:
        """Brain-like retrieval using spreading activation.
        
        Instead of just matching terms, we:
        1. Find concepts matching the query
        2. Activate connected concepts (spreading activation)
        3. Retrieve the most activated memories
        """
        query_lower = query.lower()
        activation = defaultdict(float)

        with self._connect() as conn:
            # Phase 1: Find directly matching concepts and terms
            matching_nodes = conn.execute("""
                SELECT id, kind, label, weight, metadata FROM nodes
                WHERE (kind = 'concept' OR kind = 'term')
                  AND (LOWER(label) LIKE ? OR LOWER(label) LIKE ? OR LOWER(label) LIKE ?)
                LIMIT 15
            """, (f"%{query_lower}%", f"{query_lower}%", f"%{query_lower}")).fetchall()

            for node in matching_nodes:
                nid = node["id"]
                activation[nid] += 3.0  # Direct match boost

                # Phase 2: Spread activation to connected research/sources
                connected = conn.execute("""
                    SELECT DISTINCT n.id, n.kind, n.label, n.weight, n.metadata, e.weight AS ew
                    FROM edges e
                    JOIN nodes n ON n.id = e.source_id
                    WHERE e.target_id = ? AND n.kind IN ('research', 'source', 'question')
                    UNION
                    SELECT DISTINCT n.id, n.kind, n.label, n.weight, n.metadata, e.weight AS ew
                    FROM edges e
                    JOIN nodes n ON n.id = e.target_id
                    WHERE e.source_id = ? AND n.kind IN ('research', 'source', 'question')
                    LIMIT 30
                """, (nid, nid)).fetchall()

                for row in connected:
                    activation[row["id"]] += float(row["ew"] or 0.5) * 0.8

                    # Phase 3: Second-order spread (concepts connected to THOSE results)
                    second_order = conn.execute("""
                        SELECT DISTINCT n.id, n.kind, n.label, n.metadata
                        FROM edges e
                        JOIN nodes n ON n.id = e.target_id
                        WHERE e.source_id = ? AND n.kind = 'concept'
                        LIMIT 10
                    """, (row["id"],)).fetchall()

                    for row2 in second_order:
                        activation[row2["id"]] += 0.3

            # Also match co-occurrence patterns
            for node in matching_nodes:
                label = node["label"]
                co_pairs = self.get_co_occurrences(label, limit=5)
                for other_label, count in co_pairs:
                    other_node = conn.execute(
                        "SELECT id FROM nodes WHERE LOWER(label) = ? AND kind = 'concept' LIMIT 1",
                        (other_label,)
                    ).fetchone()
                    if other_node:
                        activation[other_node["id"]] += float(count) * 0.2

            # Fallback: traditional term matching
            terms = self.extract_terms(query, limit=8)
            for term in terms:
                term_id = self.node_id("term", term)
                rows = conn.execute("""
                    SELECT n.*, e.weight AS ew
                    FROM edges e JOIN nodes n ON n.id = e.source_id
                    WHERE e.target_id = ? AND n.kind IN ('research', 'document', 'source')
                    UNION ALL
                    SELECT n.*, e.weight AS ew
                    FROM edges e JOIN nodes n ON n.id = e.target_id
                    WHERE e.source_id = ? AND n.kind IN ('research', 'document', 'source')
                """, (term_id, term_id)).fetchall()

                for row in rows:
                    activation[row["id"]] += float(row["ew"] or 0.3) * 0.5

        if not activation:
            return ""

        ordered = sorted(activation.items(), key=lambda x: x[1], reverse=True)[:limit]
        grouped = {"research": [], "source": [], "concept": [], "document": []}

        with self._connect() as conn:
            for node_id, score in ordered:
                row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
                if not row:
                    continue
                metadata = _load_json(row["metadata"])
                kind = row["kind"]
                label = row["label"]

                if kind == "research":
                    answer_text = _clip(metadata.get("answer", ""), 500)
                    q = metadata.get("question", label)
                    grouped["research"].append(f"- Q: {q}\n  Known: {answer_text}")
                elif kind == "source":
                    url = metadata.get("url", "")
                    grouped["source"].append(f"- {label} ({url})")
                elif kind == "concept":
                    grouped["concept"].append(f"- {label}")
                elif kind == "document":
                    grouped["document"].append(f"- {label}: {_clip(metadata.get('snippet', ''), 300)}")

        sections = []
        if grouped["concept"]:
            sections.append("Connected concepts:\n" + "\n".join(grouped["concept"][:5]))
        if grouped["research"]:
            sections.append("Related past research:\n" + "\n".join(grouped["research"][:3]))
        if grouped["source"]:
            sections.append("Remembered sources:\n" + "\n".join(grouped["source"][:3]))
        if grouped["document"]:
            sections.append("Related local documents:\n" + "\n".join(grouped["document"][:3]))

        if not sections:
            return ""
        return "--- GRAPH MEMORY ---\n" + "\n\n".join(sections)

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM co_occurrences")
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")

    def stats(self) -> dict:
        with self._connect() as conn:
            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            co_count = conn.execute("SELECT COUNT(*) FROM co_occurrences").fetchone()[0]
            kinds = conn.execute(
                "SELECT kind, COUNT(*) AS count FROM nodes GROUP BY kind ORDER BY count DESC"
            ).fetchall()
        return {
            "nodes": node_count,
            "edges": edge_count,
            "co_occurrences": co_count,
            "kinds": {row["kind"]: row["count"] for row in kinds},
            "path": str(self.db_path),
        }

    def recent_nodes(self, limit: int = 25) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, kind, label, weight, metadata, last_seen FROM nodes ORDER BY last_seen DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def top_nodes(self, limit: int = 160) -> list[dict]:
        """Highest-weight nodes, used to draw the graph view."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, kind, label, weight FROM nodes ORDER BY weight DESC, last_seen DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_edges(self, limit: int = 500) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_id, target_id, relation, weight FROM edges ORDER BY weight DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_connected_nodes(self, node_id: str, depth: int = 1) -> list[dict]:
        """Get nodes connected to a given node, for visualization."""
        results = []
        seen = set()
        with self._connect() as conn:
            current = {node_id}
            for _ in range(depth):
                if not current:
                    break
                placeholders = ",".join("?" for _ in current)
                rows = conn.execute(f"""
                    SELECT DISTINCT n.*, e.relation, e.weight AS ew
                    FROM edges e JOIN nodes n ON n.id = e.target_id
                    WHERE e.source_id IN ({placeholders})
                    UNION ALL
                    SELECT DISTINCT n.*, e.relation, e.weight AS ew
                    FROM edges e JOIN nodes n ON n.id = e.source_id
                    WHERE e.target_id IN ({placeholders})
                """, list(current)).fetchall()

                next_set = set()
                for row in rows:
                    if row["id"] not in seen:
                        seen.add(row["id"])
                        results.append(dict(row))
                        next_set.add(row["id"])
                current = next_set
        return results