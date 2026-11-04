"""Knowledge graph mind map generator - visualizes research concepts as a Mermaid mind map."""
from __future__ import annotations



class MindMapGenerator:
    """Generates Mermaid mind maps from the knowledge graph data.
    
    Shows how research concepts, sources, and topics connect to each other,
    forming a growing web of knowledge like a human brain.
    """

    def __init__(self, graph=None):
        self.graph = graph
        self._concept_cache = {}

    def load_from_graph(self, graph) -> dict:
        """Load mind map data from the knowledge graph."""
        self.graph = graph
        stats = graph.stats()
        kinds = stats.get("kinds", {})

        concepts = []
        research_topics = []
        sources = []
        terms = []

        with graph._connect() as conn:
            nodes = conn.execute(
                "SELECT id, kind, label, weight, metadata FROM nodes ORDER BY weight DESC LIMIT 200"
            ).fetchall()

            for node in nodes:
                kind = node["kind"]
                entry = {
                    "id": node["id"],
                    "label": node["label"],
                    "weight": node["weight"],
                }
                if kind == "concept":
                    concepts.append(entry)
                elif kind == "research":
                    research_topics.append(entry)
                elif kind == "source":
                    sources.append(entry)
                elif kind == "term":
                    terms.append(entry)

            # Get top edges for connection mapping
            edges = conn.execute(
                """SELECT source_id, target_id, relation, weight FROM edges 
                   ORDER BY weight DESC LIMIT 100"""
            ).fetchall()

        return {
            "concepts": concepts[:30],
            "research": research_topics[:20],
            "sources": sources[:15],
            "terms": terms[:20],
            "edges": [dict(e) for e in edges],
            "stats": stats,
        }

    def to_mermaid_mindmap(self, data: dict | None = None) -> str:
        """Generate a Mermaid mindmap from knowledge graph data.
        
        Shows research topics → concepts → sources as a hierarchical map.
        """
        if not data or (not data["concepts"] and not data["research"] and not data["sources"]):
            return self._empty_mindmap()

        lines = ["mindmap"]
        lines.append('  root((Knowledge Graph))')

        # Branch: Research Topics
        if data["research"]:
            lines.append("    Research Topics")
            for r in data["research"][:10]:
                label = r["label"][:60]
                lines.append(f"      {label}")

        # Branch: Concepts
        if data["concepts"]:
            lines.append("    Concepts")
            for c in data["concepts"][:15]:
                label = c["label"][:50]
                w = int(c["weight"])
                marker = "🟢" if w > 10 else "🟡" if w > 3 else "⚪"
                lines.append(f"      {marker}{label}")

        # Branch: Sources
        if data["sources"]:
            lines.append("    Sources")
            for s in data["sources"][:10]:
                label = s["label"][:50]
                lines.append(f"      {label}")

        # Branch: Key Terms
        if data["terms"]:
            lines.append("    Key Terms")
            for t in data["terms"][:10]:
                lines.append(f"      {t['label']}")

        return "\n".join(lines)

    def to_mermaid_graph(self, data: dict | None = None) -> str:
        """Generate a Mermaid flow graph showing relationships between knowledge nodes."""
        if not data or not data.get("edges"):
            return self._empty_graph()

        lines = ["graph LR"]
        added_nodes = set()
        edge_count = 0

        for edge in data["edges"]:
            if edge_count >= 30:
                break
            src = edge["source_id"][:20]
            tgt = edge["target_id"][:20]
            rel = edge.get("relation", "related")

            if src not in added_nodes:
                label = self._label_for_id(data, edge["source_id"]) or src
                lines.append(f"  {src}[\"{label[:30]}\"]")
                added_nodes.add(src)
            if tgt not in added_nodes:
                label = self._label_for_id(data, edge["target_id"]) or tgt
                lines.append(f"  {tgt}(\"{label[:30]}\")")
                added_nodes.add(tgt)

            lines.append(f"  {src} -- {rel} --> {tgt}")
            edge_count += 1

        return "\n".join(lines)

    def _label_for_id(self, data: dict, node_id: str) -> str | None:
        for group in ("concepts", "research", "sources", "terms"):
            for item in data.get(group, []):
                if item["id"] == node_id:
                    return item["label"]
        return None

    def to_mermaid_timeline(self, data: dict | None = None) -> str:
        """Generate a timeline of knowledge growth."""
        if not data or not data.get("stats"):
            return ""

        kinds = data["stats"].get("kinds", {})
        lines = ["timeline"]
        lines.append("    title Knowledge Growth")
        for kind, count in sorted(kinds.items(), key=lambda x: -x[1]):
            lines.append(f"    {kind} : {count} nodes")
        return "\n".join(lines)

    def _empty_mindmap(self) -> str:
        return """mindmap
  root((Knowledge Graph))
    Empty
      Start researching to build your knowledge map
      Concepts will appear here as you search"""

    def _empty_graph(self) -> str:
        return "graph LR\n  EMPTY[No connections yet]"
