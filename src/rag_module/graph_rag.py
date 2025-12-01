import json
from pathlib import Path

from src.rag_module.fusion_layer import HybridFusion


class GraphRAG:
    def __init__(self):
        self.fusion = HybridFusion()

    def retrieve(self, graph_id, k=5):
        """
        Trả về danh sách top-k graph theo fusion score + context (nodes, edges)
        """
        ranked = self.fusion.hybrid_search(graph_id, k=k)

        results = []
        for r in ranked:
            gid = r["graph_id"]

            base = Path("data/cpg") / gid
            nodes = json.load(open(base / "nodes.json"))
            edges = json.load(open(base / "edges.json"))

            r["nodes"] = nodes
            r["edges"] = edges
            results.append(r)

        return results
