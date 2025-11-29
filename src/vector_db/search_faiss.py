import faiss
import torch
import json
import numpy as np


class GraphVectorDB:
    def __init__(self, index_path="data/index/faiss_index.bin", meta_path="data/index/meta.json"):
        self.index = faiss.read_index(index_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.graph_ids = meta["graph_ids"]
        self.labels = meta["labels"]
        self.dim = meta["dim"]
        self.normalize = meta.get("normalize", True)

    def search_by_vector(self, vector, k=5):
        """
        vector: torch.Tensor [dim] hoặc numpy [dim]
        """
        if isinstance(vector, torch.Tensor):
            vector = vector.numpy().astype("float32")

        vector = vector.reshape(1, -1).astype("float32")

        # Normalize giống lúc build
        if self.normalize:
            faiss.normalize_L2(vector)

        D, I = self.index.search(vector, k)
        results = []

        for score, idx in zip(D[0], I[0]):
            results.append({
                "graph_id": self.graph_ids[idx],
                "label": self.labels[idx],
                "score": float(score)
            })

        return results

    def search_by_id(self, graph_id, k=5):
        """
        Tìm vector theo graph_id rồi search hàng xóm
        """
        idx = self.graph_ids.index(graph_id)
        return self.search_by_vector(self.index.reconstruct(idx), k=k)
