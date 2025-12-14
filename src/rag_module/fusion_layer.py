import torch
import os
from src.vector_db.search_faiss import GraphVectorDB
from src.train.model import DevignModel
from dataset.node_encoder import TypeOnlyEncoder
from dataset.cpg_dataset_pyg import CPGPyGDataset
from src.train.collate_fn import pyg_to_batch_tensors

import json
from pathlib import Path


class HybridFusion:
    def __init__(
        self,
        encoder_ckpt="checkpoints/best_encoder.pt",
        root="data/cpg",
        labels_file="dataset/labels.json",
        num_edge_types=5,
        hidden_dim=128,
        alpha=0.5,
        beta=0.5
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== Load FAISS DB =====
        self.db = GraphVectorDB()

    # ------ Load encoder vocab (KHÔNG fit lại) ------
        vocab_path = "checkpoints/type_vocab.pt"
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing {vocab_path}. Hãy chạy train_devign để tạo.")

        self.node_encoder = TypeOnlyEncoder()
        self.node_encoder.type_vocab = torch.load(vocab_path, map_location="cpu")
        self.node_encoder.fitted = True
        input_dim = len(self.node_encoder.type_vocab)  # = 47 (khớp checkpoint)
        print(f"[Encoder] loaded vocab with {input_dim} node types.")

        # ------ Build model (PHẢI khớp hidden_dim lúc train) ------
        hidden_dim = 64          # checkpoint bạn đang lưu là 64
        step = 8                 # khớp train_devign
        num_edge_types = 5       # khớp train_devign

        self.model = DevignModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            step=step,
            num_edge_types=num_edge_types,
            ).to(self.device)

        self.model.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=self.device))
        self.model.eval()

        self.alpha = alpha
        self.beta = beta
        self.root = Path(root)

    # -------------------------------------------------

    def _build_type_encoder(self, root, labels_file):
        from dataset.node_encoder import TypeOnlyEncoder

        root = Path(root)
        with open(labels_file, "r") as f:
            labels = json.load(f)

        all_nodes = []
        for gid in labels.keys():
            p = root / gid / "nodes.json"
            if p.exists():
                with open(p) as nf:
                    all_nodes.append(json.load(nf))

        enc = TypeOnlyEncoder()
        enc.fit(all_nodes)
        return enc

    # -------------------------------------------------

    def get_devign_score(self, graph_id):
        """Chạy Devign classifier trên 1 graph + trả về sigmoid score."""
        from dataset.data_builder import build_data_from_cpg

        path = self.root / graph_id
        nodes = path / "nodes.json"
        edges = path / "edges.json"

        data = build_data_from_cpg(
            nodes,
            edges,
            label=0,  # dummy
            node_encoder=self.node_encoder,
            make_undirected=True
        )
        data.graph_id = graph_id

        # collate 1 sample
        node_f, adj, label, _ = pyg_to_batch_tensors([data], num_edge_types=5)
        node_f = node_f.to(self.device)
        adj = adj.to(self.device)

        with torch.no_grad():
            logits, _, _ = self.model(node_f, adj)
            score = torch.sigmoid(logits)[0].item()
        return score

    # -------------------------------------------------

    def hybrid_search(self, graph_id, k=5):
        """
        Search FAISS => kết hợp Devign Score => trả về danh sách sorted.
        """
        faiss_results = self.db.search_by_id(graph_id, k=k)

        enriched = []
        for r in faiss_results:
            gid = r["graph_id"]

            devign_score = self.get_devign_score(gid)
            fusion = self.alpha * devign_score + self.beta * r["score"]

            enriched.append({
                "graph_id": gid,
                "similarity": r["score"],
                "devign": devign_score,
                "fusion": fusion,
                "label": r["label"],
            })

        enriched = sorted(enriched, key=lambda x: x["fusion"], reverse=True)
        return enriched
