from torch_geometric.data import Dataset
from pathlib import Path
import json
from .data_builder import build_data_from_cpg


class CPGPyGDataset(Dataset):
    def __init__(self, root, labels_file, node_encoder,
                 make_undirected=True, max_nodes=500, verbose=True):
        self.root = Path(root)
        self.labels_file = Path(labels_file)
        self.node_encoder = node_encoder
        self.make_undirected = make_undirected
        self.max_nodes = max_nodes

        with open(self.labels_file, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        self.graph_dirs = []
        self.graph_labels = []
        self.dropped = 0
        self.total_seen = 0

        for name, lbl in self.labels.items():
            d = self.root / name
            if not d.is_dir():
                continue

            nodes_path = d / "nodes.json"
            edges_path = d / "edges.json"
            if not (nodes_path.is_file() and edges_path.is_file()):
                continue

            # đếm node để lọc
            try:
                with open(nodes_path, "r", encoding="utf-8") as nf:
                    nodes = json.load(nf)
                n_nodes = len(nodes)
            except Exception:
                continue

            self.total_seen += 1
            if self.max_nodes is not None and n_nodes > self.max_nodes:
                self.dropped += 1
                continue

            self.graph_dirs.append(d)
            self.graph_labels.append(lbl)

        if verbose:
            kept = len(self.graph_dirs)
            drop = self.dropped
            total = self.total_seen if self.total_seen > 0 else (kept + drop)
            pct = (drop / total * 100.0) if total > 0 else 0.0
            print(f"[Dataset] max_nodes={self.max_nodes} | kept={kept} | dropped={drop} ({pct:.2f}%)")

        super().__init__()

    def len(self):
        return len(self.graph_dirs)

    def get(self, idx):
        graph_dir = self.graph_dirs[idx]
        label = self.graph_labels[idx]

        nodes_path = graph_dir / "nodes.json"
        edges_path = graph_dir / "edges.json"

        data = build_data_from_cpg(
            nodes_path,
            edges_path,
            label,
            self.node_encoder,
            make_undirected=self.make_undirected
        )
        return data
