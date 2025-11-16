from torch_geometric.data import Dataset
from pathlib import Path
import json
from .data_builder import build_data_from_cpg


class CPGPyGDataset(Dataset):
    def __init__(self, root, labels_file, node_encoder, make_undirected=True):
        self.root = Path(root)
        self.labels_file = Path(labels_file)
        self.node_encoder = node_encoder
        self.make_undirected = make_undirected

        with open(self.labels_file, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        # Tìm tất cả folder CPG hợp lệ
        self.graph_dirs = []
        self.graph_labels = []

        for name, lbl in self.labels.items():
            d = self.root / name
            if d.is_dir():
                self.graph_dirs.append(d)
                self.graph_labels.append(lbl)

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
