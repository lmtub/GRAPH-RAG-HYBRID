import json
from pathlib import Path
from torch_geometric.loader import DataLoader

from dataset.node_encoder import TypeOnlyEncoder
from dataset.cpg_dataset_pyg import CPGPyGDataset


# -------------------------------------------------------
# 1) FIT NODE ENCODER
# -------------------------------------------------------

root = Path("data/cpg")
labels_path = Path("dataset/labels.json")

with open(labels_path, "r") as f:
    labels_dict = json.load(f)

# Collect all nodes for fitting encoder
all_nodes_lists = []
for name in labels_dict.keys():
    folder = root / name
    nodes_path = folder / "nodes.json"
    if nodes_path.exists():
        with open(nodes_path, "r") as f:
            nodes = json.load(f)
        all_nodes_lists.append(nodes)

encoder = TypeOnlyEncoder()
encoder.fit(all_nodes_lists)


# -------------------------------------------------------
# 2) CREATE DATASET + DATALOADER
# -------------------------------------------------------

dataset = CPGPyGDataset(
    root="data/cpg",
    labels_file="dataset/labels.json",
    node_encoder=encoder,
    make_undirected=True
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    print("Batch x shape:", batch.x.shape)
    print("Batch edge shape:", batch.edge_index.shape)
    print("Batch y:", batch.y)
    break
