import json
from pathlib import Path
import torch
from torch_geometric.data import Data


def build_data_from_cpg(nodes_path, edges_path, label, node_encoder, make_undirected=False):
    # 1. Load JSON
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    # 2. Node index map
    node_idx_map = {n["id"]: i for i, n in enumerate(nodes)}

    # 3. Encode node features
    x = node_encoder(nodes)

    # 4. Build edge_index
    src = []
    dst = []

    for e in edges:
        s = e["src"]
        d = e["dst"]
        if s in node_idx_map and d in node_idx_map:
            src.append(node_idx_map[s])
            dst.append(node_idx_map[d])

    # Nếu muốn graph vô hướng
    if make_undirected:
        src_all = src + dst
        dst_all = dst + src
    else:
        src_all = src
        dst_all = dst

    if len(src_all) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=x.size(0)
    )
