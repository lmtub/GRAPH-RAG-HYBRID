import json
from pathlib import Path
import torch
from torch_geometric.data import Data


# Map edge label -> edge type id (tùy bạn mở rộng)
EDGE_TYPE_MAP = {
    "AST": 0,
    "CFG": 1,
    "DDG": 2,
    "CDG": 3,
    "CALL": 4,
    # fallback / unknown
    "UNKNOWN": 0,
}


def _clean_edge_label(raw):
    """
    raw có thể kiểu: "\"DDG: avbuf\"" hoặc "\"AST: \"" hoặc None
    -> trả về prefix: "DDG", "AST", "CFG", "CDG", "CALL"...
    """
    if raw is None:
        return "UNKNOWN"
    s = str(raw)

    # remove quotes & escape
    s = s.replace('\\"', '"')
    s = s.replace('"', '').strip()

    # dạng "DDG: avbuf" -> lấy phần trước dấu :
    if ":" in s:
        prefix = s.split(":", 1)[0].strip()
    else:
        prefix = s.strip()

    prefix = prefix.upper()
    return prefix if prefix else "UNKNOWN"


def build_data_from_cpg(nodes_path, edges_path, label, node_encoder, make_undirected=True):
    # 1) Load JSON
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    # 2) Node index map: node["id"] là string số lớn -> map về 0..N-1
    node_idx_map = {str(n["id"]): i for i, n in enumerate(nodes)}

    # 3) Encode node features
    x = node_encoder(nodes)  # shape [N, feat_dim]

    # 4) Build edge_index + edge_type
    src, dst, etypes = [], [], []

    for e in edges:
        s = str(e.get("src"))
        d = str(e.get("dst"))
        if s not in node_idx_map or d not in node_idx_map:
            continue

        label_raw = None
        attrs = e.get("attrs", {})
        if isinstance(attrs, dict):
            label_raw = attrs.get("label")

        prefix = _clean_edge_label(label_raw)
        et = EDGE_TYPE_MAP.get(prefix, EDGE_TYPE_MAP["UNKNOWN"])

        src.append(node_idx_map[s])
        dst.append(node_idx_map[d])
        etypes.append(et)

    # 5) undirected option: nhân đôi cạnh & giữ nguyên edge_type tương ứng
    if make_undirected and len(src) > 0:
        src_all = src + dst
        dst_all = dst + src
        et_all = etypes + etypes
    else:
        src_all, dst_all, et_all = src, dst, etypes

    if len(src_all) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
        edge_type = torch.tensor(et_all, dtype=torch.long)

    y = torch.tensor([int(label)], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,   # ✅ thêm edge_type ở đây
        y=y,
        num_nodes=x.size(0)
    )
