import json
from pathlib import Path
import torch
from torch_geometric.data import Data
import re
import html

TYPE_RE = re.compile(r"^<\s*\(?\s*([A-Z0-9_]+)", re.IGNORECASE)

def infer_type_from_label(label: str) -> str:
    if not label: 
        return "UNKNOWN"
    
    # Giải mã các ký tự đặc biệt như &lt; thành <
    s = html.unescape(str(label)).strip()

    # Tìm loại nút (Ví dụ: từ <(METHOD,...) lấy ra METHOD)
    m = TYPE_RE.match(s)
    if m:
        return m.group(1).upper()

    # Cách dự phòng 2: Nếu Regex vẫn trượt, tìm cụm chữ cái viết hoa đầu tiên sau dấu <(
    if s.startswith("<"):
        clean_s = s.replace("<", "").replace("(", "").strip()
        # Lấy từ đầu tiên trước dấu phẩy hoặc dấu ngoặc
        parts = re.split(r'[,<>\s]', clean_s)
        if parts and parts[0].isupper():
            return parts[0]

    return "UNKNOWN"

# Map edge label -> edge type id (tùy bạn mở rộng)
EDGE_TYPE_MAP = {
    "AST": 0,
    "CFG": 1,
    "DDG": 2,
    "CDG": 3,
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


# def build_data_from_cpg(nodes_path, edges_path, label, node_encoder, make_undirected=True, graph_id=None):
    # 1) Load JSON
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)
    # 1.5) Inject TYPE từ label (fix UNKNOWN)
    for n in nodes:
        attrs = n.get("attrs", {})
        if isinstance(attrs, dict):
            if not attrs.get("TYPE"):
                attrs["TYPE"] = infer_type_from_label(attrs.get("label", ""))
            n["attrs"] = attrs

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

    data = Data(
    x=x,
    edge_index=edge_index,
    edge_type=edge_type,
    y=y,
    num_nodes=x.size(0)
    )

    data.graph_id = graph_id   # <<< thêm dòng này

    return data

def build_data_from_cpg(nodes_path, edges_path, label, node_encoder, make_undirected=True, graph_id=None):
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    # Inject TYPE (Không dùng print)
    for n in nodes:
        attrs = n.get("attrs", {})
        if isinstance(attrs, dict) and not attrs.get("TYPE"):
            attrs["TYPE"] = infer_type_from_label(attrs.get("label", ""))
            n["attrs"] = attrs

    node_idx_map = {str(n["id"]): i for i, n in enumerate(nodes)}
    x = node_encoder(nodes)

    src, dst, etypes = [], [], []
    for e in edges:
        s, d = str(e.get("src")), str(e.get("dst"))
        if s not in node_idx_map or d not in node_idx_map: continue

        label_raw = e.get("attrs", {}).get("label") if isinstance(e.get("attrs"), dict) else None
        prefix = _clean_edge_label(label_raw)
        
        # Chỉ lấy 4 loại cạnh chính
        et = EDGE_TYPE_MAP.get(prefix, EDGE_TYPE_MAP["UNKNOWN"])

        src.append(node_idx_map[s])
        dst.append(node_idx_map[d])
        etypes.append(et)

    # Chuyển thành vô hướng nếu cần
    if make_undirected and len(src) > 0:
        src_all, dst_all, et_all = src + dst, dst + src, etypes + etypes
    else:
        src_all, dst_all, et_all = src, dst, etypes

    if len(src_all) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
        edge_type = torch.tensor(et_all, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, 
                y=torch.tensor([int(label)], dtype=torch.long), num_nodes=x.size(0))
    data.graph_id = graph_id
    return data