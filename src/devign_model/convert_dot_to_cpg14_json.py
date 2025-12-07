#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pydot

if len(sys.argv) != 2:
    print("Usage: python3 convert_dot_to_cpg14_json.py /path/to/SAMPLE.cpg14")
    sys.exit(1)

out_dir = Path(sys.argv[1])
if not out_dir.is_dir():
    print(f"[ERR] Not a directory: {out_dir}")
    sys.exit(2)

dot_files = sorted(p for p in out_dir.glob("*-cpg.dot") if p.is_file())
if not dot_files:
    print(f"[ERR] No *-cpg.dot in {out_dir}")
    sys.exit(3)

nodes = {}
edges = []

def clean_name(name: str) -> str:
    # bỏ quote quanh id, ví dụ "1234" -> 1234
    name = name.strip()
    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    return name

def merge_graph(dot_path: Path):
    try:
        graphs = pydot.graph_from_dot_file(str(dot_path))
    except Exception as e:
        print(f"[WARN] Failed to parse {dot_path}: {e}")
        return

    if not graphs:
        print(f"[WARN] No graphs in {dot_path}")
        return

    g = graphs[0]

    # NODES
    for node in g.get_nodes():
        nid = clean_name(node.get_name())
        # pydot hay tạo node đặc biệt tên "node" (style default) -> bỏ qua
        if nid in ("node", "graph", "edge"):
            continue
        attrs = node.get_attributes() or {}
        existing = nodes.setdefault(nid, {"id": nid, "attrs": {}})
        existing["attrs"].update(attrs)

    # EDGES
    for edge in g.get_edges():
        src = clean_name(edge.get_source())
        dst = clean_name(edge.get_destination())
        attrs = edge.get_attributes() or {}
        # đảm bảo node tồn tại trong dict nodes
        nodes.setdefault(src, {"id": src, "attrs": {}})
        nodes.setdefault(dst, {"id": dst, "attrs": {}})
        edges.append({"src": src, "dst": dst, "attrs": attrs})


for dot in dot_files:
    merge_graph(dot)

nodes_list = list(nodes.values())

(out_dir / "nodes.json").write_text(
    json.dumps(nodes_list, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(out_dir / "edges.json").write_text(
    json.dumps(edges, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(
    f"[OK] nodes={len(nodes_list)} edges={len(edges)} -> "
    f"{out_dir}/nodes.json, {out_dir}/edges.json"
)
