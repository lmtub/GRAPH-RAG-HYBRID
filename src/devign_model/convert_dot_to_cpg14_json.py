#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python3 convert_dot_to_cpg14_json.py /path/to/SAMPLE.cpg14")
    sys.exit(1)

out_dir = Path(sys.argv[1])
if not out_dir.is_dir():
    print(f"[ERR] Not a directory: {out_dir}")
    sys.exit(2)

dot_files = sorted(p for p in out_dir.glob("*-cpg.dot") if p.is_file())
if not dot_files:
    print(f"[ERR] No *.dot in {out_dir}")
    sys.exit(3)

NODE_DEF_RE = re.compile(
    r'^\s*("?[^"\s\[\]\-\>]+"?)\s*(\[(?P<attrs>.+?)\])?\s*;\s*$'
)
EDGE_DEF_RE = re.compile(
    r'^\s*("?[^"\s\[\]\-\>]+"?)\s*->\s*("?[^"\s\[\]\-\>]+"?)\s*(\[(?P<attrs>.+?)\])?\s*;\s*$'
)
ATTR_KV_RE  = re.compile(r'(?P<k>[^=\s,]+)\s*=\s*"?(?P<v>[^",]+)"?')

def parse_attrs(txt:str)->dict:
    if not txt: return {}
    return {m.group("k"): m.group("v") for m in ATTR_KV_RE.finditer(txt)}

def norm_id(s:str)->str:
    s = s.strip()
    return s[1:-1] if s.startswith('"') and s.endswith('"') else s

nodes = {}
edges = []

for dot in dot_files:
    with dot.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("//", "#")): continue
            m = EDGE_DEF_RE.match(line)
            if m:
                src, dst = norm_id(m.group(1)), norm_id(m.group(2))
                attrs = parse_attrs(m.group("attrs"))
                edges.append({"src": src, "dst": dst, "attrs": attrs})
                nodes.setdefault(src, {"id": src, "attrs": {}})
                nodes.setdefault(dst, {"id": dst, "attrs": {}})
                continue
            m = NODE_DEF_RE.match(line)
            if m:
                nid = norm_id(line.split("[",1)[0].strip().rstrip(";"))
                attrs = parse_attrs(m.group("attrs"))
                nodes.setdefault(nid, {"id": nid, "attrs": {}})["attrs"].update(attrs)

(out_dir / "nodes.json").write_text(json.dumps(list(nodes.values()), ensure_ascii=False), encoding="utf-8")
(out_dir / "edges.json").write_text(json.dumps(edges, ensure_ascii=False), encoding="utf-8")
print(f"[OK] nodes={len(nodes)} edges={len(edges)} -> {out_dir}/nodes.json, {out_dir}/edges.json")
