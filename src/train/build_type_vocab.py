import os, json, torch
from pathlib import Path
from dataset.node_encoder import TypeOnlyEncoder

def build_type_encoder(root: str, labels_file: str, max_graphs: int = 500):
    root_path = Path(root)
    labels = json.load(open(labels_file, "r", encoding="utf-8"))

    all_nodes_lists = []
    cnt = 0
    for name in labels.keys():
        if cnt >= max_graphs:
            break
        p = root_path / name / "nodes.json"
        if p.exists():
            all_nodes_lists.append(json.load(open(p, "r", encoding="utf-8")))
            cnt += 1

    enc = TypeOnlyEncoder()
    enc.fit(all_nodes_lists)
    print(f"[Info] Encoder fitted on {cnt} graphs | vocab={len(enc.type_vocab)}")
    return enc

def main():
    root = "data/cpg"
    labels_file = "dataset/labels.json"
    max_graphs = 10000   # để giống train_devign của bạn

    enc = build_type_encoder(root, labels_file, max_graphs=max_graphs)

    os.makedirs("checkpoints", exist_ok=True)
    out = "checkpoints/type_vocab.pt"
    torch.save(enc.type_vocab, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()