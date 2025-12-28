import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset.cpg_dataset_pyg import CPGPyGDataset
from dataset.node_encoder import TypeOnlyEncoder
from src.train.model import DevignModel
from src.train.collate_fn import pyg_to_batch_tensors


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    root = "data/cpg"
    labels_file = "dataset/labels.json"
    ckpt_path = "checkpoints/best_encoder.pt"
    vocab_path = "checkpoints/type_vocab.pt"
    split_path = "checkpoints/split.json"

    num_edge_types = 5
    hidden_dim = 64          # ✅ PHẢI khớp checkpoint của bạn
    batch_size = 16
    threshold = 0.5

    # ---- load vocab (không fit lại) ----
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"Missing {vocab_path}. Hãy chạy train_devign trước.")
    node_encoder = TypeOnlyEncoder()
    node_encoder.type_vocab = torch.load(vocab_path, map_location="cpu")
    node_encoder.fitted = True
    print(f"[Encoder] loaded vocab with {len(node_encoder.type_vocab)} node types")

    # ---- dataset ----
    dataset = CPGPyGDataset(root, labels_file, node_encoder, max_nodes=500, verbose=True)
    input_dim = dataset[0].x.size(1)
    print(f"Input dim: {input_dim}, total={len(dataset)}")

    # ---- load split + lấy test indices ----
    if not Path(split_path).exists():
        raise FileNotFoundError(f"Missing {split_path}. Hãy chạy: python -m src.eval.make_split")
    split = json.load(open(split_path, "r", encoding="utf-8"))
    test_idx = split["test_idx"]

    test_set = Subset(dataset, test_idx)
    print(f"[Info] Evaluating on TEST only: {len(test_set)} graphs")

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: pyg_to_batch_tensors(batch, num_edge_types=num_edge_types),
    )

    # ---- model ----
    model = DevignModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        step=8,
        num_edge_types=num_edge_types,
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []

    print("==> Evaluating ...")
    with torch.no_grad():
        for node_feat, adj, labels, _ in loader:
            node_feat = node_feat.to(device)
            adj = adj.to(device)

            logits, _, _ = model(node_feat, adj)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype("int32")
            y = labels.cpu().numpy().astype("int32")

            all_labels.extend(y.tolist())
            all_preds.extend(preds.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_bin = f1_score(all_labels, all_preds)                 # class=1
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print("\n[Eval on TEST]")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 (pos) : {f1_bin:.4f}")
    print(f"F1 macro : {f1_macro:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix [ [TN FP] [FN TP] ]:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    main()