import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, f1_score

from dataset.cpg_dataset_pyg import CPGPyGDataset
from dataset.node_encoder import TypeOnlyEncoder
from dataset.data_builder import build_data_from_cpg
from src.train.model import DevignModel
from src.train.collate_fn import pyg_to_batch_tensors


def build_type_encoder(root: str, labels_file: str) -> TypeOnlyEncoder:
    root_path = Path(root)
    with open(labels_file, "r") as f:
        labels = json.load(f)

    all_nodes = []
    for gid in labels.keys():
        n_path = root_path / gid / "nodes.json"
        if n_path.exists():
            nodes = json.load(open(n_path, "r"))
            all_nodes.append(nodes)

    enc = TypeOnlyEncoder()
    enc.fit(all_nodes)
    return enc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    root = "data/cpg"
    labels_file = "dataset/labels.json"
    ckpt_path = "checkpoints/best_encoder.pt"
    num_edge_types = 5
    hidden_dim = 128
    batch_size = 16

    print("==> Build node encoder & dataset ...")
    node_encoder = build_type_encoder(root, labels_file)
    dataset = CPGPyGDataset(root, labels_file, node_encoder)

    input_dim = dataset[0].x.size(1)
    print(f"Input dim: {input_dim}, num_samples={len(dataset)}")

    model = DevignModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        step=8,
        num_edge_types=num_edge_types,
    ).to(device)

    model.encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: pyg_to_batch_tensors(batch, num_edge_types=num_edge_types),
    )

    all_labels = []
    all_preds = []

    print("==> Evaluating ...")
    with torch.no_grad():
        for node_feat, adj, labels, _ in loader:
            node_feat = node_feat.to(device)
            adj = adj.to(device)

            logits, _, _ = model(node_feat, adj)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()
            y = labels.cpu().numpy()

            all_labels.extend(y.tolist())
            all_preds.extend(preds.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n[Eval] Accuracy: {acc:.4f}")
    print(f"[Eval] F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
