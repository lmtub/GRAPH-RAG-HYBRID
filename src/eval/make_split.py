import json
from pathlib import Path
import torch

from dataset.cpg_dataset_pyg import CPGPyGDataset
from dataset.node_encoder import TypeOnlyEncoder


def main():
    root = "data/cpg"
    labels_file = "dataset/labels.json"
    vocab_path = "checkpoints/type_vocab.pt"
    out_path = "checkpoints/split.json"

    train_ratio = 0.8
    val_ratio = 0.1
    seed = 42

    # load vocab (không fit lại)
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"Missing {vocab_path}. Hãy chạy train_devign trước.")

    enc = TypeOnlyEncoder()
    enc.type_vocab = torch.load(vocab_path, map_location="cpu")
    enc.fitted = True

    # dataset (PHẢI giống train_devign: max_nodes=500)
    ds = CPGPyGDataset(root, labels_file, enc, max_nodes=500, verbose=True)
    n = len(ds)
    print("[make_split] len(dataset) =", n)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "seed": seed,
            "n": n,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        },
        open(out_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    print("[make_split] Saved:", out_path)
    print("[make_split] sizes:", len(train_idx), len(val_idx), len(test_idx))
    print("[make_split] max(test_idx) =", max(test_idx), "len(dataset) =", n)


if __name__ == "__main__":
    main()