import os
import json
import torch
import faiss
import numpy as np

from pathlib import Path


def main():
    emb_path = "data/embeddings/devign_embeddings.pt"
    out_dir = "data/index"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[+] Loading embeddings from {emb_path}")
    data = torch.load(emb_path)

    embeddings = data["embeddings"]        # [N, 128]
    labels = data["labels"]                # [N]
    graph_ids = data["graph_ids"]          # list[str]

    # Convert to numpy float32 (FAISS requirement)
    emb_np = embeddings.numpy().astype("float32")

    # Normalize vector (cosine similarity)
    faiss.normalize_L2(emb_np)

    dim = emb_np.shape[1]  # 128
    num_vec = emb_np.shape[0]

    print(f"[+] Building FAISS index for {num_vec} vectors ({dim} dim)")

    # IndexFlatIP = cosine similarity (after normalize)
    index = faiss.IndexFlatIP(dim)

    index.add(emb_np)
    print("[+] FAISS index built successfully")

    # Save index to disk
    index_path = f"{out_dir}/faiss_index.bin"
    faiss.write_index(index, index_path)
    print(f"[+] Saved FAISS index → {index_path}")

    # Save metadata
    meta = {
        "graph_ids": graph_ids,
        "labels": labels.tolist(),
        "dim": dim,
        "num_vec": num_vec,
        "normalize": True,
        "faiss_type": "IndexFlatIP"
    }

    meta_path = f"{out_dir}/meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[+] Saved metadata → {meta_path}")


if __name__ == "__main__":
    main()
