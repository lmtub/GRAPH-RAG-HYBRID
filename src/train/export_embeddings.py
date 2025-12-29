import os
import json
from pathlib import Path

from dataset import node_encoder
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from dataset.cpg_dataset_pyg import CPGPyGDataset
from dataset.node_encoder import CombinedW2VEncoder

from src.train.collate_fn import pyg_to_batch_tensors
from src.train.model import DevignModel


# ====== Reuse logic: build & fit node encoder ======
def build_type_encoder(root: str, labels_file: str):
    root_path = Path(root)
    labels_path = Path(labels_file)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    all_nodes_lists = []

    for name in labels.keys():
        graph_dir = root_path / name
        nodes_path = graph_dir / "nodes.json"
        if nodes_path.is_file():
            with open(nodes_path, "r", encoding="utf-8") as nf:
                nodes = json.load(nf)
                all_nodes_lists.append(nodes)

    encoder = TypeOnlyEncoder()
    encoder.fit(all_nodes_lists)
    return encoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- config giống train_devign ----
    root = "data/cpg"
    labels_file = "dataset/labels.json"
    num_edge_types = 5
    #hidden_dim = 64
    #step = 8
    batch_size = 16
    ckpt_path = "checkpoints/best_encoder.pt"
    out_path = "data/embeddings/devign_embeddings.pt"

    os.makedirs("data/embeddings", exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

# ---- load encoder vocab (KHÔNG fit lại) ----
    vocab_path = "checkpoints/type_vocab.pt"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
        f"Missing {vocab_path}. Hãy chạy train_devign trước để tạo type_vocab.pt"
        )

    node_encoder = CombinedW2VEncoder(w2v_model_path="/app/word2vec_cpg.model")
    node_encoder.type_vocab = torch.load("checkpoints/type_vocab.pt")
    node_encoder.fitted = True
    print(f"[Encoder] loaded vocab with {len(node_encoder.type_vocab)} node types.")

# ---- dataset full (không split train/val/test nữa) ----
    dataset = CPGPyGDataset(
        root=root,
        labels_file=labels_file,
        node_encoder=node_encoder,
        make_undirected=True,
    )
    print("Total graphs:", len(dataset))

    sample = dataset[0]
    input_dim = sample.x.size(1)
    print("Node feature dim:", input_dim)

    collate = lambda batch: pyg_to_batch_tensors(batch, num_edge_types=num_edge_types)

    loader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        #collate_fn=collate,
    )

    # ---- Devign model + load best encoder ----
    model = DevignModel(
        input_dim=node_encoder.feat_dim,
        hidden_dim=64,
        num_heads=4
        #step=step,
        #num_edge_types=num_edge_types,
    ).to(device)

    #encoder_state = torch.load(torch.load(ckpt_path, map_location=device))
    model.encoder.load_state_dict(torch.load("checkpoints/best_encoder.pt"))
    model.eval()

    # ---- duyệt toàn bộ graph -> lấy graph_embedding ----
    all_embeddings = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Lấy node embeddings từ encoder
            node_emb = model.encoder(batch.x, batch.edge_index)
            # Dùng pooling để tạo ra 1 vector duy nhất cho cả graph (Graph Embedding)
            graph_emb = model.pool(node_emb, batch.batch)

            all_embeddings.append(graph_emb.cpu())
            all_labels.append(batch.y.cpu())
            all_ids.extend(batch.graph_id)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)        # (N_graphs,)

    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)
    print("Num graph_ids:", len(all_ids))

    # ---- save ra 1 file duy nhất ----
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "graph_ids": all_ids,
        },
        out_path,
    )

    print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
