import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dataset.cpg_dataset_pyg import CPGPyGDataset
from dataset.node_encoder import TypeOnlyEncoder

from src.train.collate_fn import pyg_to_batch_tensors
from src.train.model import DevignModel


# =========================
# 1. Build & fit node encoder
# =========================
def build_type_encoder(root: str, labels_file: str, max_graphs: int = 500):
    """
    Fit TypeOnlyEncoder trên 1 subset graph
    để tránh vocab quá lớn gây OOM.
    """
    root_path = Path(root)
    labels = json.load(open(labels_file, "r", encoding="utf-8"))

    all_nodes_lists = []
    cnt = 0

    for name in labels.keys():
        if cnt >= max_graphs:
            break

        nodes_path = root_path / name / "nodes.json"
        if nodes_path.exists():
            nodes = json.load(open(nodes_path, "r", encoding="utf-8"))
            all_nodes_lists.append(nodes)
            cnt += 1

    encoder = TypeOnlyEncoder()
    encoder.fit(all_nodes_lists)

    print(f"[Info] Encoder fitted on {cnt} graphs")
    return encoder

# =========================
# 2. Train / Eval loops
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer=None):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False)

    for step, (node_feat, adj, labels, _) in enumerate(pbar, start=1):
        node_feat = node_feat.to(device)           # (B, N, F)
        adj = adj.to(device)                       # (B, E, N, N)
        labels = labels.float().to(device)         # (B,)

        logits, _, _ = model(node_feat, adj)       # (B,)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            correct = (preds == labels.long()).sum().item()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # update progress bar
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    if writer is not None:
        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, phase="Val", writer=None):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"[{phase}] Epoch {epoch}", leave=False)

    for node_feat, adj, labels, _ in pbar:
        node_feat = node_feat.to(device)
        adj = adj.to(device)
        labels = labels.float().to(device)

        logits, _, _ = model(node_feat, adj)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        correct = (preds == labels.long()).sum().item()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    if writer is not None:
        writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
        writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


# =========================
# 3. Tạo DataLoader với split train/val/test
# =========================
def create_dataloaders(
    dataset,
    batch_size: int = 32,
    num_edge_types: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Chia dataset thành train / val / test theo tỉ lệ,
    trả về 3 DataLoader tương ứng.
    """
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    collate = lambda batch: pyg_to_batch_tensors(batch, num_edge_types=num_edge_types)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    return train_loader, val_loader, test_loader


# =========================
# 4. Main training routine (with early stopping + TB logging)
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # ----- Path config -----
    root = "data/cpg"
    labels_file = "dataset/labels.json"
    num_edge_types = 5

    # 🔽 GIẢM MẠNH ĐỂ TRÁNH OOM
    batch_size = 8
    hidden_dim = 64
    ggnn_step = 4
    max_epochs = 10

    lr = 1e-4
    weight_decay = 1e-5
    patience = 3

    # 🔽 CHỈ TRAIN SUBSET (RẤT QUAN TRỌNG)
    #max_train_graphs = 2000

    os.makedirs("checkpoints", exist_ok=True)

    # ----- Build & fit node encoder -----
    node_encoder = build_type_encoder(root, labels_file, max_graphs=500)
    # Save encoder vocab to reuse in export_embeddings
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(node_encoder.type_vocab, "checkpoints/type_vocab.pt")
    print("[Info] Saved type_vocab -> checkpoints/type_vocab.pt")

    # ----- Dataset -----
    full_dataset = CPGPyGDataset(
        root=root,
        labels_file=labels_file,
        node_encoder=node_encoder,
        make_undirected=True,
        max_nodes=500,
    )
    print("Total graphs:", len(full_dataset))

    # 🔽 LẤY SUBSET DATASET
    #indices = torch.randperm(len(full_dataset))[:max_train_graphs]
    #dataset = Subset(full_dataset, indices)
    #print(f"[Info] Training on subset: {len(dataset)} graphs")

    # Lấy input_dim
    sample = full_dataset[0]
    input_dim = sample.x.size(1)
    print("Node feature dim:", input_dim)

    # ----- Dataloaders -----
    train_loader, val_loader, test_loader = create_dataloaders(
        full_dataset,
        batch_size=batch_size,
        num_edge_types=num_edge_types,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    # ----- Model -----
    model = DevignModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        step=ggnn_step,
        num_edge_types=num_edge_types,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    writer = SummaryWriter(log_dir="runs/devign_experiment")

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_encoder_path = "checkpoints/best_encoder.pt"

    # ----- Training loop -----
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, phase="Val", writer=writer
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.encoder.state_dict(), best_encoder_path)
            print("  -> Saved best encoder")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    print("\nEvaluating on TEST set...")
    model.encoder.load_state_dict(torch.load(best_encoder_path, map_location=device))
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, epoch=0, phase="Test", writer=writer
    )
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")

    writer.close()
if __name__ == "__main__":
    main()