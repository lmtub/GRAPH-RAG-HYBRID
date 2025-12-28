import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader # Dùng DataLoader của PyG
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from dataset.cpg_dataset_pyg import CPGPyGDataset
from src.train.model import DevignModel
from dataset.node_encoder import CombinedW2VEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils import dropout_edge

    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_type_encoder(root: str, labels_file: str, max_graphs: int = 10000):
    root_path = Path(root)
    labels = json.load(open(labels_file, "r", encoding="utf-8"))
    all_nodes_lists = []
    cnt = 0
    for name in labels.keys():
        if cnt >= max_graphs: break
        nodes_path = root_path / name / "nodes.json"
        if nodes_path.exists():
            nodes = json.load(open(nodes_path, "r", encoding="utf-8"))
            all_nodes_lists.append(nodes)
            cnt += 1
    encoder = CombinedW2VEncoder(w2v_model_path="word2vec_cpg.model")
    encoder.fit(all_nodes_lists)
    return encoder

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False)

    for data in pbar:
        x, edge_index, batch, labels = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.float().to(device)

        # Giữ lại Node Masking 0.15 (Vốn có ở bản 60%)
        mask = torch.rand(x.size(0), device=device) > 0.15
        x = x * mask.view(-1, 1)

        logits, _ = model(x, edge_index, batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).long()
        total_correct += (preds == labels.long()).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}", acc=f"{total_correct/total_samples:.4f}")

    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, phase="Val"):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_labels = [], []
    
    for data in tqdm(loader, desc=f"[{phase}] Epoch {epoch}", leave=False):
        x, edge_index, batch, labels = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.float().to(device)
        logits, _ = model(x, edge_index, batch)
        loss = criterion(logits, labels)
        
        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    all_labels_np, all_preds_np = np.array(all_labels), np.array(all_preds)
    acc = (all_preds_np == all_labels_np).mean()
    f1 = f1_score(all_labels_np, all_preds_np, zero_division=0)
    pre = precision_score(all_labels_np, all_preds_np, zero_division=0)
    rec = recall_score(all_labels_np, all_preds_np, zero_division=0)

    return avg_loss, acc, f1, pre, rec

def create_dataloaders(dataset, split_path="checkpoints/split.json", batch_size=32):
    with open(split_path, "r") as f:
        split = json.load(f)
    # PyG DataLoader tự động xử lý cạnh và nút cho Transformer
    train_loader = DataLoader(Subset(dataset, split["train_idx"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, split["val_idx"]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, split["test_idx"]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root, labels_file = "data/cpg", "dataset/labels.json"
    batch_size, hidden_dim, max_epochs = 16, 128, 100
    lr, weight_decay, patience = 1e-4, 1e-3, 15

    # 1. Tạo bộ mã hóa
    node_encoder = build_type_encoder(root, labels_file)

    # --- THÊM 2 DÒNG NÀY VÀO ĐÂY ---
    os.makedirs("checkpoints", exist_ok=True) # Tạo thư mục nếu chưa có
    torch.save(node_encoder.type_vocab, "checkpoints/type_vocab.pt") 
    print("✅ Đã lưu file vocab tại checkpoints/type_vocab.pt")
    # ------------------------------

    full_dataset = CPGPyGDataset(root=root, labels_file=labels_file, node_encoder=node_encoder, make_undirected=True)
    train_loader, val_loader, test_loader = create_dataloaders(full_dataset, batch_size=batch_size)

    model = DevignModel(input_dim=node_encoder.feat_dim, hidden_dim=hidden_dim).to(device)
    # DÙNG LẠI BCE LOSS VỚI POS_WEIGHT
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(device))
    
    # Quay về Weight Decay 1e-2 để cân bằng giữa học và chống học vẹt
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    writer = SummaryWriter(log_dir="runs/transformer_v1")

    best_val_acc, best_val_loss = 0.0, float('inf')
    epochs_no_improve = 0
    best_model_path = "checkpoints/best_transformer.pt"

    try:
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss, val_acc, val_f1, val_pre, val_rec = evaluate(model, val_loader, criterion, device, epoch, "Val")
    
            print(f"[Epoch {epoch:02d}] Val Acc: {val_acc:.4f} Val Loss: {val_loss:.4f} | Train acc: {train_acc:.4f} Train loss: {train_loss:.4f}| F1: {val_f1:.4f} | Pre: {val_pre:.4f} | Rec: {val_rec:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc, epochs_no_improve = val_acc, 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  ⭐ Lưu model tốt nhất: {val_acc:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience: break
            scheduler.step(val_loss)
            
    except KeyboardInterrupt: print("\n[Dừng thủ công]")

    print("\nEvaluating on TEST set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    # NHẬN ĐỦ 5 GIÁ TRỊ ĐỂ TRÁNH LỖI VALUEERROR
    t_loss, t_acc, t_f1, t_pre, t_rec = evaluate(model, test_loader, criterion, device, 0, "Test")
    print(f"[TEST] Acc: {t_acc:.4f} | F1: {t_f1:.4f} | Precision: {t_pre:.4f} | Recall: {t_rec:.4f}")
    writer.close()

if __name__ == "__main__":
    main()
