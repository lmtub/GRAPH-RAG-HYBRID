import json
import os

def make_split():
    checkpoint_dir = "checkpoints"
    save_path = os.path.join(checkpoint_dir, "split.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # CON SỐ QUAN TRỌNG: Lấy đúng số lượng mẫu đã lọc (kept) từ log của bạn
    total_kept = 12860 
    
    # Tạo danh sách chỉ số từ 0 đến 12859
    indices = list(range(total_kept))
    
    import random
    random.seed(42)
    random.shuffle(indices)

    # Chia tỉ lệ 80/10/10 dựa trên 12860 mẫu
    train_end = int(total_kept * 0.8)
    val_end = train_end + int(total_kept * 0.1)

    split_data = {
        "train_idx": indices[:train_end],
        "val_idx": indices[train_end:val_end],
        "test_idx": indices[val_end:]
    }

    with open(save_path, "w") as f:
        json.dump(split_data, f)
    
    print(f"✅ Đã tạo file split khớp với Dataset (12.860 mẫu).")
    print(f"📊 Train: {len(split_data['train_idx'])} | Val: {len(split_data['val_idx'])} | Test: {len(split_data['test_idx'])}")

if __name__ == "__main__":
    make_split()