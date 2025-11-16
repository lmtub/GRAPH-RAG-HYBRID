import os
import json
from pathlib import Path

root = Path("../data/cpg")   # từ thư mục dataset/

labels = {}

for folder in root.iterdir():
    if folder.is_dir() and folder.name.endswith(".cpg14"):
        parts = folder.name.split("_")
        if len(parts) >= 2:
            label = parts[1].split(".")[0]
            labels[folder.name] = int(label)

with open("labels.json", "w") as f:
    json.dump(labels, f, indent=4)

print("Generated labels.json with", len(labels), "entries")
