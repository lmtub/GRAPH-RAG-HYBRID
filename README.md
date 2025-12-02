# GRAPH-RAG-HYBRID
## 🔍 Hybrid Graph-RAG for Vulnerability Detection
A hybrid architecture combining Devign’s GNN-based structural understanding of source code 
and Vul-RAG’s knowledge-level retrieval for enhanced vulnerability detection and explainability.

## 👥 Team Members

| Name | ID |  Role |  Key Contributions |
|----------|---------------|---------|-----------------------|
| **Lê Minh Tấn** | 23523198 | **Lead Engineer – Graph Pipeline** | CPG parsing, dataset creation, GGNN encoder, training, FAISS index, end-to-end pipeline |
| **Đặng Gia Nghĩa** | 23521006 | **Dataset Engineer** | Data cleaning, loader, PyG dataset integration |
| **Nguyễn Quang Thắng** | 23521425 | **Fusion/RAG Engineer** | Hybrid fusion module, retrieval integration, reasoning prompt builder |
| **Nguyễn Lê Phúc Lâm** | 24520934 | **Model Training Engineer** | Training Devign, hyperparameters, evaluation, checkpoints |


### Objectives
- Trích xuất embedding từ CPG thể hiện AST, CFG, PDG pattern
- Dò tìm các hàm tương tự qua FAISS vector search
- Kết hợp Devign + similarity → hybrid fusion score
- Tạo prompt Graph-RAG có ngữ cảnh (target graph + similar graphs)
- Chuẩn bị cho tích hợp LLM reasoning (GPT/Llama/CWE/CVE KB)

### Folder Structure
```text
GRAPH-RAG-HYBRID/
│
├── data/
│   ├── cpg/                     # CPG graphs parsed by Joern
│   ├── embeddings/             # devign_embeddings.pt
│   ├── index/                  # FAISS index + meta.json
│   ├── raw/                    # raw Devign dataset
│
├── dataset/
│   ├── cpg_dataset_pyg.py      # PyG dataset loader
│   ├── data_builder.py         # Build graph from JSON
│   ├── node_encoder.py         # TypeOnlyEncoder
│   ├── labels.json             # Devign labels
│
├── src/
│   ├── devign_model/           # GGNN encoder + Devign model
│   │   ├── encoder.py
│   │   ├── model.py
│   │
│   ├── fusion_layer/           # Hybrid fusion score (zg + zl)
│   │   └── fusion_layer.py
│   │
│   ├── rag_module/             # Retrieval + prompt generation
│   │   ├── graph_rag.py
│   │   ├── fusion_layer.py
│   │   ├── llm_rag.py
│   │   └── prompt_templates.py
│   │
│   ├── vector_db/              # FAISS search
│   │   ├── build_faiss.py
│   │   └── search_faiss.py
│   │
│   ├── train/
│   │   ├── train_devign.py     # training pipeline
│   │   ├── export_embeddings.py
│   │   ├── collate_fn.py
│   │   └── model.py
│   │
│   ├── eval/
│   │   └── eval_devign.py      # evaluation script
│   │
│   └── demo/
│       └── end2end_demo.py     # hybrid Graph-RAG demo
│
├── runs/                       # logs, saved prompts, experiment outputs
├── checkpoints/                # best_encoder.pt
│
├── Dockerfile
├── requirements.txt
└── README.md

```

### How to Run (End2end)
**1. Export embeddings:** python -m src.train.export_embeddings /n
**2. Build FAISS index:** python -m src.vector_db.build_faiss
**3. Run demo:** 
  - Random: python -m src.demo.end2end_demo
  - Chỉ định graph: python -m src.demo.end2end_demo --graph-id 0_0.cpg14
  - Lưu prompt: python -m src.demo.end2end_demo --save runs/prompt.txt
  - Evaluation (Devign): python -m src.eval.eval_devign
      *expected: F1 ~0.64*









