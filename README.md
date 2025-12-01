# GRAPH-RAG-HYBRID
## 🔍 Hybrid Graph-RAG for Vulnerability Detection
A hybrid architecture combining Devign’s GNN-based structural understanding of source code 
and Vul-RAG’s knowledge-level retrieval for enhanced vulnerability detection and explainability.

### Objectives
- Capture semantic & structural patterns of C/C++ functions via Code Property Graph (CPG)
- Retrieve and fuse security knowledge (CWE/CVE) to augment detection

- Provide interpretable vulnerability reasoning (attention maps + CWE context)

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





