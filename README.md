# GRAPH-RAG-HYBRID
## 🔍 Hybrid Graph-RAG for Vulnerability Detection
A hybrid architecture combining Devign’s GNN-based structural understanding of source code 
and Vul-RAG’s knowledge-level retrieval for enhanced vulnerability detection and explainability.

### Objectives
- Capture semantic & structural patterns of C/C++ functions via Code Property Graph (CPG)
- Retrieve and fuse security knowledge (CWE/CVE) to augment detection

- Provide interpretable vulnerability reasoning (attention maps + CWE context)

### Folder Structure
hybrid-graph-rag/
├─ data/                      # datasets (Devign, DiverseVul, CWE/CVE KB)
├─ src/
│  ├─ devign_model/          # GNN encoder (GGNN / R-GCN)
│  ├─ rag_module/            # retriever (BM25/Elasticsearch) + LLM reasoning
│  ├─ fusion_layer/          # MLP fusion zg + zl
│  ├─ utils/
│  └─ train.py
├─ notebooks/                # experiments, visualization
├─ docs/                     # papers, design docs
├─ requirements.txt
├─ Dockerfile
└─ README.md



