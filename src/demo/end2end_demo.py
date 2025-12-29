import argparse
import random
import os
from src.llm.ollama_client import OllamaClient
from pathlib import Path

from src.vector_db.search_faiss import GraphVectorDB
from src.rag_module.llm_rag import LLMGraphRAGEngine
from src.rag_module.graph_rag import GraphRAG


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end Graph-RAG demo: graph_id -> hybrid search -> prompt"
    )
    p.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="ID của graph (VD: 0_0.cpg14). Nếu bỏ trống sẽ chọn random.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Số lượng similar graphs dùng cho RAG.",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Đường dẫn file để lưu prompt (VD: runs/demo_prompt.txt). Nếu bỏ trống thì không lưu.",
    )
    p.add_argument("--llm", action="store_true", help="Bật gọi LLM (Ollama) để trả lời prompt.")
    p.add_argument("--ollama-url", type=str, default=None, help="VD: http://<WSL_IP>:11434 (override env OLLAMA_BASE_URL)")
    p.add_argument("--ollama-model", type=str, default=None, help="VD: llama3.1:8b (override env OLLAMA_MODEL)")
    p.add_argument("--max-tokens", type=int, default=512, help="Giới hạn output tokens (num_predict).")
    p.add_argument("--temp", type=float, default=0.2, help="temperature cho LLM.")
    p.add_argument("--save-answer", type=str, default=None, help="Lưu câu trả lời LLM ra file.")
    
    return p.parse_args()


def main():
    args = parse_args()

    print("==> Loading FAISS index & GraphVectorDB ...")
    db = GraphVectorDB()

    if not db.graph_ids:
        raise RuntimeError("Không có graph_ids trong FAISS meta. Kiểm tra lại data/index/meta.json")

    if args.graph_id is None:
        graph_id = random.choice(db.graph_ids)
        print(f"[Info] Không truyền --graph-id, chọn random: {graph_id}")
    else:
        graph_id = args.graph_id
        if graph_id not in db.graph_ids:
            raise ValueError(f"graph_id={graph_id} không tồn tại trong index. Thử 1 ID khác từ db.graph_ids")

    print(f"==> Target graph_id: {graph_id}")
    print(f"==> Khởi tạo LLMGraphRAGEngine (k={args.k}) ...")

    engine = LLMGraphRAGEngine(k_default=args.k)

    print("==> Retrieve + build prompt ...")
    out = engine.build_prompt_for_graph(graph_id)

    prompt = out["prompt"]
    target_graph = out["target_graph"]
    similar_graphs = out["similar_graphs"]

    print("\n===== SUMMARY =====")
    print(f"Target graph: {target_graph.get('graph_id')}")
    print(f"  Devign score : {target_graph.get('devign')}")
    print(f"  Similarity   : {target_graph.get('similarity')}")
    print(f"  Fusion       : {target_graph.get('fusion')}")
    print(f"  Label        : {target_graph.get('label')}")

    print(f"\nTop-{len(similar_graphs)} similar graphs (sau fusion):")
    for i, g in enumerate(similar_graphs, start=1):
        print(
            f"  #{i} | id={g.get('graph_id')} "
            f"| devign={g.get('devign'):.4f} "
            f"| sim={g.get('similarity'):.4f} "
            f"| fusion={g.get('fusion'):.4f} "
            f"| label={g.get('label')}"
        )

    print("\n===== PROMPT (preview 800 chars) =====")
    # ====== OPTIONAL: Call LLM (Ollama) ======
    if args.llm:
        base_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL", "http://172.25.35.205:11434")
        model = args.ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

        print("\n==> Calling Ollama LLM ...")
        client = OllamaClient(base_url=base_url, model=model, timeout=300)

        if not client.ping():
            raise RuntimeError(f"Không ping được Ollama tại {base_url}. Kiểm tra ollama đang chạy + port 11434 mở.")

        answer = client.generate(
            prompt=prompt,
            temperature=args.temp,
            num_predict=args.max_tokens,
            stream=False,
        )

        print("\n===== LLM ANSWER =====")
        print(answer)

        if args.save_answer:
            ans_path = Path(args.save_answer)
            ans_path.parent.mkdir(parents=True, exist_ok=True)
            ans_path.write_text(answer, encoding="utf-8")
            print(f"\n[Saved] Answer đã được lưu vào: {ans_path}")
    print(prompt[:800])
    if len(prompt) > 800:
        print("... [truncated] ...")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(prompt, encoding="utf-8")
        print(f"\n[Saved] Prompt đã được lưu vào: {out_path}")


if __name__ == "__main__":
    main()
