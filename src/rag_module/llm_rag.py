"""
Module: llm_rag.py

LLM-facing RAG pipeline:
- Nhận graph_id + (optional) user_query
- Dùng GraphRAG để lấy top-k graph context
- Dùng prompt_templates.build_graph_vuln_prompt() để tạo prompt hoàn chỉnh
- (Phần gọi LLM thực tế sẽ được làm ở layer khác / script khác)
"""

from typing import Dict, Any, Optional

from src.rag_module.graph_rag import GraphRAG
from .prompt_templates import build_graph_vuln_prompt



class LLMGraphRAGEngine:
    def __init__(self, k_default: int = 5):
        """
        k_default: số lượng similar graphs mặc định để đưa vào prompt.
        """
        self.k_default = k_default
        self.graph_rag = GraphRAG()

    def build_prompt_for_graph(
        self,
        graph_id: str,
        user_query: Optional[str] = None,
        k: Optional[int] = None,
        extra_knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tạo prompt cho 1 graph_id cụ thể:
        - Retrieve top-k graph bằng GraphRAG
        - Chia 1 cái làm target_graph, phần còn lại làm similar_graphs
        - Build prompt string
        Trả về:
        {
          "prompt": <str>,
          "target_graph": <dict>,
          "similar_graphs": <list[dict]>
        }
        """
        if k is None:
            k = self.k_default

        retrieved = self.graph_rag.retrieve(graph_id, k=k)

        if not retrieved:
            raise ValueError(f"No graph retrieved for id={graph_id}")

        target_graph = retrieved[0]
        similar_graphs = retrieved[1:]

        prompt = build_graph_vuln_prompt(
            target_graph=target_graph,
            similar_graphs=similar_graphs,
            user_query=user_query,
            extra_knowledge=extra_knowledge,
        )

        return {
            "prompt": prompt,
            "target_graph": target_graph,
            "similar_graphs": similar_graphs,
        }
