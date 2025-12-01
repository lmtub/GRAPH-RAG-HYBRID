"""
Module: prompt_templates.py

Chứa các hàm build prompt cho LLM trong bối cảnh Graph-RAG.
Mục tiêu: từ context (code graph + similar graphs + scores + extra knowledge),
tạo ra 1 chuỗi prompt hoàn chỉnh, có cấu trúc, dễ mở rộng (thêm CWE/CVE sau này).
"""

from typing import List, Dict, Optional
import textwrap


def _summarize_graph(nodes: List[Dict], edges: List[Dict],
                     max_nodes: int = 30, max_edges: int = 50) -> str:
    """
    Tạo summary text ngắn gọn cho 1 graph:
    - Liệt kê node: id, TYPE, optional CODE / NAME
    - Liệt kê một số edge: src --EDGE_TYPE--> dst
    Giới hạn số lượng node/edge để prompt không quá dài.
    """
    lines = []

    # ---- Nodes ----
    lines.append("Nodes (truncated):")
    for i, n in enumerate(nodes):
        if i >= max_nodes:
            lines.append(f"... ({len(nodes) - max_nodes} more nodes omitted)")
            break

        nid = n.get("id", i)
        attrs = n.get("attrs", {})
        ntype = attrs.get("TYPE", "UNKNOWN")
        code = attrs.get("CODE") or attrs.get("NAME") or ""
        code = code.replace("\n", " ")[:120]

        if code:
            lines.append(f"  - Node #{nid} | TYPE={ntype} | CODE={code}")
        else:
            lines.append(f"  - Node #{nid} | TYPE={ntype}")

    # ---- Edges ----
    lines.append("\nEdges (truncated):")
    for i, e in enumerate(edges):
        if i >= max_edges:
            lines.append(f"... ({len(edges) - max_edges} more edges omitted)")
            break

        src = e.get("src")
        dst = e.get("dst")
        etype = e.get("type", "EDGE")
        lines.append(f"  - {src} --{etype}--> {dst}")

    return "\n".join(lines)


def build_graph_vuln_prompt(
    target_graph: Dict,
    similar_graphs: List[Dict],
    user_query: Optional[str] = None,
    extra_knowledge: Optional[str] = None,
) -> str:
    """
    Build prompt cho LLM phân tích 1 graph code + các graph tương tự.

    target_graph: dict chứa:
        - graph_id, devign, fusion, similarity, label, nodes, edges
    similar_graphs: list các dict tương tự (context thêm)
    user_query: câu hỏi cụ thể của user (nếu có)
    extra_knowledge: block text tri thức thêm (CWE/CVE...) -> sẽ feed sau.

    Trả về: 1 string prompt hoàn chỉnh.
    """

    gid = target_graph.get("graph_id", "<unknown>")
    devign_score = target_graph.get("devign", None)
    fusion_score = target_graph.get("fusion", None)
    sim_score = target_graph.get("similarity", None)
    label = target_graph.get("label", None)

    # 1) Header
    header = f"""\
    You are a security analysis assistant focusing on C/C++ vulnerability detection.

    Your task is to analyze a target function represented as a code property graph (CPG),
    together with several similar vulnerable/non-vulnerable examples, and answer about
    its potential vulnerabilities, risk level, and reasoning.

    --- USER QUESTION ---
    {user_query or "Please analyze whether the following function is vulnerable. Explain your reasoning step by step."}
    """

    header = textwrap.dedent(header).strip()

    # 2) Target graph section
    target_summary = _summarize_graph(
        target_graph.get("nodes", []),
        target_graph.get("edges", []),
        max_nodes=40,
        max_edges=60,
    )

    target_section = f"""\
    === TARGET GRAPH ===
    Graph ID: {gid}
    Devign vulnerability score (0~1): {devign_score}
    Similarity score (0~1): {sim_score}
    Fusion score (0~1): {fusion_score}
    Dataset label (0/1, if available): {label}

    [Target graph summary]
    {target_summary}
    """

    target_section = textwrap.dedent(target_section).rstrip()

    # 3) Similar graphs
    sim_sections = []
    for i, g in enumerate(similar_graphs, start=1):
        sgid = g.get("graph_id", "<unknown>")
        sdev = g.get("devign", None)
        sfus = g.get("fusion", None)
        ssim = g.get("similarity", None)
        slabel = g.get("label", None)

        gsum = _summarize_graph(
            g.get("nodes", []),
            g.get("edges", []),
            max_nodes=15,
            max_edges=25,
        )

        block = f"""\
        --- Similar example #{i} ---
        Graph ID: {sgid}
        Devign score: {sdev}
        Similarity: {ssim}
        Fusion: {sfus}
        Label: {slabel}

        {gsum}
        """
        sim_sections.append(textwrap.dedent(block).rstrip())

    similar_section = "=== SIMILAR GRAPHS CONTEXT ===\n" + "\n\n".join(sim_sections)

    # 4) Extra knowledge (CWE/CVE...) – để trống, sau này fill
    if extra_knowledge:
        knowledge_section = f"""\
        === VULNERABILITY PATTERNS / KNOWLEDGE ===
        {extra_knowledge}
        """
        knowledge_section = textwrap.dedent(knowledge_section).rstrip()
    else:
        knowledge_section = "=== VULNERABILITY PATTERNS / KNOWLEDGE ===\n(No external knowledge provided yet.)"

    # 5) Instructions
    instructions = """\
    === INSTRUCTIONS ===
    1. First, restate briefly what the target function seems to do.
    2. Then, identify any potential vulnerability patterns in the target graph
       (e.g., buffer overflow, use-after-free, null dereference, integer overflow, etc.).
    3. Use the similar examples to support your reasoning (compare patterns, flows, APIs).
    4. Provide a clear conclusion:
       - Is this function likely vulnerable? (Yes/No/Uncertain)
       - If vulnerable, what is the most likely vulnerability type?
    5. Optionally, suggest high-level mitigation ideas.
    6. Answer in a concise, structured way.
    """
    instructions = textwrap.dedent(instructions).rstrip()

    parts = [
        header,
        "",
        target_section,
        "",
        similar_section,
        "",
        knowledge_section,
        "",
        instructions,
    ]
    prompt = "\n".join(parts)
    return prompt
