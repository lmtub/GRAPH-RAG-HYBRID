import torch


def pyg_to_batch_tensors(batch, num_edge_types: int = 5):
    """
    Chuyển list[PyG Data] -> 
      - node_features: (B, N, F)
      - adj_matrices : (B, E, N, N)
      - labels       : (B,)
      - graph_ids    : list[str]
    """

    labels = []
    graph_ids = []
    node_feat_list = []
    adj_list = []

    # Tìm số node lớn nhất trong batch để padding
    max_nodes = max(data.x.size(0) for data in batch)
    feat_dim = batch[0].x.size(1)

    for data in batch:
        N = data.x.size(0)

        # 1) Pad node features -> (max_nodes, feat_dim)
        x = torch.zeros((max_nodes, feat_dim), dtype=torch.float32)
        x[:N] = data.x
        node_feat_list.append(x)

        # 2) Tạo adjacency matrices -> (num_edge_types, max_nodes, max_nodes)
        adj = torch.zeros(
            (num_edge_types, max_nodes, max_nodes),
            dtype=torch.float32
        )

        # edge_index: (2, num_edges) -> (num_edges, 2)
        edges = data.edge_index.t()  # (E, 2)

        # Nếu Data có edge_type thì dùng, không thì cho tất cả = 0
        if hasattr(data, "edge_type"):
            edge_types = data.edge_type
        else:
            edge_types = torch.zeros(edges.size(0), dtype=torch.long)

        # Fill vào adj
        for (u, v), t in zip(edges, edge_types):
            t = int(t)
            if 0 <= t < num_edge_types:
                adj[t, u, v] = 1.0

        adj_list.append(adj)

        # 3) Label & graph_id
        labels.append(int(data.y))
        graph_ids.append(
            getattr(data, "graph_id", "")  # nếu không có graph_id thì để rỗng
        )

    node_features = torch.stack(node_feat_list)  # (B, N, F)
    adj_matrices = torch.stack(adj_list)        # (B, E, N, N)
    labels = torch.tensor(labels, dtype=torch.float32)

    return node_features, adj_matrices, labels, graph_ids
