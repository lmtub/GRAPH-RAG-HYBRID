import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GlobalAttention

class GraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=3): # Quay về 4 heads, 3 layers
        super().__init__()
        self.lin_input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim)) # Quay về LayerNorm ổn định

    def forward(self, x, edge_index):
        x = self.lin_input(x)
        for conv, norm in zip(self.layers, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            x = x + F.gelu(h) 
        return x

class DevignModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.encoder = GraphTransformerEncoder(input_dim, hidden_dim, num_heads=num_heads)
        self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            # Giảm Dropout xuống 0.4 như bản 60%
            nn.Dropout(p=0.4), 
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        node_embeddings = self.encoder(x, edge_index)
        graph_embedding = self.pool(node_embeddings, batch)
        logits = self.classifier(graph_embedding)
        return logits.squeeze(-1), None