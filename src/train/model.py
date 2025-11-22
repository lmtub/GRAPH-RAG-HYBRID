# model.py
import torch
import torch.nn as nn

from src.train.encoder import GGNNEncoder



class DevignModel(nn.Module):

    def __init__(
        self,
        input_dim: int,    
        hidden_dim: int = 128, 
        step: int = 8,     
        num_edge_types: int = 5
    ):
        super().__init__()

        self.encoder = GGNNEncoder(
            input_dim=input_dim,
            step=step,
            output_dim=hidden_dim,
            num_edge_types=num_edge_types,
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, adj_matrices):
        graph_emb, node_states, attn_map = self.encoder(node_features, adj_matrices)
        logits = self.classifier(graph_emb)
        logits = logits.squeeze(-1)

        return logits, graph_emb, attn_map
