import torch
import torch.nn as nn

class GGNNEncoder(nn.Module):
    def __init__(self, input_dim, step, output_dim, num_edge_types=5):
        super(GGNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.step = step
        self.output_dim = output_dim
        self.num_edge_types = num_edge_types

        self.fc_in = nn.Linear(input_dim, output_dim)

        self.fc_eq3_w = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_edge_types)
        ])
        
        self.gru = nn.GRUCell(output_dim, output_dim)

        self.conv_l1 = nn.Conv1d(output_dim, output_dim, kernel_size=1)
        self.conv_l2 = nn.Conv1d(output_dim, output_dim, kernel_size=1)

    def forward(self, node_features, adj_matrices):
        batch_size = node_features.size(0)
        max_nodes = node_features.size(1)

        hidden_state = self.fc_in(node_features)
        hidden_state = hidden_state.view(-1, self.output_dim)

        for _ in range(self.step):
            h_curr = hidden_state.view(batch_size, max_nodes, self.output_dim)
            accumulated_messages = 0
            
            for edge_type in range(self.num_edge_types):
                adj = adj_matrices[:, edge_type, :, :]
                msg = torch.bmm(adj, h_curr)
                msg = self.fc_eq3_w[edge_type](msg.view(-1, self.output_dim))
                accumulated_messages += msg
            
            hidden_state = self.gru(accumulated_messages, hidden_state)

        node_states = hidden_state.view(batch_size, max_nodes, self.output_dim)

        check_input = node_states.permute(0, 2, 1)
        
        attn_logits = self.conv_l1(check_input)
        attn_weights = torch.sigmoid(attn_logits)
        
        features_transformed = torch.tanh(self.conv_l2(check_input))

        graph_embedding = torch.sum(attn_weights * features_transformed, dim=2)
        attn_map = torch.mean(attn_weights, dim=1)

        return graph_embedding, node_states, attn_map