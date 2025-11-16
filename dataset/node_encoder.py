import torch

class TypeOnlyEncoder:
    """
    Node encoder dựa trên TYPE -> one-hot vector.
    """

    def __init__(self):
        self.type_vocab = {}
        self.fitted = False

    def fit(self, all_nodes_lists):
        for nodes in all_nodes_lists:
            for n in nodes:
                node_type = n.get("attrs", {}).get("TYPE", "<UNK>")
                if node_type not in self.type_vocab:
                    self.type_vocab[node_type] = len(self.type_vocab)

        self.fitted = True
        print(f"[Encoder] fitted with {len(self.type_vocab)} node types.")

    @property
    def feat_dim(self):
        return len(self.type_vocab)

    def __call__(self, nodes):
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before use")

        x = torch.zeros(len(nodes), self.feat_dim, dtype=torch.float32)

        for i, n in enumerate(nodes):
            t = n.get("attrs", {}).get("TYPE", "<UNK>")
            if t in self.type_vocab:
                x[i, self.type_vocab[t]] = 1.0

        return x
