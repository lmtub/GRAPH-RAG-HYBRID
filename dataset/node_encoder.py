import torch
import re


def extract_type(label: str):
    """
    Trích TYPE từ label Joern.
    Ví dụ:
      <METHOD<BR/>foo>            -> METHOD
      <CALL<BR/>malloc>           -> CALL
      <IDENTIFIER<BR/>x>          -> IDENTIFIER
      <METHOD_RETURN<BR/>ANY>     -> METHOD_RETURN
    """
    if not label:
        return "<UNK>"

    # Lấy phần trước <BR/>
    # <METHOD<BR/>foo> -> METHOD
    m = re.match(r"<([^<]+?)(?:<BR/>)?", label)
    if m:
        return m.group(1)

    return "<UNK>"


class TypeOnlyEncoder:
    """
    Node encoder dựa trên TYPE (đã chuẩn hóa) -> one-hot vector.
    """

    def __init__(self):
        self.type_vocab = {}
        self.fitted = False

    def fit(self, all_nodes_lists):
        """
        all_nodes_lists: list[list[node_dict]]
        """
        for nodes in all_nodes_lists:
            for n in nodes:
                label = n.get("attrs", {}).get("label", "")
                node_type = extract_type(label)

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
            label = n.get("attrs", {}).get("label", "")
            node_type = extract_type(label)

            if node_type in self.type_vocab:
                x[i, self.type_vocab[node_type]] = 1.0

        return x
