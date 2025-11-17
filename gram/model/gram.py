import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


# ============================================================
# Load ancestor tree (always return a single ancestor chain per code)
# ============================================================
def load_tree(tree_prefix, device="cpu"):
    """
    Return:
       ancestors: tensor shape (C, K)
          where K = length of ancestor chain including leaf
          Example: [leaf, A5, A4, A3, A2, A1, ROOT]
    """
    full_chain = []

    # level5 → level1
    levels = []
    for i in range(5, 0, -1):
        path = f"{tree_prefix}.level{i}.pk"
        try:
            tree = pickle.load(open(path, "rb"))
            if len(tree) == 0:
                continue
            levels.append(tree)
        except:
            pass

    # build a full chain for every leaf code
    # all level dicts have same keys: leaf_index → list of ancestors
    # we concatenate levels vertically
    master_keys = list(levels[0].keys())
    C = len(master_keys)

    # Determine K
    K = sum(len(levels[i][master_keys[0]]) for i in range(len(levels)))

    ancestors = torch.zeros((C, K), dtype=torch.long, device=device)

    ptr = 0
    for level in levels:
        for leaf_idx, leaf_key in enumerate(master_keys):
            arr = torch.tensor(level[leaf_key], dtype=torch.long, device=device)
            ancestors[leaf_idx, ptr:ptr+len(arr)] = arr
        ptr += len(arr)

    return ancestors  # shape (C, K)


# ============================================================
# GRAM model — ORIGINAL (Theano version)
# ============================================================
class GRAM(nn.Module):
    def __init__(
        self,
        input_dim,          # number of leaf codes (C)
        num_classes,
        ancestors_tensor,   # shape (C, K)
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        device="cpu"
    ):
        super().__init__()
        self.C = input_dim
        self.num_classes = num_classes
        self.device = device

        # ancestors tensor: (C, K)
        self.ancestors = ancestors_tensor
        self.K = ancestors_tensor.shape[1]

        # max index in tree
        self.num_embeddings = int(ancestors_tensor.max().item()) + 1

        # embedding matrix
        self.W_emb = nn.Embedding(self.num_embeddings, emb_dim)

        # attention
        self.W_att = nn.Linear(emb_dim * 2, att_dim)
        self.v_att = nn.Linear(att_dim, 1, bias=False)

        # GRU
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=False
        )

        # output
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    # --------------------------------------------------------
    # Compute embedding for all codes
    # --------------------------------------------------------
    def compute_gram_embedding(self):
        """
        Return embedding E shape (C, D)
        """

        leaf_ids = self.ancestors[:, 0]   # leaf index
        anc_ids  = self.ancestors         # full chain

        leaf_emb = self.W_emb(leaf_ids)              # (C, D)
        anc_emb  = self.W_emb(anc_ids)               # (C, K, D)

        # repeat leaf embedding as (C, K, D)
        leaf_rep = leaf_emb.unsqueeze(1).repeat(1, self.K, 1)

        att_input = torch.cat([leaf_rep, anc_emb], dim=-1)   # (C, K, 2D)
        h = torch.tanh(self.W_att(att_input))                # (C, K, A)
        alpha = F.softmax(self.v_att(h).squeeze(-1), dim=-1) # (C, K)

        # sum of ancestor embeddings weighted by alpha
        e = torch.sum(anc_emb * alpha.unsqueeze(-1), dim=1)  # (C, D)
        return e

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x, mask):
        """
        x:    (T, B, C)
        mask: (T, B)
        """

        T, B, C = x.shape

        # embedding (C, D)
        E = self.compute_gram_embedding()

        # visit embedding
        x_flat = x.reshape(T * B, C)
        visit_emb = torch.matmul(x_flat, E)        # (T*B, D)
        visit_emb = visit_emb.reshape(T, B)

        # GRU
        h, _ = self.gru(visit_emb)
        h = h * mask.unsqueeze(-1)

        logits = self.output_layer(h)
        y_hat = torch.sigmoid(logits)
        return y_hat
