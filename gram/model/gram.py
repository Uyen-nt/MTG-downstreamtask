import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


# ===============================================================
# LOAD TREE (PAD TO SAME ANCESTOR LENGTH)
# ===============================================================
def load_tree(tree_prefix, device="cpu"):

    raw_leaves = []
    raw_anc = []
    max_K = 0

    # Load levels 5 → 1
    for level in range(5, 0, -1):
        path = f"{tree_prefix}.level{level}.pk"
        try:
            tree = pickle.load(open(path, "rb"))
            if len(tree) == 0:
                print(f"[WARN] {path} empty → skip")
                continue

            leaves = np.array(list(tree.keys()))          # (C,)
            ancestors = np.array(list(tree.values()))     # (C, K)

            raw_leaves.append(leaves)
            raw_anc.append(ancestors)

            max_K = max(max_K, ancestors.shape[1])

        except:
            print(f"[ERR] cannot load {path}")
            continue

    fixed_leaves = []
    fixed_anc = []

    # Pad all ancestor lists to same K
    for leaves, ancestors in zip(raw_leaves, raw_anc):
        C, K = ancestors.shape
        if K < max_K:
            pad_width = max_K - K
            last_col = ancestors[:, -1:]
            pad_block = np.repeat(last_col, pad_width, axis=1)
            ancestors = np.concatenate([ancestors, pad_block], axis=1)

        leaves_exp = np.repeat(leaves[:, None], max_K, axis=1)

        fixed_leaves.append(torch.tensor(leaves_exp, dtype=torch.long, device=device))
        fixed_anc.append(torch.tensor(ancestors, dtype=torch.long, device=device))

    return fixed_leaves, fixed_anc



# ===============================================================
# GRAM MODEL
# ===============================================================
class GRAM(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes,
                 num_levels,
                 emb_dim,
                 att_dim,
                 hidden_dim,
                 tree_leaves,
                 tree_ancestors,
                 max_index_in_tree,
                 device="cpu"):
        
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.hidden_dim = hidden_dim

        # Embedding: must include ancestors (max_index_in_tree)
        self.W_emb = nn.Embedding(max_index_in_tree + 1, emb_dim)

        self.W_att = nn.Linear(emb_dim * 2, att_dim)
        self.v_att = nn.Linear(att_dim, 1, bias=False)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=False
        )

        self.out = nn.Linear(hidden_dim, num_classes)

        self.tree_leaves = tree_leaves
        self.tree_anc = tree_ancestors


    def forward(self, x, mask):
        """
        x: (T, B, input_dim)
        """

        Tt, B, _ = x.shape
        device = x.device

        active_codes = (x.sum(dim=(0,1)) > 0).nonzero(as_tuple=True)[0]
        if len(active_codes) == 0:
            return torch.zeros(Tt, B, self.num_classes, device=device)

        per_level_emb = []

        for leaves, ancestors in zip(self.tree_leaves, self.tree_anc):

            # Filter valid indices
            valid = active_codes[active_codes < leaves.shape[0]]

            leaves_b = leaves[valid]      # (N, K)
            anc_b = ancestors[valid]      # (N, K)

            leaf_emb = self.W_emb(leaves_b)   # (N, K, D)
            anc_emb = self.W_emb(anc_b)       # (N, K, D)

            att_in = torch.cat([leaf_emb, anc_emb], dim=-1)
            att_h = torch.tanh(self.W_att(att_in))
            att_logits = self.v_att(att_h).squeeze(-1)
            att = torch.softmax(att_logits, dim=-1)

            final = (att.unsqueeze(-1) * anc_emb).sum(dim=1)   # (N, D)
            per_level_emb.append(final)

        # Now all are (N, D)
        gram_emb = torch.cat(per_level_emb, dim=-1)    # (N, num_levels*D)

        # Broadcast to full embedding matrix
        code_emb = torch.zeros(self.input_dim, gram_emb.shape[1], device=device)
        code_emb.index_copy_(0, active_codes, gram_emb)

        # Extract first segment (GRAM original logic)
        final_emb = code_emb[:, :self.emb_dim]         # (input_dim, D)

        # Convert x multi-hot to embedding
        x_flat = x.view(-1, self.input_dim)
        visit_emb = torch.tanh(x_flat @ final_emb)      # (T*B, D)
        visit_emb = visit_emb.view(Tt, B, self.emb_dim)

        h, _ = self.gru(visit_emb)
        h = h * mask.unsqueeze(-1)

        y_hat = torch.sigmoid(self.out(h))
        return y_hat



# ===============================================================
# PADDING (OUTSIDE MODEL)
# ===============================================================
def pad_batch(seqs, num_classes, input_dim, device="cpu"):

    lengths = [len(p) - 1 for p in seqs]
    T = max(lengths)
    B = len(seqs)

    x = torch.zeros(T, B, input_dim, device=device)
    y = torch.zeros(T, B, num_classes, device=device)
    mask = torch.zeros(T, B, device=device)

    for b, patient in enumerate(seqs):
        for t in range(len(patient)-1):
            for c in patient[t]:
                x[t, b, c] = 1.0
            for c in patient[t+1]:
                y[t, b, c] = 1.0
        mask[:lengths[b], b] = 1.0

    return x, y, mask, torch.tensor(lengths, device=device)
