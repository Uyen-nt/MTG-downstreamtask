import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


# ============================================================
# 1) LOAD TREE: đọc tree Theano và pad đúng chuẩn
# ============================================================
def load_tree(tree_prefix, device="cpu"):
    levels_leaves = []
    levels_anc = []
    max_K = 0

    # Load level 5 → 1 (như GRAM)
    for level in range(5, 0, -1):
        path = f"{tree_prefix}.level{level}.pk"
        try:
            tree = pickle.load(open(path, "rb"))
        except:
            continue

        if len(tree) == 0:
            print(f"[WARN] empty: {path}")
            continue

        leaves = np.array(list(tree.keys()))
        ancestors_list = list(tree.values())

        # convert each to array
        ancestors = [np.array(a, dtype=np.int64) for a in ancestors_list]
        max_K = max(max_K, max(len(a) for a in ancestors))

        levels_leaves.append(leaves)
        levels_anc.append(ancestors)

    # --- pad all levels to max_K ---
    leaves_out = []
    ancestors_out = []

    for leaves, anc_list in zip(levels_leaves, levels_anc):
        C = len(leaves)
        fixed_anc = np.zeros((C, max_K), dtype=np.int64)

        for i, a in enumerate(anc_list):
            L = len(a)
            fixed_anc[i, :L] = a
            fixed_anc[i, L:] = a[-1]        # repeat last ancestor

        leaves_exp = np.repeat(leaves[:, None], max_K, axis=1)

        leaves_out.append(torch.tensor(leaves_exp, device=device))
        ancestors_out.append(torch.tensor(fixed_anc, device=device))

    return leaves_out, ancestors_out



# ============================================================
# 2) GRAM MODEL — PyTorch IMPLEMENTATION
# ============================================================
class GRAM(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 num_ancestors,
                 emb_dim=128,
                 att_dim=128,
                 hidden_dim=128,
                 tree_leaves=None,
                 tree_ancestors=None,
                 max_index_in_tree=None,
                 device="cpu"):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        self.tree_leaves = tree_leaves
        self.tree_ancestors = tree_ancestors
        self.num_levels = len(tree_leaves)

        # --------------------------
        # W_emb: (max_index+1, D)
        # --------------------------
        if max_index_in_tree is None:
            max_index_in_tree = input_dim + num_ancestors

        self.W_emb = nn.Embedding(max_index_in_tree + 1, emb_dim)

        # attention MLP
        self.W_att = nn.Linear(emb_dim * 2, att_dim)
        self.v_att = nn.Linear(att_dim, 1, bias=False)

        # GRU (same dims)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=False
        )

        # output
        self.out = nn.Linear(hidden_dim, num_classes)

    # -------------------------------------------------------------
    # ATTENTION per tree level
    # -------------------------------------------------------------
    def attention_for_level(self, leaves, anc):
        # leaves: (C,K)
        # anc   : (C,K)
        leaf_emb = self.W_emb(leaves)     # (C,K,D)
        anc_emb = self.W_emb(anc)         # (C,K,D)

        att_input = torch.cat([leaf_emb, anc_emb], dim=-1)   # (C,K,2D)
        h = torch.tanh(self.W_att(att_input))                # (C,K,A)
        att = torch.softmax(self.v_att(h).squeeze(-1), dim=-1)   # (C,K)

        emb = (att.unsqueeze(-1) * anc_emb).sum(dim=1)       # (C, D)
        return emb

    # -------------------------------------------------------------
    # forward
    # -------------------------------------------------------------
    def forward(self, x, mask):
        T, B, C = x.shape

        # find active code indices
        active = (x.sum(dim=(0, 1)) > 0).nonzero(as_tuple=True)[0]
        if len(active) == 0:
            return torch.zeros(T, B, self.num_classes, device=self.device)

        # compute embeddings for ACTIVE codes only
        per_level_emb = []
        for leaves, anc in zip(self.tree_leaves, self.tree_ancestors):
            # chọn những leaf nằm trong active codes
            valid_idx = active[active < leaves.shape[0]]
            if len(valid_idx) == 0:
                per_level_emb.append(torch.zeros((0, self.emb_dim), device=self.device))
                continue
            emb = self.attention_for_level(leaves[valid_idx], anc[valid_idx])
            per_level_emb.append(emb)

        # concat all levels → (N, L*D)
        gram_emb = torch.cat(per_level_emb, dim=-1)

        # build full embedding table for input_dim codes
        full_emb = torch.zeros(self.input_dim, self.num_levels * self.emb_dim,
                               device=self.device)
        full_emb.index_copy_(0, active, gram_emb)

        # split into L matrices
        matrices = torch.split(full_emb, self.emb_dim, dim=-1)

        # x → embedding
        x_flat = x.view(-1, self.input_dim)
        visit_emb = sum([torch.matmul(x_flat, M) for M in matrices])
        visit_emb = torch.tanh(visit_emb).view(T, B, self.emb_dim)

        # GRU
        h, _ = self.gru(visit_emb)
        h = h * mask.unsqueeze(-1)

        logits = self.out(h)
        return torch.sigmoid(logits)
        
def pad_batch(seqs, num_classes, input_dim, device="cpu"):
    lengths = [len(p)-1 for p in seqs]
    T = max(lengths)
    B = len(seqs)

    x = torch.zeros(T, B, input_dim, device=device)
    y = torch.zeros(T, B, num_classes, device=device)
    mask = torch.zeros(T, B, device=device)

    for b, patient in enumerate(seqs):
        for t in range(len(patient)-1):
            for code in patient[t]:
                x[t, b, code] = 1
            for code in patient[t+1]:
                y[t, b, code] = 1
        mask[:lengths[b], b] = 1

    return x, y, mask, torch.tensor(lengths, device=device)

