import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


# ===============================================================
# LOAD TREE (PAD TO SAME ANCESTOR LENGTH)
# ===============================================================
def load_tree(prefix, num_codes, device):
    leaves_list = []
    ancestors_list = []

    for level in range(5, 0, -1):
        path = f"{prefix}.level{level}.pk"
        tree = pickle.load(open(path, "rb"))

        # treeMap: {leaf_code: [anc1, anc2, ...]}
        # GRAM gốc yêu cầu đủ num_codes dòng
        leaves = np.zeros((num_codes, len(next(iter(tree.values())))), dtype=np.int32)
        ancestors = np.zeros((num_codes, len(next(iter(tree.values())))), dtype=np.int32)

        for leaf_id in range(num_codes):
            if leaf_id in tree:
                ancestors[leaf_id] = tree[leaf_id]
                leaves[leaf_id] = leaf_id
            else:
                # nếu code không có trong tree → dùng A_ROOT (theo GRAM gốc)
                root = list(tree.values())[0][-1]
                ancestors[leaf_id] = [root] * ancestors.shape[1]
                leaves[leaf_id] = leaf_id

        leaves_list.append(torch.tensor(leaves, device=device))
        ancestors_list.append(torch.tensor(ancestors, device=device))

    return leaves_list, ancestors_list



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
        self.device = device
        
        # Total dim = L * D
        self.total_emb_dim = num_levels * emb_dim

        # Embedding table (codes + ancestors)
        self.W_emb = nn.Embedding(max_index_in_tree + 1, emb_dim)

        # Attention network
        self.W_att = nn.Linear(emb_dim * 2, att_dim)
        self.v_att = nn.Linear(att_dim, 1, bias=False)

        # GRU must receive total_emb_dim
        self.gru = nn.GRU(
            input_size=self.total_emb_dim,
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

        # find all active codes in batch
        active_codes = (x.sum(dim=(0,1)) > 0).nonzero(as_tuple=True)[0]

        if len(active_codes) == 0:
            return torch.zeros(Tt, B, self.num_classes, device=device)

        per_level_emb = []

        # ---------------------------------------------------------
        # Compute GRAM embedding for each level
        # ---------------------------------------------------------
        for leaves, ancestors in zip(self.tree_leaves, self.tree_anc):

            valid_idx = active_codes

            leaves_b = leaves[valid_idx]      # (N, K)
            anc_b = ancestors[valid_idx]      # (N, K)

            leaf_emb = self.W_emb(leaves_b)   # (N, K, D)
            anc_emb  = self.W_emb(anc_b)      # (N, K, D)

            att_in = torch.cat([leaf_emb, anc_emb], dim=-1)
            att_h = torch.tanh(self.W_att(att_in))
            att_logits = self.v_att(att_h).squeeze(-1)     # (N,K)
            att = torch.softmax(att_logits, dim=-1)

            final = (att.unsqueeze(-1) * anc_emb).sum(dim=1) # (N,D)
            per_level_emb.append(final)

        # ---------------------------------------------------------
        # CONCAT LEVEL EMBEDDINGS THEO CHIỀU FEATURES
        # result: (N, total_emb_dim)
        # ---------------------------------------------------------
        gram_emb = torch.cat(per_level_emb, dim=-1)

        # Fill full code embedding table
        code_emb = torch.zeros(self.max_index_in_tree + 1,
                       gram_emb.shape[1],
                       device=device)
        code_emb.index_copy_(0, active_codes, gram_emb)

        final_emb = code_emb[:self.input_dim]
        
        x_flat = x.view(-1, self.input_dim)
        visit_emb = torch.tanh(x_flat @ final_emb)
        visit_emb = visit_emb.view(Tt, B, self.total_emb_dim)

        # GRU
        h, _ = self.gru(visit_emb)
        h = h * mask.unsqueeze(-1)

        # Output per time-step
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
