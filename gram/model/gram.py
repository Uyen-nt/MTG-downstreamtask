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
        try:
            tree = pickle.load(open(path, "rb"))
        except:
            print(f"Warning: Cannot load {path}")
            continue

        # Tạo arrays với kích thước chính xác
        sample_ancestors = next(iter(tree.values()))
        K = len(sample_ancestors)  # Số ancestors
        
        leaves = np.zeros((num_codes, K), dtype=np.int32)
        ancestors = np.zeros((num_codes, K), dtype=np.int32)

        # Lấy root code từ tree
        root_code = sample_ancestors[-1]  # Root thường là phần tử cuối
        
        for leaf_id in range(num_codes):
            if leaf_id in tree:
                # Code có trong tree
                ancestors[leaf_id] = tree[leaf_id]
                leaves[leaf_id] = leaf_id
            else:
                # Code không có trong tree → dùng root
                ancestors[leaf_id] = [root_code] * K
                leaves[leaf_id] = leaf_id

        leaves_list.append(torch.tensor(leaves, device=device, dtype=torch.long))
        ancestors_list.append(torch.tensor(ancestors, device=device, dtype=torch.long))

    print(f"Loaded {len(leaves_list)} tree levels")
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
        self.max_index_in_tree = max_index_in_tree
    
        # Embedding table (codes + ancestors)
        self.W_emb = nn.Embedding(max_index_in_tree + 1, emb_dim)
        
        # BỎ layer reduce này - không cần thiết
        # self.W_reduce = nn.Linear(self.num_levels * self.emb_dim, self.emb_dim)
    
        # Attention network
        self.W_att = nn.Linear(emb_dim * 2, att_dim)
        self.v_att = nn.Linear(att_dim, 1, bias=False)
    
        # GRU phải nhận input_size = num_levels * emb_dim
        self.gru = nn.GRU(
            input_size=num_levels * emb_dim,  # Đảm bảo đúng kích thước
            hidden_size=hidden_dim,
            batch_first=False  # (T, B, features)
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
    
        # Tìm tất cả các mã active trong batch - sửa cách tìm
        active_codes = torch.unique(x.nonzero()[:, 2])  # Lấy tất cả unique codes xuất hiện
        
        if len(active_codes) == 0:
            return torch.zeros(Tt, B, self.num_classes, device=device)
    
        per_level_emb = []
    
        # ---------------------------------------------------------
        # Compute GRAM embedding for each level
        # ---------------------------------------------------------
        for leaves, ancestors in zip(self.tree_leaves, self.tree_anc):
            # CHỈ lấy các codes nằm trong phạm vi hợp lệ
            valid_idx = active_codes[active_codes < len(leaves)]
            
            if len(valid_idx) == 0:
                # Nếu không có code nào valid, tạo embedding zero
                dummy_emb = torch.zeros(len(active_codes), self.emb_dim, device=device)
                per_level_emb.append(dummy_emb)
                continue
    
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
        # CONCAT LEVEL EMBEDDINGS
        # ---------------------------------------------------------
        if len(per_level_emb) > 0 and len(per_level_emb[0]) > 0:
            gram_emb = torch.cat(per_level_emb, dim=-1)  # (N, num_levels * emb_dim)
        else:
            # Fallback: tạo embedding zero
            gram_emb = torch.zeros(len(active_codes), self.num_levels * self.emb_dim, device=device)
    
        # Tạo code embedding table
        code_emb = torch.zeros(self.input_dim, self.num_levels * self.emb_dim, device=device)
        
        # CHỈ copy các codes hợp lệ (nằm trong phạm vi input_dim)
        valid_codes = active_codes[active_codes < self.input_dim]
        if len(valid_codes) > 0:
            # Đảm bảo indices hợp lệ
            valid_indices = valid_codes[valid_codes < len(gram_emb)]
            if len(valid_indices) > 0:
                code_emb.index_copy_(0, valid_codes, gram_emb[valid_indices])
    
        # ---------------------------------------------------------
        # TÍNH VISIT EMBEDDING
        # ---------------------------------------------------------
        x_flat = x.view(-1, self.input_dim)  # (T*B, input_dim)
        visit_emb = torch.tanh(x_flat @ code_emb)  # (T*B, num_levels * emb_dim)
        visit_emb = visit_emb.view(Tt, B, self.num_levels * self.emb_dim)  # (T, B, num_levels * emb_dim)
    
        # GRU
        h, _ = self.gru(visit_emb)  # (T, B, hidden_dim)
        h = h * mask.unsqueeze(-1)
    
        # Output per time-step - DÙNG SOFTMAX CHO MULTI-CLASS
        y_hat = F.softmax(self.out(h), dim=-1)  # (T, B, num_classes)
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
