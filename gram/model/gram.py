import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


# ----------------------------------------------------------------
# Load ancestor tree (same logic as GRAM build_tree)
# Returns: leaves_list, ancestors_list for levels 5→1
# ----------------------------------------------------------------
def load_tree(tree_prefix, device="cpu"):
    raw_leaves = []
    raw_ancestors = []
    max_K = 0

    # ---- Load all levels ----
    for level in range(5, 0, -1):
        path = f"{tree_prefix}.level{level}.pk"
        try:
            tree = pickle.load(open(path, "rb"))
            if len(tree) == 0:
                print(f"[WARN] Empty level {level}, skip.")
                continue

            ancestors = np.array(list(tree.values()))  # (C, K)
            leaves = np.array(list(tree.keys()))       # (C,)

            raw_leaves.append(leaves)
            raw_ancestors.append(ancestors)

            max_K = max(max_K, ancestors.shape[1])

        except Exception as e:
            print(f"[ERR] cannot load {path}: {e}")
            continue


    # ---- Pad levels to max_K ----
    fixed_leaves = []
    fixed_ancestors = []

    for leaves, ancestors in zip(raw_leaves, raw_ancestors):
        C, K = ancestors.shape

        # Pad ancestors (repeat last ancestor ID)
        if K < max_K:
            pad_width = max_K - K
            last_col = ancestors[:, -1:]
            pad_block = np.repeat(last_col, pad_width, axis=1)
            ancestors = np.concatenate([ancestors, pad_block], axis=1)

        # Expand leaves to shape (C, max_K)
        leaves_expanded = np.repeat(leaves[:, None], max_K, axis=1)

        fixed_leaves.append(torch.tensor(leaves_expanded, dtype=torch.long, device=device))
        fixed_ancestors.append(torch.tensor(ancestors, dtype=torch.long, device=device))

    return fixed_leaves, fixed_ancestors


# ----------------------------------------------------------------
# GRAM Model in PyTorch
# ----------------------------------------------------------------
class GRAM(nn.Module):
    def __init__(
        self,
        input_dim,        # number of codes
        num_classes,      # number of prediction classes
        num_ancestors,    # inputDimSize + numAncestors, same as GRAM
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=None,
        tree_ancestors=None,
        device="cpu",
        max_index_in_tree=None 
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_ancestors = num_ancestors
        self.num_levels = len(tree_leaves)
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # ----------------------------------------------------------------
        # Embedding matrix W_emb (same dimension as Theano)
        # Shape: (input_dim + numAncestors) x emb_dim
        # ----------------------------------------------------------------
        #self.W_emb = nn.Embedding(input_dim + num_ancestors, emb_dim) # dùng khi train với data mimic3
        if max_index_in_tree is not None:
            num_embeddings = max_index_in_tree + 1
        else:
            # fallback: trường hợp train trực tiếp trên mimc3, cây & input_dim khớp
            num_embeddings = input_dim + num_ancestors

        self.W_emb = nn.Embedding(num_embeddings, emb_dim)

        # Attention MLP parameters (same as GRAM)
        self.W_attention = nn.Linear(emb_dim * 2, att_dim)
        self.v_attention = nn.Linear(att_dim, 1, bias=False)

        # GRU layer
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=False,
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

        # Save tree tensors
        self.tree_leaves = tree_leaves
        self.tree_ancestors = tree_ancestors
   
    def forward(self, x, mask):
        T, B, _ = x.shape
        device = x.device
    
        # Lấy chỉ các mã xuất hiện trong batch
        active_codes = (x.sum(dim=(0,1)) > 0).nonzero(as_tuple=True)[0]  # (N,)
        if len(active_codes) == 0:
            return torch.zeros(T, B, self.num_classes, device=device)
            
        active_codes = active_codes.to(device)
        # Tính GRAM embedding CHỈ cho các mã active
        gram_embeddings = []
        for leaves, ancestors in zip(self.tree_leaves, self.tree_ancestors):
            valid_idx = active_codes[active_codes < leaves.shape[0]]

            if len(valid_idx) == 0:
                gram_embeddings.append(
                    torch.zeros((0, self.emb_dim), device=device)
                )
                continue
        
            leaves_batch = leaves[valid_idx]
            anc_batch = ancestors[valid_idx]
    
            leaf_emb = self.W_emb(leaves_batch)      # (N, K, D)
            anc_emb = self.W_emb(anc_batch)          # (N, K, D)
    
            att_input = torch.cat([leaf_emb, anc_emb], dim=-1)  # (N, K, 2D)
            mlp_out = torch.tanh(self.W_attention(att_input))   # (N, K, A)
            att_logits = self.v_attention(mlp_out).squeeze(-1)  # (N, K)
            att_weight = torch.softmax(att_logits, dim=-1)      # (N, K)
    
            weighted_emb = (att_weight.unsqueeze(-1) * anc_emb).sum(dim=1)  # (N, D)
            gram_embeddings.append(weighted_emb)
    
        # Concat theo level → (N, 5*D)
        gram_emb_batch = torch.cat(gram_embeddings, dim=-1)  # (N, 5*D)
    
        # Ánh xạ lại: active_codes → gram_emb_batch
        code_to_emb = torch.zeros(self.input_dim, 5 * self.emb_dim, device=device)
        code_to_emb = code_to_emb.index_copy_(0, active_codes, gram_emb_batch)
    
        # Tách thành 5 embedding riêng (mỗi level một ma trận)
        emb_matrices = torch.split(code_to_emb, self.emb_dim, dim=-1)  # 5 x (input_dim, D)
        # Thay vì split cố định 5 phần
        emb_per_level = gram_emb_batch.shape[1] // self.num_levels
        emb_matrices = torch.split(gram_emb_batch, emb_per_level, dim=-1)
        final_emb = torch.cat(emb_matrices, dim=0)  # (num_levels * input_dim, D)
        #final_emb = torch.cat(emb_matrices, dim=0)  # (5*input_dim, D)
    
        # Ánh xạ input → embedding
        x_flat = x.view(-1, self.input_dim)  # (T*B, input_dim)
        visit_emb = torch.tanh(torch.matmul(x_flat, final_emb[:self.input_dim]))  # (T*B, D)
        visit_emb = visit_emb.view(T, B, self.emb_dim)
    
        # GRU
        h, _ = self.gru(visit_emb)  # (T, B, hidden_dim)
        h = h * mask.unsqueeze(-1)
    
        # Output
        logits = self.output_layer(h)
        y_hat = torch.sigmoid(logits)
        return y_hat


# ----------------------------------------------------------------
# Padding function (same as padMatrix in GRAM)
# seqs: list of list of list (patients -> visits -> codes)
# ----------------------------------------------------------------
def pad_batch(seqs, num_classes, input_dim, device="cpu"):

    lengths = [len(p) - 1 for p in seqs]
    T = max(lengths)
    B = len(seqs)

    x = torch.zeros(T, B, input_dim, device=device)
    y = torch.zeros(T, B, num_classes, device=device)
    mask = torch.zeros(T, B, device=device)

    for b, patient in enumerate(seqs):
        for t in range(len(patient)-1):
            for code in patient[t]:
                x[t, b, code] = 1.0
            for code in patient[t+1]:
                y[t, b, code] = 1.0
        mask[: lengths[b], b] = 1.0

    return x, y, mask, torch.tensor(lengths, device=device)



# ----------------------------------------------------------------
# Training GRAM (pretrain or finetune)
# ----------------------------------------------------------------
def train_gram(
        model,
        train_seqs,
        valid_seqs,
        input_dim,
        num_classes,
        batch_size=32,
        lr=0.001,
        epochs=20,
        device="cpu"
    ):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="none")

    def run_epoch(data, train=True):
        total_loss = 0
        steps = 0

        if train:
            model.train()
        else:
            model.eval()

        for i in range(0, len(data), batch_size):
            batch = data[i : i+batch_size]
            x, y, mask, lengths = pad_batch(batch, num_classes, input_dim, device)

            y_hat = model(x, mask)
            loss = loss_fn(y_hat, y)
            loss = (loss.sum(dim=2).sum(dim=0) / lengths).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / steps

    for ep in range(epochs):
        tr = run_epoch(train_seqs, train=True)
        va = run_epoch(valid_seqs, train=False)
        print(f"[Epoch {ep}] Train loss={tr:.4f} | Valid loss={va:.4f}")

    return model
