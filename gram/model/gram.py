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
    leaves_list = []
    ancestors_list = []

    for i in range(5, 0, -1):
        tree = pickle.load(open(f"{tree_prefix}.level{i}.pk", "rb"))
        # tree is dict: key=leaf_id, value=[leaf, root, cat1, cat2 ...]
        ancestors = np.array(list(tree.values())).astype("int64")
        leaves = np.array([[k] * ancestors.shape[1] for k in tree.keys()]).astype("int64")

        leaves_list.append(torch.tensor(leaves, dtype=torch.long, device=device))
        ancestors_list.append(torch.tensor(ancestors, dtype=torch.long, device=device))

    return leaves_list, ancestors_list


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
        device="cpu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_ancestors = num_ancestors
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # ----------------------------------------------------------------
        # Embedding matrix W_emb (same dimension as Theano)
        # Shape: (input_dim + numAncestors) x emb_dim
        # ----------------------------------------------------------------
        # self.W_emb = nn.Embedding(input_dim + num_ancestors, emb_dim) # dùng khi train với data mimic3
        self.W_emb = nn.Embedding(max_index_in_tree + 1, emb_dim)

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


    # ----------------------------------------------------------------
    # Compute GRAM embedding for all codes (matrix of shape input_dim x emb_dim)
    # ----------------------------------------------------------------
    def build_gram_embedding(self):
        emb_list = []
        for leaves, ancestors in zip(self.tree_leaves, self.tree_ancestors):
            # W_emb[leaves]: (num_codes, num_ancestors, emb_dim)
            # W_emb[ancestors]: same shape

            leaf_emb = self.W_emb(leaves)          # (C, K, D)
            anc_emb  = self.W_emb(ancestors)       # (C, K, D)

            att_input = torch.cat([leaf_emb, anc_emb], dim=-1)  # (C, K, 2D)
            mlp_out = torch.tanh(self.W_attention(att_input))    # (C, K, A)
            att_logits = self.v_attention(mlp_out).squeeze(-1)   # (C, K)

            att_weight = torch.softmax(att_logits, dim=-1)       # (C, K)

            # Weighted sum over ancestors: Σ att * ancestor_embedding
            code_emb = torch.sum(att_weight.unsqueeze(-1) * anc_emb, dim=1)  # (C, D)
            emb_list.append(code_emb)

        # Concatenate embeddings of all levels (same as Theano concatenate)
        final_embedding = torch.cat(emb_list, dim=0)  # shape: (5*C, D)
        return final_embedding


    # ----------------------------------------------------------------
    # Forward
    # Inputs:
    #   x: (T, B, input_dim) multi-hot
    #   mask: (T, B)
    # ----------------------------------------------------------------
    def forward(self, x, mask):

        # Build GRAM embedding
        gram_emb = self.build_gram_embedding()  # (total_codes, emb_dim)

        # Encode visits: x @ gram_emb  (weighted sum of embeddings)
        # x shape: (T, B, input_dim)
        # → result: (T, B, emb_dim)
        x_emb = torch.tanh(torch.matmul(x, gram_emb[:self.input_dim, :]))

        # GRU
        h, _ = self.gru(x_emb)  # (T, B, hidden_dim)

        # Apply mask
        h = h * mask.unsqueeze(-1)

        # Predict next-visit codes
        logits = self.output_layer(h)    # (T, B, num_classes)
        y_hat = torch.sigmoid(logits)    # multi-label sigmoid

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
