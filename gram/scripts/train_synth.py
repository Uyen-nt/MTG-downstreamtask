# gram/scripts/train_synth.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from gram.model.gram import GRAM, load_tree, pad_batch


# ==========================
# PATH CONFIG
# ==========================

SEQ_FILE = "gram/data/synthetic_converted/synthetic_mapped.seqs"
TREE_PREFIX = "gram/data/mimic3_tree"      
MODEL_OUT = "gram/data/synth_train.pt"



def clean_seqs(seqs):
    clean = []
    for p in seqs:
        v = [x for x in p if len(x) > 0]
        if len(v) >= 2:
            clean.append(v)
    return clean


def build_labels(seqs):
    X, Y = [], []
    for p in seqs:
        X.append(p[:-1])
        Y.append(p[1:])
    return X, Y


# ==========================
# TRAINING FUNCTION
# ==========================

def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== TRAIN GRAM WITH SYNTHETIC DATA =====")

    # 1) LOAD SEQS
    seqs = pickle.load(open(SEQ_FILE, "rb"))
    seqs = clean_seqs(seqs)
    X, Y = build_labels(seqs)

    # 2) Compute num_codes
    num_codes = max(max(max(v) for v in patient) for patient in seqs) + 1
    num_classes = num_codes
    print("num_codes =", num_codes)

    # 3) LOAD TREE (REAL MIMIC3)
    tree_leaves, tree_ancestors = load_tree(TREE_PREFIX, num_codes, device=device)

    # Compute max index in tree for embedding table size
    all_idx = []
    for L, A in zip(tree_leaves, tree_ancestors):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())

    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))

    max_index_in_tree = max(all_idx)
    print("max_index_in_tree =", max_index_in_tree)

    # 4) CREATE MODEL — KHÔNG CÓ num_ancestors NỮA
    model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_levels=len(tree_leaves),
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_ancestors,
        device=device,
        max_index_in_tree=max_index_in_tree
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss(reduction="none")

    # 5) TRAIN LOOP
    for epoch in range(20):
        model.train()
        total_loss = 0

        for i in range(0, len(X), 32):
            x_batch = X[i : i+32]
            y_batch = Y[i : i+32]

            x, _, mask, lengths = pad_batch(x_batch, num_classes, num_codes, device)
            _, y, _, _          = pad_batch(y_batch, num_classes, num_codes, device)

            pred = model(x, mask)

            loss_raw = loss_fn(pred, y)
            loss = (loss_raw.sum(dim=2).sum(dim=0) / lengths).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print("Model saved to:", MODEL_OUT)



if __name__ == "__main__":
    train()
