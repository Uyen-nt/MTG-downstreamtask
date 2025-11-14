# gram/scripts/train_synth.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from gram.model.gram import GRAM, load_tree, pad_batch


SEQ_FILE = "gram/data/synthetic.seqs"        
TREE_PREFIX = "gram/data/mimic3_tree"        
MODEL_OUT = "gram/data/synth_train.pt"


def clean_seqs(seqs):
    clean = []
    for patient in seqs:
        visits = [v for v in patient if len(v) > 0]
        if len(visits) >= 2:
            clean.append(visits)
    return clean


def build_labels(seqs):
    X, Y = [], []
    for p in seqs:
        if len(p) >= 2:
            X.append(p[:-1])
            Y.append(p[1:])
    return X, Y


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== TRAIN GRAM WITH SYNTHETIC DATA =====")

    # -------------------
    # Load synthetic seqs
    # -------------------
    seqs = pickle.load(open(SEQ_FILE, "rb"))
    seqs = clean_seqs(seqs)
    X, Y = build_labels(seqs)

    # -------------------
    # Compute num_codes
    # -------------------
    num_codes = max(max(max(v) for v in patient) for patient in seqs) + 1
    num_classes = num_codes

    print("num_codes =", num_codes)

    # -------------------
    # Load REAL tree
    # -------------------
        # -------------------
    # Load REAL tree
    # -------------------
    tree_leaves, tree_ancestors = load_tree(TREE_PREFIX, device=device)

    if len(tree_ancestors) == 0:
        raise ValueError("No tree levels loaded! Check your tree files.")

    max_indices = []
    for anc in tree_ancestors:
        if anc.numel() > 0:
            max_indices.append(anc.max().item())
        else:
            print(f"Warning: Skipping empty ancestor tensor (level with 0 nodes)")

    if max_indices:
        max_index_in_tree = max(max_indices)
    else:
        max_index_in_tree = 0  # fallback an toàn
    print("max_index_in_tree =", max_index_in_tree)
    # -------------------
    # Create GRAM model
    # -------------------
    model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_ancestors=tree_ancestors[0].shape[1] - 1,
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

    # -------------------
    # Training
    # -------------------
    for epoch in range(20):
        model.train()
        total_loss = 0

        for i in range(0, len(X), 32):
            batch_X = X[i:i+32]
            batch_Y = Y[i:i+32]

            x, _, mask, lengths = pad_batch(batch_X, num_classes, num_codes, device)
            _, y, _, _ = pad_batch(batch_Y, num_classes, num_codes, device)

            pred = model(x, mask)

            loss_raw = loss_fn(pred, y)
            loss = (loss_raw.sum(2).sum(0) / lengths).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print("Model saved to:", MODEL_OUT)


if __name__ == "__main__":
    train()
