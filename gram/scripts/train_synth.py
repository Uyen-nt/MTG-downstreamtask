# gram/scripts/train_synth.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
from gram.model.gram import GRAM, load_tree, pad_batch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


SEQ_FILE = "gram/data/synthetic_tree.seqs"
TREE_PREFIX = "gram/data/synthetic_tree"
MODEL_OUT = "gram/data/synth_train.pt"


def build_labels(seqs):
    """Tạo next-visit labels đúng chuẩn GRAM."""
    labels = []
    X = []

    for p in seqs:
        if len(p) >= 2:
            X.append(p[:-1])
            labels.append(p[1:])
    return X, labels


def clean_seqs(seqs):
    clean = []
    for patient in seqs:
        visits = [v for v in patient if len(v) > 0]
        if len(visits) >= 2:
            clean.append(visits)
    return clean


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== TRAIN GRAM =====")

    # -------------------
    # Load cleaned seqs
    # -------------------
    seqs = pickle.load(open(SEQ_FILE, "rb"))
    seqs = clean_seqs(seqs)

    X, Y = build_labels(seqs)

    # -------------------
    # Compute num_codes
    # -------------------
    num_codes = max(max(max(v) for v in patient) for patient in seqs) + 1
    num_classes = num_codes  # same output size

    # -------------------
    # Load tree
    # -------------------
    tree_leaves, tree_ancestors = load_tree(TREE_PREFIX, device=device)
    num_ancestors = tree_ancestors[0].shape[1] - 1

    # -------------------
    # Create model
    # -------------------
    model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_ancestors=num_ancestors,
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_ancestors,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss(reduction="none")

    # -------------------
    # Training loop
    # -------------------
    for epoch in range(20):
        model.train()
        total_loss = 0

        for i in range(0, len(X), 32):
            batch_X = X[i:i+32]
            batch_Y = Y[i:i+32]

            x, y, mask, lengths = pad_batch(batch_X, num_classes, num_codes, device)
            y = pad_batch(batch_Y, num_classes, num_codes, device)[1]

            pred = model(x, mask)

            loss_raw = loss_fn(pred, y)       # (T,B,C)
            loss_masked = (loss_raw.sum(2).sum(0) / lengths).mean()

            optimizer.zero_grad()
            loss_masked.backward()
            optimizer.step()

            total_loss += loss_masked.item()

        print(f"[Epoch {epoch+1}] Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print("Model saved:", MODEL_OUT)


if __name__ == "__main__":
    train()
