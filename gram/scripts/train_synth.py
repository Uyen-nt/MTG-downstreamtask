# gram/scripts/train_synth.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
from gram.model.gram import GRAM
from gram.model.dataset import VisitDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


SEQ_FILE = "gram/data/synthetic_tree.seqs"
TREE_PREFIX = "gram/data/synthetic_tree"
MODEL_OUT = "gram/data/synth_train.pt"


def build_labels_from_seqs(seqs):
    """Tạo next-visit labels y từ seqs."""
    labels = []
    for patient in seqs:
        if len(patient) < 2:
            continue
        labels.append(patient)   # Sử dụng negative sampling implicit
    return labels


def load_tree_levels(prefix):
    tree = {}
    for i in range(1, 6):
        level_path = f"{prefix}.level{i}.pk"
        with open(level_path, "rb") as f:
            tree[f"level{i}"] = pickle.load(f)
    return tree


def train():
    print("===== TRAIN GRAM =====")

    seqs = pickle.load(open(SEQ_FILE, "rb"))
    labels = build_labels_from_seqs(seqs)
    tree = load_tree_levels(TREE_PREFIX)

    # Thông số cần thiết từ tree
    num_codes = max(max(max(v) for v in patient) for patient in seqs) + 1
    num_classes = num_codes  # multi-label prediction

    dataset = VisitDataset(seqs, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.pad)

    model = GRAM(
        num_codes=num_codes,
        num_classes=num_classes,
        emb_dim=128,
        hidden_dim=128,
        attention_dim=128,
        tree=tree
    )

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print("Training...")
    for epoch in range(20):
        total_loss = 0

        for x, y, mask in loader:
            x = x.cuda()
            y = y.cuda()
            mask = mask.cuda()

            pred = model(x, mask)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print("Model saved:", MODEL_OUT)


if __name__ == "__main__":
    train()
