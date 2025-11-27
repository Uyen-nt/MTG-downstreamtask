import torch
from torch.utils.data import DataLoader

from micron.model import MICRON_DX
from micron.dataset import MicronDataset, micron_collate
from micron.utils import load_synthetic_npz, split_patients
from micron.evaluate import eval_recall_k

import os

def main():

    print("Loading data...")
    seqs, labels, n_codes = load_synthetic_npz("data/result/synthetic_mimic3.npz")
    train_seqs, train_labels, val_seqs, val_labels = split_patients(seqs, labels)

    print(f"Training samples: {len(train_seqs)}")
    print(f"Val samples: {len(val_seqs)}")

    train_dataset = MicronDataset(train_seqs, train_labels, n_codes)
    val_dataset = MicronDataset(val_seqs, val_labels, n_codes)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=micron_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=micron_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MICRON_DX(vocab_size=n_codes, emb_dim=256, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    os.makedirs("micron/result", exist_ok=True)

    best_recall = 0

    for epoch in range(10):
        model.train()
        total_loss = 0

        for visits, labels in train_loader:
            visits = visits[0]
            labels = labels[0].to(device)

            optimizer.zero_grad()
            logits = model(visits).squeeze(0)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss = {total_loss/len(train_loader):.4f}")

        recall = eval_recall_k(model, val_loader, k=30)

        if recall > best_recall:
            print("🔥 New best model!")
            best_recall = recall
            torch.save(model.state_dict(), "micron/result/micron_dx_best.pth")

    print("Training completed.")
    print(f"Best Recall@30 = {best_recall:.4f}")

if __name__ == "__main__":
    main()
