import numpy as np
import torch
from torch.utils.data import DataLoader

from micron.model import MICRON_DX
from micron.dataset import MicronDataset, micron_collate
from micron.utils import load_synthetic_npz, split_patients
from micron.build_dcm import build_dcm
from micron.loss import micron_loss
from micron.evaluate import evaluate_all_metrics, print_metrics

import os

def main():

    print("Loading synthetic data...")
    x, labels, n_codes = load_synthetic_npz("data/result/synthetic_mimic3.npz")

    print("Building DCM...")
    dcm = build_dcm(x)

    train_x, train_l, val_x, val_l = split_patients(x, labels)

    train_ds = MicronDataset(train_x, train_l, n_codes)
    val_ds = MicronDataset(val_x, val_l, n_codes)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=micron_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=micron_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MICRON_DX(vocab_size=n_codes, dcm=dcm, emb_dim=256, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_f1 = 0
    os.makedirs("micron/result", exist_ok=True)

    for epoch in range(30):
        model.train()
        total_loss = 0

        for visits, label in train_loader:
            visits = visits[0]
            label = label[0].to(device)

            optimizer.zero_grad()
            logits = model(visits).squeeze(0)

            loss = micron_loss(logits, label, model.dcm)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        metrics = evaluate_all_metrics(model, val_loader, n_codes)
        print_metrics(metrics)

        f1 = metrics["f1_micro"]

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "micron/result/micron_best.pth")
            print(f"🔥 NEW BEST MICRO-F1 = {best_f1:.4f}")

    print("Training finished!")

if __name__ == "__main__":
    main()
