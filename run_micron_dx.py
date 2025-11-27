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
    x_raw, sequences, labels, n_codes = load_synthetic_npz("data/result/synthetic_mimic3.npz")

    print("Building DCM from raw x...")
    dcm = build_dcm(x_raw)

    print("Splitting patients...")
    train_x, train_l, val_x, val_l = split_patients(sequences, labels)

    train_ds = MicronDataset(train_x, train_l, n_codes)
    val_ds = MicronDataset(val_x, val_l, n_codes)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=micron_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=micron_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MICRON_DX(vocab_size=n_codes, dcm=dcm, emb_dim=256, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_f1 = 0
    os.makedirs("micron/result", exist_ok=True)

    print(f"🔥 Initial DCM FORCE = {model.dcm_force}")

    for epoch in range(20):
        model.train()
        total_loss = 0

        for visits, label in train_loader:
            visits = visits[0]
            label = label[0].to(device)

            optimizer.zero_grad()
            logits = model(visits).squeeze(0)

            # 🟢 ANTI-COLLAPSE WARMUP
            if epoch < 5:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
            else:
                loss = micron_loss(logits, label, model.dcm, model.dcm_force)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ============================
        # Debug output và top-10 codes
        # ============================
        with torch.no_grad():
            example_logits = model(val_x[0]).squeeze(0)
            example_probs = torch.sigmoid(example_logits)
            topk = torch.topk(example_probs, k=10)

            print(f"\nEpoch {epoch} — DEBUG")
            print("Max prob =", example_probs.max().item())
            print("Min prob =", example_probs.min().item())
            print("Mean prob=", example_probs.mean().item())
            print("Top-10 probs:", topk.values.cpu().tolist())
            print("Top-10 idx:", topk.indices.cpu().tolist())

        metrics = evaluate_all_metrics(model, val_loader, n_codes)
        print_metrics(metrics)
        avg_pred = metrics["avg_predicted_codes"]
        print(f"👉 avg_predicted_codes = {avg_pred:.4f}")

        # =============================
        # 🔥 BƯỚC 4 — ANTI-COLLAPSE HERE
        # =============================
        if epoch >= 5:
            if avg_pred < 1.0:
                print("⚠️ Model is collapsing → reducing DCM FORCE")
                model.dcm_force *= 0.5
                print(f"   New dcm_force = {model.dcm_force}\n")

        # =============================

        f1 = metrics["f1_micro"]
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "micron/result/micron_best.pth")
            print(f"🔥 NEW BEST MICRO-F1 = {best_f1:.4f}")

    print("Training finished!")
    print(f"Best F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
