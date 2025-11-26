import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

from retain_micron.utils import load_and_preprocess_synthetic
from retain_micron.model import RETAIN_Diagnosis
from retain_micron.dataset import EHRDataset, collate_fn
from retain_micron.train import train_model

if __name__ == "__main__":
    seqs, labels, n_codes = load_and_preprocess_synthetic(
        data_path="data/result/synthetic_mimic3.npz"
    )

    # Split train/val (vì real_next thường chỉ có train)
    from sklearn.model_selection import train_test_split
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        seqs, labels, test_size=0.1, random_state=42, stratify=None
    )

    train_dataset = EHRDataset(train_seqs, train_labels)
    val_dataset   = EHRDataset(val_seqs, val_labels)

    collate = lambda batch: collate_fn(batch, n_codes=n_codes)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate)

    model = RETAIN_Diagnosis(n_codes=n_codes, emb_size=256, dropout=0.5)

    train_model(model, train_loader, val_loader, epochs=20, save_path="retain_micron/result")
