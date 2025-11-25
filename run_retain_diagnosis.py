# run_retain_diagnosis.py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from retain_micron.utils import load_and_preprocess_synthetic
from retain_micron.model import RETAIN_Diagnosis
from retain_micron.dataset import EHRDataset, collate_fn
from retain_micron.train import train_model

if __name__ == "__main__":
    # Load synthetic data với splitting đúng
    (train_seqs, train_labels), (test_seqs, test_labels), n_codes = load_and_preprocess_synthetic("data/result/synthetic_mimic3.npz")
    
    train_dataset = EHRDataset(train_seqs, train_labels)
    test_dataset = EHRDataset(test_seqs, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    model = RETAIN_Diagnosis(n_codes=n_codes, emb_size=256, dropout=0.5)
    
    print("Training RETAIN on SYNTHETIC data...")
    train_model(model, train_loader, test_loader, epochs=20)
