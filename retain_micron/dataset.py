# retain_micron/dataset.py
from torch.utils.data import Dataset
import torch
import numpy as np

class EHRDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        #self.n_codes = n_codes

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch, n_codes=2869):
    sequences, labels = zip(*batch)
    batch_size = len(labels)
    
    print(f"\n=== COLLATE DEBUG ===")
    print(f"Real batch size: {batch_size}")
    print(f"First sequence has {len(sequences[0])} visits")
    
    # KIỂM TRA CẤU TRÚC CHI TIẾT
    for i, (seq, lbl) in enumerate(zip(sequences[:2], labels[:2])):  # 2 samples đầu
        print(f"Sample {i}:")
        print(f"  - {len(seq)} visits in history")
        for j, visit in enumerate(seq[:3]):  # 3 visits đầu
            print(f"  - Visit {j}: {len(visit)} codes - {visit[:3]}...")
        print(f"  - Label: {len(lbl)} codes - {lbl[:5]}...")
        print(f"  - Label type check: {[type(x) for x in lbl[:3]]}")
    
    labels_onehot = torch.zeros(batch_size, n_codes)

    for i, lbl in enumerate(labels):
        for code in lbl:
            if isinstance(code, (int, np.integer)) and 0 <= code < n_codes:
                labels_onehot[i, code] = 1.0
            else:
                print(f"ERROR: Invalid code {code} (type: {type(code)})")
                
    print(f"Labels onehot sum: {labels_onehot.sum().item()}")
    print("====================\n")
    return list(sequences), labels_onehot

# def collate_fn(batch, n_codes=2869):
#     sequences, labels = zip(*batch)
#     batch_size = len(labels)
#     labels_onehot = torch.zeros(batch_size, n_codes)

#     for i, lbl in enumerate(labels):
#         for code in lbl:
#             if isinstance(code, (int, np.integer)) and 0 <= code < n_codes:
#                 labels_onehot[i, code] = 1.0
#     return list(sequences), labels_onehot

