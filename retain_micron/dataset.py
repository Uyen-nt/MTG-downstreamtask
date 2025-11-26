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
    
    print(f"\n=== COLLATE - BATCH SIZE: {batch_size} ===")
    
    # KIỂM TRA CẤU TRÚC CHẶT CHẼ
    for i, (seq, lbl) in enumerate(zip(sequences[:2], labels[:2])):
        print(f"Sample {i}:")
        print(f"  - History: {len(seq)} visits")
        
        # Kiểm tra từng visit trong history
        for j, visit in enumerate(seq[:2]):  # 2 visits đầu
            if isinstance(visit, list) and visit and isinstance(visit[0], list):
                print(f"  ❌ ERROR: Visit {j} is nested list: {visit[:1]}")
            else:
                print(f"  ✅ Visit {j}: {len(visit)} codes - {visit[:3]}...")
        
        # Kiểm tra label
        if isinstance(lbl, list) and lbl and isinstance(lbl[0], list):
            print(f"  ❌ ERROR: Label is nested list: {lbl[:1]}")
            # Tự động sửa: flatten label
            lbl = [item for sublist in lbl for item in sublist]
            labels = list(labels)  # Convert to list để sửa
            labels[i] = lbl
            print(f"  ✅ Fixed label: {lbl[:5]}...")
        else:
            print(f"  ✅ Label: {len(lbl)} codes - {lbl[:5]}...")
    
    labels_onehot = torch.zeros(batch_size, n_codes)
    valid_codes = 0
    
    for i, lbl in enumerate(labels):
        if not isinstance(lbl, list):
            print(f"ERROR: Label {i} is not a list: {type(lbl)}")
            continue
            
        for code in lbl:
            if isinstance(code, (int, np.integer)) and 0 <= code < n_codes:
                labels_onehot[i, code] = 1.0
                valid_codes += 1
            else:
                print(f"ERROR: Invalid code {code} (type: {type(code)})")
    
    print(f"Valid codes in batch: {valid_codes}")
    print(f"Labels onehot sum: {labels_onehot.sum().item()}")
    
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

