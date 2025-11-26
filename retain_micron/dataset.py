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
    
    print(f"DEBUG - Real batch size: {batch_size}")
    print(f"DEBUG - First sequence: {len(sequences[0])} visits")
    print(f"DEBUG - First label: {labels[0][:5]}...")
    
    labels_onehot = torch.zeros(batch_size, n_codes)

    for i, lbl in enumerate(labels):
        for code in lbl:
            if isinstance(code, (int, np.integer)) and 0 <= code < n_codes:
                labels_onehot[i, code] = 1.0
    
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

