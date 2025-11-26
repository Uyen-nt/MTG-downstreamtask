# retain_micron/dataset.py
from torch.utils.data import Dataset
import torch

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
    labels_onehot = torch.zeros(batch_size, n_codes)

    for i, lbl in enumerate(labels):
        for code in lbl:
            if isinstance(code, (int, np.integer)) and 0 <= code < n_codes:
                labels_onehot[i, code] = 1.0
    return list(sequences), labels_onehot

# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     labels_onehot = []
#     for lbl in labels:
#         onehot = torch.zeros(2869)  # n_codes = 2869, indices từ 0-2868
        
#         # Lọc các index hợp lệ (0-2868)
#         valid_indices = [idx for idx in lbl if idx < 2869]
#         if valid_indices:
#             onehot[valid_indices] = 1.0
#         labels_onehot.append(onehot)
    
#     return list(sequences), torch.stack(labels_onehot)
