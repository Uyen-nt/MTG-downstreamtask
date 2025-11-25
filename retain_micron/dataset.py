# retain_micron/dataset.py
from torch.utils.data import Dataset
import torch

class EHRDataset(Dataset):
    def __init__(self, sequences, labels, n_codes):
        self.sequences = sequences
        self.labels = labels
        self.n_codes = n_codes

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    labels_onehot = []
    for lbl in labels:
        onehot = torch.zeros(2869)
        if len(lbl) > 0:
            onehot[list(lbl)] = 1.0
        labels_onehot.append(onehot)
    return list(sequences), torch.stack(labels_onehot)
