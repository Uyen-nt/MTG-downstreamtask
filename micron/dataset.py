import torch
from torch.utils.data import Dataset

class MicronDataset(Dataset):
    def __init__(self, sequences, labels, n_codes):
        self.sequences = sequences
        self.labels = labels
        self.n_codes = n_codes

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        visits = self.sequences[idx]
        label_codes = self.labels[idx]

        multi_hot = torch.zeros(self.n_codes)
        for c in label_codes:
            if c < self.n_codes:
                multi_hot[c] = 1.0

        return visits, multi_hot


def micron_collate(batch):
    visits, labels = zip(*batch)
    return list(visits), torch.stack(labels)
