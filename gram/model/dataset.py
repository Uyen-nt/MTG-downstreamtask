# gram/model/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np


class VisitDataset(Dataset):
    """
    Dataset cho GRAM PyTorch
    seqs  = list[list[list[int]]]  (visit sequences)
    labels = list[list[list[int]]] (next-visit labels)
    """
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

    @staticmethod
    def pad(batch):
        """
        Pad batch cho seqs & labels → tensor shape:
            x: (B, T, C)
            y: (B, T, C)
            mask: (B, T)
        """

        seqs, labels = zip(*batch)

        max_len = max(len(s) for s in seqs)
        num_codes = max(max(max(v) if v else 0 for v in patient) for patient in seqs) + 1

        B = len(seqs)
        T = max_len

        x = torch.zeros(B, T, num_codes)
        y = torch.zeros(B, T, num_codes)
        mask = torch.zeros(B, T)

        for i, (seq, lab) in enumerate(zip(seqs, labels)):
            L = len(seq)
            mask[i, :L] = 1

            for t in range(L):
                for code in seq[t]:
                    x[i, t, code] = 1.0

            for t in range(len(lab)):
                for code in lab[t]:
                    y[i, t, code] = 1.0

        return x, y, mask
