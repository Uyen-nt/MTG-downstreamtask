import numpy as np
from sklearn.model_selection import train_test_split

def load_synthetic_npz(path):
    data = np.load(path)
    x = data['x']      # shape (patients, max_visits, n_codes)
    lens = data['lens']
    n_codes = x.shape[-1]

    sequences = []
    labels = []

    for pid in range(len(lens)):
        L = lens[pid]
        if L < 2:
            continue

        visits = []
        for j in range(L):
            codes = np.where(x[pid, j] == 1)[0].tolist()
            visits.append(codes)

        sequences.append(visits[:-1])
        labels.append(visits[-1])

    return sequences, labels, n_codes


def split_patients(seqs, labels, test_size=0.2):
    idx = np.arange(len(seqs))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=42)
    
    train_seqs = [seqs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_seqs = [seqs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_seqs, train_labels, test_seqs, test_labels
