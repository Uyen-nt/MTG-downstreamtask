# retain_micron/utils.py
import numpy as np

def load_and_preprocess_synthetic(data_path="data/result/synthetic_mimic3.npz"):
    data = np.load(data_path)
    x = data['x']
    lens = data['lens']
    
    n_patients, max_visits, n_codes = x.shape
    print(f"Loaded: {n_patients} patients, {max_visits} max visits, {n_codes} codes")

    valid_seqs = []
    valid_labels = []

    for i in range(n_patients):
        L = lens[i]
        if L < 2:
            continue
        history = []
        for t in range(L - 1):
            codes = np.where(x[i, t] > 0.5)[0].tolist()
            if not codes:
                codes = [0]
            history.append(codes)
        label = np.where(x[i, L-1] > 0.5)[0]
        valid_seqs.append(history)
        valid_labels.append(label)

    print(f"Valid patients (≥2 visits): {len(valid_seqs)}")
    return valid_seqs, valid_labels, n_codes
