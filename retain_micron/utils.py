# retain_micron/utils.py
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_synthetic(data_path="data/result/synthetic_mimic3.npz", test_size=0.2):
    data = np.load(data_path)
    x = data['x'].astype(np.float32)  # (N, max_visits, n_codes)
    lens = data['lens']               # (N,)

    n_codes = x.shape[-1]
    n_patients = len(x)

    # Split by patient (rất quan trọng!)
    patient_ids = np.arange(n_patients)
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42, stratify=lens//2)

    def patients_to_samples(ids):
        sequences = []
        labels = []
        for pid in ids:
            L = int(lens[pid])
            if L < 2:  # cần ít nhất 2 visit để có "next"
                continue

            # History: tất cả visit từ 0 đến L-2
            history = []
            for j in range(L-1):  # chỉ lấy đến visit áp chót
                codes = np.where(x[pid, j] > 0)[0].tolist()
                if not codes:
                    codes = [n_codes]  # padding code (hoặc dùng 0 và để emb học)
                history.append(codes)

            # Label: visit cuối cùng (L-1)
            label = np.where(x[pid, L-1] > 0)[0].tolist()
            if not label:
                label = [n_codes]

            sequences.append(history)
            labels.append(label)

        return sequences, labels

    train_seqs, train_labels = patients_to_samples(train_ids)
    test_seqs, test_labels = patients_to_samples(test_ids)

    print(f"Synthetic FIXED - Train: {len(train_seqs)} patients → samples")
    print(f"Synthetic FIXED - Test: {len(test_seqs)} patients → samples")

    return (train_seqs, train_labels), (test_seqs, test_labels), n_codes
