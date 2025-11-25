# retain_micron/utils.py
import numpy as np

def load_and_preprocess_synthetic(data_path="data/result/synthetic_mimic3.npz"):
    data = np.load(data_path)
    x = data['x']
    lens = data['lens']
    
    n_patients, max_visits, n_codes = x.shape
    print(f"Loaded: {n_patients} patients, {max_visits} max visits, {n_codes} codes")
    
    sequences = []
    labels = []
    
    for i in range(n_patients):
        L = lens[i]
        if L < 2: continue
        for t in range(1, L):  # dự đoán visit t từ 0..t-1
            history = []
            for j in range(t):
                codes = np.where(x[i, j] > 0.5)[0].tolist()
                if not codes:
                    codes = [n_codes]  # padding code
                history.append(codes)
            label = np.where(x[i, t] > 0.5)[0]
            sequences.append(history)
            labels.append(label.tolist())
    
    print(f"Total sequences for next-visit prediction: {len(sequences)}")
    return sequences, labels, n_codes
