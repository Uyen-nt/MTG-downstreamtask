# retain_micron/utils_mimic3.py
import numpy as np
import torch
import os

def load_and_preprocess(train_path="data/mimic3/standard/real_next/train.npz", test_path=None):
    """
    Dùng cho file real_next/train.npz (có x, lens, y)
    Đây là chuẩn next-visit prediction – đúng 100% với RETAIN gốc
    """
    print("Loading MIMIC-III real_next data...")
    data = np.load(train_path)
    x = data['x']          # (N, max_visits, n_codes) – binary/float32
    y = data['y']          # (N, n_codes) – next visit
    lens = data['lens']    # (N,)

    n_samples, max_visits, n_codes = x.shape
    print(f"MIMIC-III train: {n_samples} samples, {max_visits} max visits, {n_codes} codes")

    sequences = []
    labels = []

    for i in range(n_samples):
        L = int(lens[i])
        if L < 1:
            continue
        # History: visit 0 đến L-1
        history = []
        for j in range(L):
            codes = np.where(x[i, j] > 0.5)[0].tolist()   # hoặc == 1.0 nếu chắc chắn binary
            if not codes:
                codes = [n_codes]  # padding code
            history.append(codes)
        # Label: visit tiếp theo = y[i]
        label = np.where(y[i] > 0.5)[0].tolist()
        if not label:
            label = [n_codes]

        sequences.append(history)
        labels.append(label)

    print(f"Total training sequences: {len(sequences)}")
    
    if test_path is not None:
        # Nếu có test (một số pipeline có)
        test_data = np.load(test_path)
        tx = test_data['x']
        ty = test_data['y']
        tlens = test_data['lens']
        test_sequences = []
        test_labels = []
        for i in range(tx.shape[0]):
            L = int(tlens[i])
            if L < 1: continue
            hist = [np.where(tx[i,j]>0.5)[0].tolist() or [n_codes] for j in range(L)]
            lab = np.where(ty[i]>0.5)[0].tolist() or [n_codes]
            test_sequences.append(hist)
            test_labels.append(lab)
        return sequences, labels, test_sequences, test_labels, n_codes

    return sequences, labels, n_codes

    # Sửa utils_mimic3.py để handle realnext format đúng cách
# def load_and_preprocess(train_path, test_path=None):
#     """
#     Load RealNext data với format CHUẨN (multi-predictions per sample)
#     """
#     print("Loading MIMIC-III real_next data (corrected format)...")
#     train_data = np.load(train_path)
#     x_train = train_data['x']      # (patients, max_visits, n_codes)
#     y_train = train_data['y']      # (patients, max_visits, n_codes) - next visits
#     lens_train = train_data['lens'] # (patients,) - actual history lengths
    
#     n_patients, max_visits, n_codes = x_train.shape
#     print(f"MIMIC-III train: {n_patients} patients, {max_visits} max visits, {n_codes} codes")
#     print(f"Total next-visit predictions: {np.sum(lens_train)}")
    
#     sequences = []
#     labels = []
    
#     # Convert từ realnext format sang format RETAIN cần
#     for i in range(n_patients):
#         actual_len = int(lens_train[i])
#         if actual_len < 1:
#             continue
            
#         # Với mỗi patient, tạo multiple samples
#         for t in range(actual_len):
#             # History: visits 0..t
#             history = []
#             for j in range(t + 1):  # Bao gồm cả visit hiện tại
#                 codes = np.where(x_train[i, j] > 0.5)[0].tolist()
#                 codes = [c for c in codes if c < n_codes]
#                 if not codes:
#                     codes = [n_codes - 1]
#                 history.append(codes)
            
#             # Label: next visit (t+1)
#             if t < actual_len:  # Đảm bảo có next visit
#                 label = np.where(y_train[i, t] > 0.5)[0].tolist()
#                 label = [l for l in label if l < n_codes]
#                 if not label:
#                     label = [n_codes - 1]
#             else:
#                 label = [n_codes - 1]
                
#             sequences.append(history)
#             labels.append(label)
    
#     print(f"Converted to {len(sequences)} next-visit samples")
    
#     if test_path:
#         test_data = np.load(test_path)
#         tx, ty, tlens = test_data['x'], test_data['y'], test_data['lens']
#         test_sequences, test_labels = [], []
        
#         for i in range(len(tx)):
#             actual_len = int(tlens[i])
#             if actual_len < 1:
#                 continue
#             for t in range(actual_len):
#                 hist = [np.where(tx[i,j]>0.5)[0].tolist() or [n_codes-1] for j in range(t+1)]
#                 lab = np.where(ty[i,t]>0.5)[0].tolist() or [n_codes-1]
#                 test_sequences.append(hist)
#                 test_labels.append(lab)
                
#         return sequences, labels, test_sequences, test_labels, n_codes
    
#     return sequences, labels, n_codes
