# retain_micron/utils_mimic3.py
import numpy as np
import torch
import os

def load_and_preprocess_mimic3_next(train_path="data/mimic3/standard/real_next/train.npz",
                                   test_path=None):
    # """
    # Dùng cho file real_next/train.npz (có x, lens, y)
    # Đây là chuẩn next-visit prediction – đúng 100% với RETAIN gốc
    # """
    # print("Loading MIMIC-III real_next data...")
    # data = np.load(train_path)
    # x = data['x']          # (N, max_visits, n_codes) – binary/float32
    # y = data['y']          # (N, n_codes) – next visit
    # lens = data['lens']    # (N,)

    # n_samples, max_visits, n_codes = x.shape
    # print(f"MIMIC-III train: {n_samples} samples, {max_visits} max visits, {n_codes} codes")

    # sequences = []
    # labels = []

    # for i in range(n_samples):
    #     L = int(lens[i])
    #     if L < 1:
    #         continue
    #     # History: visit 0 đến L-1
    #     history = []
    #     for j in range(L):
    #         codes = np.where(x[i, j] > 0.5)[0].tolist()   # hoặc == 1.0 nếu chắc chắn binary
    #         if not codes:
    #             codes = [n_codes]  # padding code
    #         history.append(codes)
    #     # Label: visit tiếp theo = y[i]
    #     label = np.where(y[i] > 0.5)[0].tolist()
    #     if not label:
    #         label = [n_codes]

    #     sequences.append(history)
    #     labels.append(label)

    # print(f"Total training sequences: {len(sequences)}")
    
    # if test_path is not None:
    #     # Nếu có test (một số pipeline có)
    #     test_data = np.load(test_path)
    #     tx = test_data['x']
    #     ty = test_data['y']
    #     tlens = test_data['lens']
    #     test_sequences = []
    #     test_labels = []
    #     for i in range(tx.shape[0]):
    #         L = int(tlens[i])
    #         if L < 1: continue
    #         hist = [np.where(tx[i,j]>0.5)[0].tolist() or [n_codes] for j in range(L)]
    #         lab = np.where(ty[i]>0.5)[0].tolist() or [n_codes]
    #         test_sequences.append(hist)
    #         test_labels.append(lab)
    #     return sequences, labels, test_sequences, test_labels, n_codes

    # return sequences, labels, n_codes

    
    # Load train data
    train_data = np.load(train_path)
    x_train, y_train, lens_train = train_data['x'], train_data['y'], train_data['lens']
    
    sequences = []
    labels = []
    
    for i in range(len(x_train)):
        L = int(lens_train[i])  # Số visits trong history
        if L < 1:
            continue
            
        # History: x_train[i, 0:L] 
        history = []
        for j in range(L):
            codes = np.where(x_train[i, j] > 0.5)[0].tolist()
            codes = [c for c in codes if c < 2869]  # Filter valid codes
            history.append(codes)
        
        # Label: y_train[i] (next visit)
        label = np.where(y_train[i] > 0.5)[0].tolist()
        label = [l for l in label if l < 2869]
        
        sequences.append(history)
        labels.append(label)
    
    print(f"Loaded {len(sequences)} next-visit samples from MIMIC-III")
    return sequences, labels, 2869
