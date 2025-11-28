import numpy as np
import pickle
from tqdm import tqdm

data = np.load("data/result/synthetic_mimic3.npz")
x = data['x']          # (1500, 34, 2869)
lens = data['lens']    # (1500,)

print(f"Loaded synthetic data: {x.shape}, lens: {lens.shape}")

# Tìm các code thực sự xuất hiện (loại bỏ all-zero visits nếu có)
all_codes = []
for i in range(x.shape[2]):
    if np.any(x[:, :, i] > 0.5):  # threshold 0.5
        all_codes.append(i)

print(f"Number of unique codes in synthetic data: {len(all_codes)}")
code_to_index = {code: idx for idx, code in enumerate(all_codes)}
index_to_code = {idx: code for code, idx in code_to_index.items()}
vocab_size = len(all_codes)
print(f"Final vocab size: {vocab_size}")

# Chuyển sang định dạng list of list (như MIMIC thật)
synthetic_records = []

for i in tqdm(range(len(x)), desc="Converting to sequences"):
    patient_visits = []
    actual_len = int(lens[i])
    
    for v in range(actual_len):
        visit_vector = x[i, v]  # (2869,)
        codes_in_visit = np.where(visit_vector > 0.5)[0]  # indices where == 1
        codes_idx = [code_to_index[c] for c in codes_in_visit if c in code_to_index]
        if len(codes_idx) > 0:
            patient_visits.append(sorted(codes_idx))  # sort để ổn định
    
    if len(patient_visits) >= 2:  # chỉ giữ patient có ít nhất 2 visits
        synthetic_records.append({
            'visits': patient_visits,           # list of list[int]
            'original_indices': all_codes,      # để reverse nếu cần
        })

print(f"Final number of usable patients (>=2 visits): {len(synthetic_records)}")

# Chia train/val (vì chỉ có 1500 → chia 90/10 hoặc dùng toàn bộ để pretrain)
train_size = int(0.9 * len(synthetic_records))
train_data = synthetic_records[:train_size]
val_data = synthetic_records[train_size:]

# Lưu lại
import os
os.makedirs("synthetic_processed", exist_ok=True)

pickle.dump({
    'train': train_data,
    'val': val_data,
    'code_to_index': code_to_index,
    'index_to_code': index_to_code,
    'vocab_size': vocab_size,
    'original_code_dim': x.shape[2]
}, open("synthetic_processed/processed.pkl", "wb"))

print("Done! Saved to synthetic_processed/processed.pkl")
