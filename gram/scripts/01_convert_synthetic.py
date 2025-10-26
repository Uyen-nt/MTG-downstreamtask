# scripts/01_convert_synthetic.py
# ĐÃ SỬA: Hỗ trợ multi-label per visit (nhiều mã = 1)

import numpy as np
import pickle
import os

SYNTH_NPZ = "data/result/mimic3/synthetic_mimic3.npz"
DATA_DIR = "gram/data"

os.makedirs(DATA_DIR, exist_ok=True)

print("Loading synthetic data...")
data = np.load(SYNTH_NPZ, allow_pickle=True)
x = data['x']        # (6000, 34, 2869) → float hoặc int
lens = data['lens']  # (6000,)

print(f"Loaded 'x': {x.shape}, 'lens': {lens.shape}")

# Tạo seqs
seqs = []
code_to_id = {}
next_id = 0

print("Converting multi-hot to code lists...")
total_codes_in_visits = 0
multi_label_count = 0

for i in range(len(lens)):
    real_len = lens[i]
    patient = []
    for j in range(real_len):
        visit_vector = x[i, j]  # (2869,)
        # Lấy tất cả chỉ số có giá trị > 0.5 (hoặc == 1)
        code_indices = np.where(visit_vector > 0.5)[0]
        
        if len(code_indices) == 0:
            continue  # bỏ qua visit rỗng
        
        total_codes_in_visits += len(code_indices)
        if len(code_indices) > 1:
            multi_label_count += 1
        
        visit_codes = []
        for code_idx in code_indices:
            code_str = f"D_{int(code_idx):04d}"
            if code_str not in code_to_id:
                code_to_id[code_str] = next_id
                next_id += 1
            visit_codes.append(code_to_id[code_str])
        
        patient.append(visit_codes)  # visit = [mã1, mã2, ...]
    seqs.append(patient)

# Lưu
synth_seqs = f"{DATA_DIR}/synth_mimic3.seqs"
synth_types = f"{DATA_DIR}/synth_mimic3.types"

with open(synth_seqs, 'wb') as f:
    pickle.dump(seqs, f, -1)
with open(synth_types, 'wb') as f:
    pickle.dump(code_to_id, f, -1)

print(f"Done! Saved {len(seqs)} patients → {synth_seqs}")
print(f"       Total unique codes: {len(code_to_id)}")
print(f"       Total multi-label visits: {multi_label_count}")
print(f"       Avg codes per visit: {total_codes_in_visits / sum(lens):.2f}")
