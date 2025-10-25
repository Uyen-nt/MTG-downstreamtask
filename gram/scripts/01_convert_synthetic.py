# scripts/01_convert_synthetic.py
# ĐÃ SỬA: xử lý one-hot vector → chỉ lấy mã có giá trị 1

import numpy as np
import pickle
import os

SYNTH_NPZ = "data/result/synthetic_mimic3.npz"
DATA_DIR = "gram/data"

os.makedirs(DATA_DIR, exist_ok=True)

print("Loading synthetic data...")
data = np.load(SYNTH_NPZ, allow_pickle=True)
x = data['x']        # (6000, 34, 2869)
lens = data['lens']  # (6000,)

print(f"Loaded 'x': {x.shape}, 'lens': {lens.shape}")

# Tạo seqs
seqs = []
code_to_id = {}
next_id = 0

print("Converting one-hot to code indices...")
for i in range(len(lens)):
    real_len = lens[i]
    patient = []
    for j in range(real_len):
        visit_vector = x[i, j]  # (2869,)
        # Tìm chỉ số có giá trị 1
        code_idx = np.where(visit_vector == 1)[0]
        if len(code_idx) != 1:
            print(f"Warning: visit {j} of patient {i} has {len(code_idx)} codes")
            code_idx = code_idx[0] if len(code_idx) > 0 else 0
        else:
            code_idx = code_idx[0]
        
        code_str = f"D_{int(code_idx):04d}"
        if code_str not in code_to_id:
            code_to_id[code_str] = next_id
            next_id += 1
        patient.append([code_to_id[code_str]])  # visit = [mã]
    seqs.append(patient)

# Lưu
synth_seqs = f"{DATA_DIR}/synth_mimic3.seqs"
synth_types = f"{DATA_DIR}/synth_mimic3.types"

with open(synth_seqs, 'wb') as f:
    pickle.dump(seqs, f, -1)
with open(synth_types, 'wb') as f:
    pickle.dump(code_to_id, f, -1)

print(f"Done! Saved {len(seqs)} patients → {synth_seqs}")
print(f"       Total unique codes: {len(code_to_id)} (should be 2869)")
