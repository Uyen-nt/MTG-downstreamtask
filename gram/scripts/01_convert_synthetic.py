# scripts/01_convert_synthetic.py
import numpy as np
import pickle
import os

SYNTH_NPZ = "data/result/mimic3/synthetic_mimic3.npz"
DATA_DIR = "gram/data"

os.makedirs(DATA_DIR, exist_ok=True)

print("Loading synthetic data...")
data = np.load(SYNTH_NPZ, allow_pickle=True)

# LẤY DỮ LIỆU
x = data['x']        # shape: (N, max_len, num_codes) hoặc (N, T, C)
lens = data['lens']  # shape: (N,) → số visit thật của mỗi bệnh nhân

print(f"Loaded 'x': {x.shape}, 'lens': {lens.shape}")

# Chuyển thành list of list of list
patients = []
for i in range(len(lens)):
    real_len = lens[i]
    patient = x[i, :real_len]  # cắt bỏ padding
    patients.append(patient.tolist())  # chuyển thành list

# Tạo types + seqs
types = {}
code_to_id = {}
next_id = 0
seqs = []

print("Converting to GRAM format (seqs)...")
for patient in patients:
    new_patient = []
    for visit in patient:
        new_visit = []
        for code in visit:
            # code là số nguyên → chuyển thành D_xxxx
            code_str = f"D_{int(code):04d}"
            if code_str not in code_to_id:
                code_to_id[code_str] = next_id
                next_id += 1
            new_visit.append(code_to_id[code_str])
        new_patient.append(new_visit)
    seqs.append(new_patient)

# Lưu
synth_seqs = f"{DATA_DIR}/synth_mimic3.seqs"
synth_types = f"{DATA_DIR}/synth_mimic3.types"

with open(synth_seqs, 'wb') as f:
    pickle.dump(seqs, f, -1)
with open(synth_types, 'wb') as f:
    pickle.dump(code_to_id, f, -1)

print(f"Done! Saved {len(seqs)} patients → {synth_seqs}")
print(f"       Total unique codes: {len(code_to_id)}")
