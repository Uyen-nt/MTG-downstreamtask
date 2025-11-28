# gpt/preprocess_synthetic.py
import numpy as np
import pickle
from tqdm import tqdm
import os

# Đường dẫn đúng
data_path = "data/result/synthetic_mimic3.npz"
data = np.load(data_path)
x = data['x']      # (1500, 34, 2869)
lens = data['lens']

print(f"Loaded: {x.shape}, lens max: {lens.max()}")

# Tìm các code xuất hiện
active_codes = np.where(np.any(x > 0.5, axis=(0,1)))[0]
print(f"Active codes: {len(active_codes)}")

code_to_index = {code: idx for idx, code in enumerate(active_codes)}
index_to_code = {idx: code for code, idx in code_to_index.items()}
vocab_size = len(active_codes)
eos_token = vocab_size  # End-of-Visit token
pad_token = vocab_size + 1

print(f"Vocab size: {vocab_size}, EOS: {eos_token}, PAD: {pad_token}")

# Chuyển sang sequence
records = []
for i in tqdm(range(len(x)), desc="Converting patients"):
    visits = []
    for v in range(int(lens[i])):
        codes = np.where(x[i, v] > 0.5)[0]
        codes_idx = [code_to_index[c] for c in codes if c in code_to_index]
        if len(codes_idx) > 0:
            visits.append(sorted(codes_idx))
    
    if len(visits) >= 2:
        # Flatten thành sequence: code1 code2 ... EOS code1 ... EOS ...
        seq = []
        for visit in visits:
            seq.extend(visit)
            seq.append(eos_token)
        records.append(seq)

print(f"Final patients: {len(records)}, Avg length: {np.mean([len(r) for r in records]):.1f}")

# Chia train/val
np.random.seed(42)
np.random.shuffle(records)
train_size = int(0.9 * len(records))
train_data = records[:train_size]
val_data = records[train_size:]

# Override config
from config import GPTConfig
config = GPTConfig(
    total_vocab_size=vocab_size + 2,  # codes + EOS + PAD
    n_positions=1024,
    n_ctx=1024,
    batch_size=12,
    epoch=120,
    lr=3e-4
)

# Lưu tất cả vào gpt/
save_data = {
    'train': train_data,
    'val': val_data,
    'code_to_index': code_to_index,
    'index_to_code': index_to_code,
    'vocab_size': vocab_size,
    'eos_token': eos_token,
    'pad_token': pad_token,
    'config': config
}

os.makedirs("gpt", exist_ok=True)
with open("gpt/processed_synthetic.pkl", "wb") as f:
    pickle.dump(save_data, f)

print("DONE! Saved to gpt/processed_synthetic.pkl")
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
print(f"Config: vocab={config.total_vocab_size}, n_layer={config.n_layer}, n_embd={config.n_embd}")
