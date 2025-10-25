# scripts/06_train_hybrid.py
import pickle
import os

DATA_DIR = "data"
REAL_SEQS = "../../data/result/mimic3/real_mimic3.3digitICD9.seqs"
REAL_LABELS = f"{DATA_DIR}/real_mimic3.labels"
SYNTH_SEQS = f"{DATA_DIR}/synth_mimic3.seqs"
SYNTH_LABELS = f"{DATA_DIR}/synth_mimic3.labels"
HYBRID_DIR = "results/hybrid"
K = 2  # 2x synthetic

os.makedirs(HYBRID_DIR, exist_ok=True)

real_seqs = pickle.load(open(REAL_SEQS, 'rb'))
real_labels = pickle.load(open(REAL_LABELS, 'rb'))
synth_seqs = pickle.load(open(SYNTH_SEQS, 'rb'))
synth_labels = pickle.load(open(SYNTH_LABELS, 'rb'))

n_real = len(real_seqs)
merged_seqs = real_seqs + synth_seqs[:n_real * K]
merged_labels = real_labels + synth_labels[:n_real * K]

with open(f"{HYBRID_DIR}/merged.seqs", 'wb') as f:
    pickle.dump(merged_seqs, f, -1)
with open(f"{HYBRID_DIR}/merged.labels", 'wb') as f:
    pickle.dump(merged_labels, f, -1)

print(f"Hybrid data saved → {HYBRID_DIR}/merged.*")
