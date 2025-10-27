# scripts/06_train_hybrid.py
import pickle
import os
from pathlib import Path

# =========================
# 📁 CẤU HÌNH ĐƯỜNG DẪN
# =========================
DATA_DIR = Path("/kaggle/input/downstream-data/mtg_downstream_data")
RESULTS_DIR = Path("/kaggle/working/hybrid_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# 📦 FILE DỮ LIỆU
# =========================
REAL_SEQS = DATA_DIR / "real_mimic3.seqs"
REAL_LABELS = DATA_DIR / "real_mimic3.labels"
SYNTH_SEQS = DATA_DIR / "synth_mimic3.seqs"
SYNTH_LABELS = DATA_DIR / "synth_mimic3.labels"

K = 2  # synthetic gấp đôi real

# =========================
# 🔄 GHÉP DỮ LIỆU LAI (HYBRID)
# =========================
print("🔹 Loading real data ...")
real_seqs = pickle.load(open(REAL_SEQS, "rb"))
real_labels = pickle.load(open(REAL_LABELS, "rb"))

print("🔹 Loading synthetic data ...")
synth_seqs = pickle.load(open(SYNTH_SEQS, "rb"))
synth_labels = pickle.load(open(SYNTH_LABELS, "rb"))

n_real = len(real_seqs)
print(f"📊 Real samples: {n_real}, Synthetic samples: {len(synth_seqs)}")

# Lấy K lần synthetic để ghép
merged_seqs = real_seqs + synth_seqs[: n_real * K]
merged_labels = real_labels + synth_labels[: n_real * K]

# Lưu lại
with open(RESULTS_DIR / "merged.seqs", "wb") as f:
    pickle.dump(merged_seqs, f, -1)
with open(RESULTS_DIR / "merged.labels", "wb") as f:
    pickle.dump(merged_labels, f, -1)

print(f"✅ Hybrid data saved → {RESULTS_DIR}/merged.*")
print(f"📊 Tổng số mẫu: {len(merged_seqs)} (real={n_real}, synth={n_real*K})")
