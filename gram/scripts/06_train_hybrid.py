# scripts/06_train_hybrid.py
import pickle
import os
from pathlib import Path

# =========================
# 📁 CẤU HÌNH ĐƯỜNG DẪN
# =========================
PROJECT_ROOT = Path("/kaggle/working/MTG-downstreamtask/gram")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
HYBRID_DIR = RESULTS_DIR / "hybrid"
os.makedirs(HYBRID_DIR, exist_ok=True)

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
real_seqs = pickle.load(open(REAL_SEQS, "rb"))
real_labels = pickle.load(open(REAL_LABELS, "rb"))
synth_seqs = pickle.load(open(SYNTH_SEQS, "rb"))
synth_labels = pickle.load(open(SYNTH_LABELS, "rb"))

n_real = len(real_seqs)
merged_seqs = real_seqs + synth_seqs[: n_real * K]
merged_labels = real_labels + synth_labels[: n_real * K]

with open(HYBRID_DIR / "merged.seqs", "wb") as f:
    pickle.dump(merged_seqs, f, -1)
with open(HYBRID_DIR / "merged.labels", "wb") as f:
    pickle.dump(merged_labels, f, -1)

print(f"✅ Hybrid data saved → {HYBRID_DIR}/merged.*")
print(f"📊 Tổng số mẫu: {len(merged_seqs)} (real={n_real}, synth={n_real*K})")
