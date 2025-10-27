# scripts/04_train_real.py
import os, sys, subprocess
from pathlib import Path

# =========================
# 🧭 CẤU HÌNH ĐƯỜNG DẪN
# =========================
PROJECT_ROOT = Path("/kaggle/working/MTG-downstreamtask")
GRAM_DIR = PROJECT_ROOT / "gram"
RESULTS_DIR = GRAM_DIR / "results"
REAL_RESULT_DIR = RESULTS_DIR / "train_real"
os.makedirs(REAL_RESULT_DIR, exist_ok=True)

# =========================
# 📂 DỮ LIỆU REMAPPED (CHUẨN CHO TRAIN)
# =========================
DATA_ROOT = Path("/kaggle/input/downstream-data/mtg_downstream_data")

REAL_SEQS = DATA_ROOT / "tree_mimic3.seqs"              # ✅ remapped seqs
REAL_LABELS = DATA_ROOT / "tree_mimic3.labels"          # ✅ remapped labels
TREE_PREFIX = DATA_ROOT / "tree_mimic3/tree_mimic3"     # ✅ ICD tree prefix
EMBED_INIT = DATA_ROOT / "pretrain_model.npz"           # ✅ pretrained model để khởi tạo embedding

# =========================
# ⚙️ LỆNH CHẠY GRAM TRAIN
# =========================
GRAM_PY = GRAM_DIR / "model" / "gram.py"

cmd = [
    "python", "-u", str(GRAM_PY),
    str(REAL_SEQS),
    str(REAL_LABELS),
    str(TREE_PREFIX),
    str(REAL_RESULT_DIR),
    "--n_epochs", "2",
    "--batch_size", "64",
    "--rnn_size", "64",
    "--attention_size", "64",
    "--dropout_rate", "0.5",
    "--L2", "0.001",
    "--verbose"
]

print("\n🚀 Training GRAM on remapped real MIMIC-III data...")
print("Command:", " ".join(cmd))
print("─────────────────────────────────────────────────────────────")

# =========================
# 📡 STREAM LOG TRỰC TIẾP
# =========================
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["AESARA_FLAGS"] = "device=cuda,floatX=float32,optimizer_including=cudnn"

with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as p:
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    ret = p.wait()

if ret == 0:
    print("\n✅ HOÀN TẤT TRAINING REAL MIMIC-III (remapped)!")
    print(f"→ Model saved in: {REAL_RESULT_DIR}")
else:
    print("\n❌ LỖI TRONG QUÁ TRÌNH TRAIN!")
    raise RuntimeError(f"Training thất bại (exit code {ret})")
