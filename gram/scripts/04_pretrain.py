# gram/scripts/04_pretrain.py

# ======================================================
# 🚀 Train GRAM model using synthetic data (MTGAN output)
# ======================================================

import os, sys, subprocess
from pathlib import Path

# -----------------------------
# 🧭 Định nghĩa đường dẫn
# -----------------------------
PROJECT_ROOT = Path("/kaggle/working/MTG-downstreamtask")
GRAM_DIR = PROJECT_ROOT / "gram"
DATA_DIR = GRAM_DIR / "data"            # synth_mimic3.* nằm ở đây
RESULTS_DIR = GRAM_DIR / "results"
SYNTH_RESULT_DIR = RESULTS_DIR / "train_synth"
os.makedirs(SYNTH_RESULT_DIR, exist_ok=True)

# -----------------------------
# 📂 Dữ liệu synthetic (MTGAN)
# -----------------------------
SYNTH_SEQS = DATA_DIR / "synth_mimic3.seqs"
SYNTH_LABELS = DATA_DIR / "synth_mimic3.labels"
TREE_PREFIX = DATA_DIR / "tree_synth"   # dummy tree sinh từ 03b_build_tree_synth.py

# -----------------------------
# ⚙️ Cấu hình train GRAM
# -----------------------------
GRAM_PY = GRAM_DIR / "model" / "gram.py"

cmd = [
    "python", "-u", str(GRAM_PY),
    str(SYNTH_SEQS),
    str(SYNTH_LABELS),
    str(TREE_PREFIX),
    str(SYNTH_RESULT_DIR),
    "--n_epochs", "5",           # bạn có thể tăng khi test xong
    "--batch_size", "64",
    "--rnn_size", "64",
    "--attention_size", "64",
    "--dropout_rate", "0.5",
    "--L2", "0.001",
    "--verbose"
]

print("\n🚀 Training GRAM on synthetic MTGAN data...")
print("Command:", " ".join(cmd))
print("─────────────────────────────────────────────")

# -----------------------------
# 📡 Stream log trực tiếp
# -----------------------------
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["AESARA_FLAGS"] = "device=cuda,floatX=float32,optimizer_including=cudnn"

with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as p:
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    ret = p.wait()

# -----------------------------
# ✅ Kết quả
# -----------------------------
if ret == 0:
    print("\n✅ HOÀN TẤT TRAINING GRAM (synthetic)!")
    print(f"→ Model saved in: {SYNTH_RESULT_DIR}")
else:
    print("\n❌ LỖI TRONG QUÁ TRÌNH TRAIN!")
    raise RuntimeError(f"Training thất bại (exit code {ret})")
