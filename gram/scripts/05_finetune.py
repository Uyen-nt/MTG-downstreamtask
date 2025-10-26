# scripts/05_finetune.py
import os, sys, subprocess, shutil, glob
from pathlib import Path

# =========================
# 🧭 CẤU HÌNH ĐƯỜNG DẪN
# =========================
PROJECT_ROOT = Path("/kaggle/working/MTG-downstreamtask")
GRAM_DIR = PROJECT_ROOT / "gram"
DATA_DIR = GRAM_DIR / "data"
RESULTS_DIR = GRAM_DIR / "results"
PRETRAIN_DIR = RESULTS_DIR
FINETUNE_DIR = RESULTS_DIR / "finetune_synth"
os.makedirs(FINETUNE_DIR, exist_ok=True)

# =========================
# 📂 FILE DỮ LIỆU SYNTHETIC (MTGAN)
# =========================
SYNTH_SEQS = DATA_DIR / "synth_mimic3.seqs"
SYNTH_LABELS = DATA_DIR / "synth_mimic3.labels"
TREE_PREFIX = DATA_DIR / "tree_synth"  

# =========================
# 🔍 TÌM FILE PRETRAIN (.npz)
# =========================
pretrain_models = sorted(glob.glob(str(PRETRAIN_DIR / "*.npz")))
if not pretrain_models:
    raise FileNotFoundError(
        f"❌ Không tìm thấy model pretrain (.npz) tại {PRETRAIN_DIR}\n"
        "👉 Hãy chạy 04_pretrain.py trước!"
    )

best_model = pretrain_models[-1]
finetune_init = FINETUNE_DIR / "pretrain_model.npz"
shutil.copy(best_model, finetune_init)
print(f"✅ Loaded pre-trained weights: {best_model}")
print(f"📦 Copied to: {finetune_init}")

# =========================
# ⚙️ LỆNH CHẠY GRAM FINE-TUNE
# =========================
GRAM_PY = GRAM_DIR / "model" / "gram.py"

cmd = [
    "python", "-u", str(GRAM_PY),
    str(SYNTH_SEQS),
    str(SYNTH_LABELS),
    str(TREE_PREFIX),
    str(FINETUNE_DIR),
    "--embed_file", str(finetune_init),
    "--n_epochs", "50",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--attention_size", "128",
    "--dropout_rate", "0.5",
    "--L2", "0.001",
    "--verbose"
]

print("\n🚀 Fine-tuning on synthetic MTGAN data...")
print("Command:", " ".join(cmd))
print("─────────────────────────────────────────────────────────────")

# =========================
# 📡 STREAM LOG TRỰC TIẾP
# =========================
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
) as p:
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    ret = p.wait()

if ret == 0:
    print("\n✅ HOÀN TẤT FINE-TUNE!")
    print(f"→ Model saved in: {FINETUNE_DIR}")
else:
    print("\n❌ LỖI TRONG QUÁ TRÌNH FINE-TUNE!")
    raise RuntimeError(f"Finetune thất bại (exit code {ret})")
