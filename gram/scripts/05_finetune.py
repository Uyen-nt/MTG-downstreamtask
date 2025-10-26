# scripts/05_finetune.py
import os
import shutil
import subprocess
import glob

# === CẤU HÌNH ĐƯỜNG DẪN ===
DATA_DIR = "data"
PRETRAIN_DIR = "/kaggle/working/MTG-downstreamtask/gram/results/pretrain"
FINETUNE_DIR = "/kaggle/working/MTG-downstreamtask/gram/results/finetune"
REAL_SEQS = "/kaggle/working/MTG-downstreamtask/data/result/mimic3/real_mimic3.3digitICD9.seqs"
REAL_LABELS = f"{DATA_DIR}/real_mimic3.labels"
TREE = f"{DATA_DIR}/tree_mimic3"

# Tạo thư mục finetune
os.makedirs(FINETUNE_DIR, exist_ok=True)

# === TÌM MODEL PRETRAIN (.npz) ===
pretrain_models = glob.glob(f"{PRETRAIN_DIR}/*.npz")

if not pretrain_models:
    raise FileNotFoundError(
        f"Không tìm thấy model pretrain (.npz) tại {PRETRAIN_DIR}\n"
        "Hãy chạy 04_pretrain.py trước!"
    )

best_model = sorted(pretrain_models)[-1]  # lấy model cuối (best epoch)
finetune_init = f"{FINETUNE_DIR}/pretrain_model.npz"
shutil.copy(best_model, finetune_init)
print(f"✅ Loaded pre-trained weights: {best_model}")
print(f"📦 Copied to: {finetune_init}")

# === CHẠY GRAM VỚI AESARA (hoặc Theano) ===
cmd = [
    "python", "model/gram.py",
    REAL_SEQS,
    REAL_LABELS,
    TREE,
    FINETUNE_DIR,
    "--embed_file", finetune_init,        # Dùng .pt làm embedding
    "--n_epochs", "50",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--attention_size", "128",
    "--dropout_rate", "0.5",
    "--L2", "0.001",
    "--verbose"
]

print("\nFine-tuning on real MIMIC-III data...")
print("Command:", " ".join(cmd))

result = subprocess.run(cmd, check=False)

if result.returncode == 0:
    print("HOÀN TẤT FINETUNE!")
    print(f"→ Model saved in: {FINETUNE_DIR}")
else:
    raise RuntimeError(f"Finetune thất bại: {result.returncode}")
