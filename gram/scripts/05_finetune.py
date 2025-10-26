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

# === TÌM MODEL PRETRAIN (.pt) ===
pretrain_models = glob.glob(f"{PRETRAIN_DIR}/model_best.pt")

if not pretrain_models:
    raise FileNotFoundError(
        f"Không tìm thấy model pretrain tại {PRETRAIN_DIR}/model_best.pt\n"
        "Hãy chạy 04_pretrain.py trước!"
    )

best_model = pretrain_models[0]  # Chỉ có 1 file .pt
finetune_init = f"{FINETUNE_DIR}/pretrain_model.pt"
shutil.copy(best_model, finetune_init)
print(f"Loaded pre-trained weights: {best_model}")
print(f"Copied to: {finetune_init}")

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
