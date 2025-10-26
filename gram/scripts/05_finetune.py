# scripts/05_finetune.py
import os
import shutil
import subprocess
import glob

# === CẤU HÌNH ĐƯỜNG DẪN ===
DATA_DIR = "data"
PRETRAIN_DIR = "/kaggle/working/MTG-downstreamtask/gram/results/pretrain_real"
FINETUNE_DIR = "/kaggle/working/MTG-downstreamtask/gram/results/finetune"
REAL_SEQS = "/kaggle/working/MTG-downstreamtask/data/result/mimic3/real_mimic3.3digitICD9.seqs"
REAL_LABELS = f"{DATA_DIR}/real_mimic3.labels"
TREE = f"{DATA_DIR}/tree_mimic3"

os.makedirs(FINETUNE_DIR, exist_ok=True)

# === TÌM MODEL PRETRAIN (.npz) ===
pretrain_models = sorted(glob.glob(f"{PRETRAIN_DIR}/*.npz"))
if not pretrain_models:
    raise FileNotFoundError(
        f"Không tìm thấy model pretrain (.npz) tại {PRETRAIN_DIR}\n"
        "Hãy chạy 04_pretrain.py trước!"
    )


best_model = sorted(pretrain_models)[-1]
finetune_init = f"{FINETUNE_DIR}/pretrain_model.npz"
shutil.copy(best_model, finetune_init)
print(f"✅ Loaded pre-trained weights: {best_model}")
print(f"📦 Copied to: {finetune_init}")

# === CHẠY GRAM VỚI AESARA ===
cmd = [
    "python", "model/gram.py",
    REAL_SEQS,
    REAL_LABELS,
    TREE,
    FINETUNE_DIR,
    "--embed_file", finetune_init,
    "--n_epochs", "50",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--attention_size", "128",
    "--dropout_rate", "0.5",
    "--L2", "0.001",
    "--verbose"
]

print("\n🚀 Fine-tuning on real MIMIC-III data...")
print("Command:", " ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("✅ HOÀN TẤT FINETUNE!")
    print(f"→ Model saved in: {FINETUNE_DIR}")
else:
    print("❌ LỖI TỪ model/gram.py:")
    print(result.stderr)
    raise RuntimeError(f"Finetune thất bại: {result.returncode}")
