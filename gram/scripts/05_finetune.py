# scripts/05_finetune.py
import os
import shutil
import subprocess
import glob

DATA_DIR = "data"
PRETRAIN_DIR = "results/pretrain"
FINETUNE_DIR = "results/finetune"
REAL_SEQS = "../../data/result/mimic3/real_mimic3.3digitICD9.seqs"
REAL_LABELS = f"{DATA_DIR}/real_mimic3.labels"
TREE = f"{DATA_DIR}/tree_mimic3"

os.makedirs(FINETUNE_DIR, exist_ok=True)

# Tìm model pretrain tốt nhất
pretrain_models = glob.glob(f"{PRETRAIN_DIR}/*.npz")
best_model = max(pretrain_models, key=os.path.getctime)  # mới nhất
shutil.copy(best_model, f"{FINETUNE_DIR}/0.npz")
print(f"Loaded pre-trained weights: {best_model}")

cmd = [
    "python", "model/gram.py",
    REAL_SEQS, REAL_LABELS, TREE, FINETUNE_DIR,
    "--n_epochs", "50",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--verbose"
]

print("Fine-tuning on real data...")
subprocess.run(cmd, check=True)
