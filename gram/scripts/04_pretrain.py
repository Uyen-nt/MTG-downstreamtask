# scripts/04_pretrain.py
import os
import subprocess

DATA_DIR = "data"
RESULTS_DIR = "results/pretrain"
SYNTH_SEQS = f"{DATA_DIR}/synth_mimic3.seqs"
SYNTH_LABELS = f"{DATA_DIR}/synth_mimic3.labels"
TREE = f"{DATA_DIR}/tree_mimic3"

os.makedirs(RESULTS_DIR, exist_ok=True)

cmd = [
    "python", "model/gram.py",
    SYNTH_SEQS, SYNTH_LABELS, TREE, RESULTS_DIR,
    "--n_epochs", "30",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--verbose"
]

print("Pre-training on synthetic data...")
subprocess.run(cmd, check=True)
print(f"Pre-trained model → {RESULTS_DIR}/*.npz")
