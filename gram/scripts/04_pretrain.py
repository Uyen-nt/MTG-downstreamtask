# scripts/04_pretrain.py
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
GRAM_DIR = SCRIPT_DIR.parent
DATA_DIR = GRAM_DIR / "data"
RESULTS_DIR = GRAM_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ĐÚNG FILE TRAIN MODEL
GRAM_MODEL_PY = GRAM_DIR / "model" / "gram.py"   # <-- ĐÚNG
SYNTH_SEQS = DATA_DIR / "synth_mimic3.seqs"
SYNTH_LABELS = DATA_DIR / "synth_mimic3.labels"
TREE_PREFIX = DATA_DIR / "tree_mimic3"
PRETRAIN_DIR = RESULTS_DIR / "pretrain"
os.makedirs(PRETRAIN_DIR, exist_ok=True)

# KIỂM TRA FILE
missing = []
for f in [GRAM_MODEL_PY, SYNTH_SEQS, SYNTH_LABELS]:
    if not f.exists():
        missing.append(f)

level_files = list(DATA_DIR.glob("tree_mimic3.level*.pk"))
if not level_files:
    missing.append("tree_mimic3.level*.pk")

if missing:
    print("THIẾU FILE:")
    for m in missing:
        print(f"  → {m}")
    raise FileNotFoundError("Chạy 03_build_tree.py trước!")

print("Pre-training on synthetic data...")
cmd = [
    "python", str(GRAM_MODEL_PY),        # <-- GỌI ĐÚNG model/gram.py
    str(SYNTH_SEQS),
    str(SYNTH_LABELS),
    str(TREE_PREFIX),
    str(PRETRAIN_DIR),
    "--n_epochs", "30",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--attention_size", "128",
    "--dropout_rate", "0.5",
    "--L2", "0.001",
    "--verbose"
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("LỖI TỪ model/gram.py:")
    print(result.stderr)
    raise RuntimeError(f"Pretrain thất bại: {result.returncode}")

print("HOÀN TẤT PRETRAIN!")
print(f"→ Model saved: {PRETRAIN_DIR}/*.npz (Theano/Aesara)")
print(f"→ Vocab saved: {PRETRAIN_DIR}/vocab.pkl")
