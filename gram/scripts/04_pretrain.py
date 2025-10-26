# gram/scripts/04_pretrain.py

import os, subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
GRAM_DIR = SCRIPT_DIR.parent
DATA_DIR = GRAM_DIR / "data"
RESULTS_DIR = GRAM_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GRAM_PY = GRAM_DIR / "model" / "gram.py"

SEQS = DATA_DIR / "tree_mimic3.seqs"
LABELS = DATA_DIR / "tree_mimic3.labels"
TREE_PREFIX = DATA_DIR / "tree_mimic3"
OUT_DIR = RESULTS_DIR / "pretrain_real"

cmd = [
    "python", str(GRAM_PY),
    str(SEQS),
    str(LABELS),
    str(TREE_PREFIX),
    str(OUT_DIR),
    "--n_epochs", "30",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--verbose"
]

print("Pre-training on remapped data...")
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# 🔁 Hiển thị log theo thời gian thực
for line in iter(process.stdout.readline, ''):
    print(line, end='')

process.wait()

if process.returncode != 0:
    print("\n❌ LỖI TỪ model/gram.py:")
    raise RuntimeError("Pretrain thất bại!")
else:
    print("\n✅ Hoàn tất pretrain, model lưu trong:", OUT_DIR)
