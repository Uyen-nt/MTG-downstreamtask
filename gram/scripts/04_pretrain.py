# gram/scripts/04_train_real.py

import os, sys, subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
GRAM_DIR = SCRIPT_DIR.parent
RESULTS_DIR = GRAM_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GRAM_PY = GRAM_DIR / "model" / "gram.py"

# ====== ĐIỀN ĐÚNG BỘ REMAP (trong input) ======
INPUT_ROOT = Path("/kaggle/input/downstream-data/mtg_downstream_data")

SEQS = INPUT_ROOT / "tree_mimic3" / "tree_mimic3.seqs"     # <-- remap seqs
LABELS = INPUT_ROOT / "tree_mimic3" / "tree_mimic3.labels" # <-- remap labels
TREE_PREFIX = INPUT_ROOT / "tree_mimic3" / "tree_mimic3"   # <-- prefix (không đuôi)
OUT_DIR = RESULTS_DIR / "train_real"

cmd = [
    "python", "-u", str(GRAM_PY),
    str(SEQS),
    str(LABELS),
    str(TREE_PREFIX),
    str(OUT_DIR),
    "--n_epochs", "2",          # tuỳ bạn
    "--batch_size", "64",
    "--rnn_size", "64",
    "--attention_size", "64",
    "--dropout_rate", "0.5",
    "--L2", "0.001",
    "--verbose"
]

print("🚀 Training GRAM on real MIMIC-III (remapped) ...")
print("Command:", " ".join(cmd))
ret = subprocess.call(cmd)
if ret != 0:
    raise SystemExit(ret)
print("✅ Done. Models saved to:", OUT_DIR)
