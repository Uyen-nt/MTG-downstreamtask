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

print("Pre-training on remapped data...")

cmd = [
    "python", "-u", str(GRAM_PY),            # -u để unbuffered
    str(SEQS),
    str(LABELS),
    str(TREE_PREFIX),
    str(OUT_DIR),
    "--n_epochs", "30",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--verbose"
]

# Truyền env để chắc chắn unbuffered
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
) as p:
    for line in p.stdout:
        sys.stdout.write(line)   # stream ngay lập tức
        sys.stdout.flush()
    ret = p.wait()

if ret != 0:
    raise RuntimeError("Pretrain thất bại!")
print("✅ Hoàn tất pretrain, model lưu trong:", OUT_DIR)
