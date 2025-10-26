# gram/scripts/04_pretrain.py
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()      # gram/scripts/
GRAM_DIR = SCRIPT_DIR.parent                        # gram/
DATA_DIR = GRAM_DIR / "data"                        # ĐÃ SỬA: gram/data/
RESULTS_DIR = GRAM_DIR / "results"                  # ĐÃ SỬA: gram/results/
os.makedirs(RESULTS_DIR, exist_ok=True)

# ĐƯỜNG DẪN ĐÚNG TRONG KAGGLE
GRAM_PY = GRAM_DIR / "model" / "gram.py"  
SYNTH_SEQS = DATA_DIR / "synth_mimic3.seqs"
SYNTH_LABELS = DATA_DIR / "synth_mimic3.labels"
TREE_PREFIX = DATA_DIR / "tree_mimic3"
PRETRAIN_DIR = RESULTS_DIR / "pretrain"

# KIỂM TRA FILE
missing = [f for f in [GRAM_PY, SYNTH_SEQS, SYNTH_LABELS, TREE_PREFIX] if not f.exists()]
if missing:
    for f in missing:
        print(f"Không tìm thấy: {f}")
    raise FileNotFoundError("Thiếu file cần thiết!")

print("Pre-training on synthetic data...")
cmd = [
    "python", str(GRAM_PY),
    str(SYNTH_SEQS),
    str(SYNTH_LABELS),
    str(TREE_PREFIX),
    str(PRETRAIN_DIR),
    "--n_epochs", "30",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--verbose"
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("LỖI TỪ gram.py:")
    print(result.stderr)
    raise RuntimeError(f"Pretrain thất bại: {result.returncode}")

print("HOÀN TẤT PRETRAIN!")
print(f"→ Model saved: {PRETRAIN_DIR}/model_best.pt")
print(f"→ Vocab saved: {PRETRAIN_DIR}/vocab.pkl")
