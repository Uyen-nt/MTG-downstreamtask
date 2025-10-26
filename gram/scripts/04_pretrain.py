# gram/scripts/04_pretrain.py
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
GRAM_DIR = SCRIPT_DIR.parent
DATA_DIR = GRAM_DIR / "data"
RESULTS_DIR = GRAM_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ĐƯỜNG DẪN
GRAM_PY = GRAM_DIR / "model" / "gram.py"
SYNTH_SEQS = DATA_DIR / "synth_mimic3.seqs"
SYNTH_LABELS = DATA_DIR / "synth_mimic3.labels"
TREE_PREFIX = DATA_DIR / "tree_mimic3"  # chỉ là prefix
PRETRAIN_DIR = RESULTS_DIR / "pretrain"

# KIỂM TRA FILE CẦN THIẾT
missing = []
for f in [GRAM_PY, SYNTH_SEQS, SYNTH_LABELS]:
    if not f.exists():
        missing.append(f)

# KIỂM TRA CÁC FILE .level*.pk
level_files = list(DATA_DIR.glob("tree_mimic3.level*.pk"))
if not level_files:
    missing.append("tree_mimic3.level*.pk (không tìm thấy file level)")

if missing:
    print("CẢNH BÁO: Thiếu file cần thiết!")
    for item in missing:
        if "level" in str(item):
            print(f"  → Không tìm thấy file tree_mimic3.level*.pk")
        else:
            print(f"  → Không tìm thấy: {item}")
    raise FileNotFoundError("Vui lòng chạy lại 03_build_tree.py!")

print("Pre-training on synthetic data...")
cmd = [
    "python", str(GRAM_PY),
    str(SYNTH_SEQS),
    str(SYNTH_LABELS),
    str(TREE_PREFIX),  # gram.py sẽ tự tìm .level*.pk
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
