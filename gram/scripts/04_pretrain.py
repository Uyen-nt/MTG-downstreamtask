# gram/scripts/04_pretrain.py

from pathlib import Path
import os, subprocess

SCRIPT_DIR = Path(__file__).parent.resolve()
GRAM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = GRAM_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = GRAM_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GRAM_PY = GRAM_DIR / "model" / "gram.py"

# DÙNG REAL thay vì SYNTH để đồng bộ với tree_mimic3.*
REAL_SEQS   = DATA_DIR / "result" / "mimic3" / "real_mimic3.3digitICD9.seqs"
REAL_LABELS = GRAM_DIR / "data" / "real_mimic3.labels"
TREE_PREFIX = GRAM_DIR / "data" / "tree_mimic3"
PRETRAIN_DIR = RESULTS_DIR / "pretrain"

missing = []
for f in [GRAM_PY, REAL_SEQS, REAL_LABELS]:
    if not f.exists():
        missing.append(f)

level_files = list((GRAM_DIR / "data").glob("tree_mimic3.level*.pk"))
if not level_files:
    missing.append("tree_mimic3.level*.pk (không tìm thấy file level)")

if missing:
    print("CẢNH BÁO: Thiếu file cần thiết!")
    for item in missing:
        print(f"  → Không tìm thấy: {item}")
    raise FileNotFoundError("Vui lòng chạy lại 03_build_tree.py!")

print("Pre-training on REAL data (khớp tree_mimic3)...")
cmd = [
    "python", str(GRAM_PY),
    str(REAL_SEQS),
    str(REAL_LABELS),
    str(TREE_PREFIX),
    str(PRETRAIN_DIR),
    "--n_epochs", "30",
    "--batch_size", "100",
    "--rnn_size", "128",
    "--verbose"
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("LỖI TỪ model/gram.py:")
    print(result.stderr)
    raise RuntimeError(f"Pretrain thất bại: {result.returncode}")

print("HOÀN TẤT PRETRAIN!")
print(f"→ Model saved: {PRETRAIN_DIR}/model_best.pt")
print(f"→ Vocab saved: {PRETRAIN_DIR}/vocab.pkl")
