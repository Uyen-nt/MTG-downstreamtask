# gram/scripts/03_build_tree.py
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()           # gram/scripts/
GRAM_DIR = SCRIPT_DIR.parent                            # gram/
PROJECT_ROOT = GRAM_DIR.parent                          # /kaggle/working/MTG-downstreamtask/
DATA_DIR = PROJECT_ROOT / "data"
GRAM_DATA_DIR = GRAM_DIR / "data"

BUILD_TREES_PY = GRAM_DIR / "model" / "build_trees.py"   # ĐÃ SỬA
HIERARCHY_CSV = DATA_DIR / "icd9_hierarchy.csv"
REAL_SEQS = DATA_DIR / "result" / "mimic3" / "real_mimic3.3digitICD9.seqs"
REAL_TYPES = DATA_DIR / "result" / "mimic3" / "real_mimic3.3digitICD9.types"
TREE_PREFIX = GRAM_DATA_DIR / "tree_mimic3"

# Tạo thư mục
os.makedirs(GRAM_DATA_DIR, exist_ok=True)

# KIỂM TRA TẤT CẢ FILE
for f in [BUILD_TREES_PY, HIERARCHY_CSV, REAL_SEQS, REAL_TYPES]:
    if not f.exists():
        raise FileNotFoundError(f"Không tìm thấy: {f}")

print(f"Đang chạy: {BUILD_TREES_PY.name}")
cmd = [
    "python", str(BUILD_TREES_PY),
    str(HIERARCHY_CSV),
    str(REAL_SEQS),
    str(REAL_TYPES),
    str(TREE_PREFIX)
]

print("Building ICD9 tree...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("LỖI TỪ build_trees.py:")
    print(result.stderr)  # IN RA LỖI CHI TIẾT
    raise RuntimeError(f"build_trees.py thất bại: {result.returncode}")
print(result.stdout)

print(f"Tree saved → {TREE_PREFIX}.level*.pk")
