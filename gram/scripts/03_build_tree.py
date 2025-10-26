# gram/scripts/03_build_tree.py
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GRAM_DATA_DIR = PROJECT_ROOT / "gram" / "data"

# ĐƯỜNG DẪN CHÍNH XÁC
HIERARCHY_CSV = DATA_DIR / "icd9_hierarchy.csv"          # ĐÃ SỬA
REAL_SEQS = DATA_DIR / "result" / "mimic3" / "real_mimic3.3digitICD9.seqs"
REAL_TYPES = DATA_DIR / "result" / "mimic3" / "real_mimic3.3digitICD9.types"
TREE_PREFIX = GRAM_DATA_DIR / "tree_mimic3"

os.makedirs(GRAM_DATA_DIR, exist_ok=True)

for f in [HIERARCHY_CSV, REAL_SEQS, REAL_TYPES]:
    if not f.exists():
        raise FileNotFoundError(f"Không tìm thấy: {f}")

cmd = [
    "python", str(SCRIPT_DIR / ".." / "model" / "build_trees.py"),
    str(HIERARCHY_CSV),
    str(REAL_SEQS),
    str(REAL_TYPES),
    str(TREE_PREFIX)
]

print("Building ICD9 tree...")
result = subprocess.run(cmd, check=True, capture_output=True, text=True)
print(result.stdout)
print(f"Tree saved → {TREE_PREFIX}.level*.pk")
