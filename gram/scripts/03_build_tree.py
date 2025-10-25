import os
import subprocess

DATA_DIR = "data"
HIERARCHY_CSV = "icd9_hierarchy.csv"  # bạn cần có file này
REAL_SEQS = "../../data/result/mimic3/real_mimic3.3digitICD9.seqs"
REAL_TYPES = "../../data/result/mimic3/real_mimic3.3digitICD9.types"
TREE_PREFIX = f"{DATA_DIR}/tree_mimic3"

os.makedirs(DATA_DIR, exist_ok=True)

cmd = [
    "python", "model/build_trees.py",
    HIERARCHY_CSV,
    REAL_SEQS,
    REAL_TYPES,
    TREE_PREFIX
]

print("Building ICD9 tree...")
subprocess.run(cmd, check=True)
print(f"Tree saved → {TREE_PREFIX}.level*.pk")
