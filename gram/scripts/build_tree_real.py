# gram/scripts/build_tree_real.py

import subprocess

CCS_FILE = "data/ccs_multi_dx_tool_2015.csv"
SEQ_FILE = "gram/data/mimic3.seqs"
TYPE_FILE = "gram/data/mimic3.types"
OUT_PREFIX = "gram/data/mimic3_tree"

def build_tree():
    print("[BUILD TREE FROM REAL MIMIC3]")

    cmd = [
        "python3",
        "gram/model/build_trees.py",
        CCS_FILE,
        SEQ_FILE,
        TYPE_FILE,
        OUT_PREFIX
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ Tree built: gram/data/mimic3_tree.level*.pk")

if __name__ == "__main__":
    build_tree()
