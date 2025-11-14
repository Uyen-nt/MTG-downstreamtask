# gram/scripts/build_tree_real.py

import subprocess
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

CCS_FILE = "data/ccs_multi_dx_tool_2015.csv"
SEQ_FILE = "gram/data/mimic.3digitICD9.seqs"   
TYPE_FILE = "gram/data/mimic.3digitICD9.types" 
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
