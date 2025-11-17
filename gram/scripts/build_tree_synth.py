# gram/scripts/build_tree_synth.py

import subprocess
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

CCS_FILE = "data/ccs_multi_dx_tool_2015.csv"
SEQ_FILE = "gram/data/synthetic_converted/synthetic.seqs"   
TYPE_FILE = "gram/data/synthetic_converted/synthetic.types" 
OUT_PREFIX = "gram/data/synth_tree"

def build_tree():
    print("[BUILD TREE FROM SYNTHETIC MIMIC3]")

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
    print("✅ Tree built: gram/data/synth_tree.level*.pk")

if __name__ == "__main__":
    build_tree()
