# gram/scripts/build_tree_synth.py

import os
import subprocess
import sys

# ROOT project: MTG-downstreamtask
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

# ====== CONFIG ======
# File CCS multi-level (giống lúc build tree cho mimic real)
CCS_FILE  = "data/ccs_multi_dx_tool_2015.csv"

# Seq + types vừa convert từ synthetic MTGAN sang format GRAM
SEQ_FILE  = "gram/data/synthetic_tmp.seqs"
TYPE_FILE = "gram/data/synthetic_tmp.types"

# Prefix cho cây synthetic
OUT_PREFIX = "gram/data/synth_tree"


def build_tree_synth():
    print("[BUILD TREE FROM SYNTHETIC (MTGAN) USING CCS]")

    cmd = [
        "python3",
        "gram/model/build_trees.py",
        CCS_FILE,
        SEQ_FILE,
        TYPE_FILE,
        OUT_PREFIX
    ]

    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("✅ Tree built under prefix:", OUT_PREFIX)
    print("   -", OUT_PREFIX + ".level1.pk")
    print("   -", OUT_PREFIX + ".level2.pk")
    print("   -", OUT_PREFIX + ".level3.pk")
    print("   -", OUT_PREFIX + ".level4.pk")
    print("   -", OUT_PREFIX + ".level5.pk")
    print("   -", OUT_PREFIX + ".types")
    print("   -", OUT_PREFIX + ".seqs")


if __name__ == "__main__":
    build_tree_synth()
