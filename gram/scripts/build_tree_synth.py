# gram/scripts/build_tree_synth.py

import subprocess

CCS_FILE = "data/ccs_multi_dx_tool_2015.csv"
SEQ_FILE = "gram/data/synthetic_3digit.seqs"
TYPE_FILE = "gram/data/synthetic_3digit.types"
OUT_PREFIX = "gram/data/synth3digit_tree"

def main():
    print("[BUILD TREE FROM SYNTHETIC (3DIGIT) USING CCS]")
    cmd = [
        "python3",
        "gram/model/build_trees.py",
        CCS_FILE,
        SEQ_FILE,
        TYPE_FILE,
        OUT_PREFIX
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("Tree built under:", OUT_PREFIX)

if __name__ == "__main__":
    main()
