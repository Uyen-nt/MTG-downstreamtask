# gram/scripts/build_tree_synthetic.py

import subprocess

# ========================
# FIXED PATHS (KAGGLE)
# ========================
CCS_FILE = "data/ccs_multi_dx_tool_2015.csv"
SEQ_FILE = "gram/data/synthetic.seqs"
TYPE_FILE = "gram/data/synthetic.types"
OUT_PREFIX = "gram/data/synthetic_tree"


def build_tree(ccs_file, seq_file, type_file, out_prefix):
    print("[BUILD TREE]")
    print(" CCS file:", ccs_file)
    print(" SEQ file:", seq_file)
    print(" TYPE file:", type_file)
    print(" OUT prefix:", out_prefix)

    cmd = [
        "python3",
        "gram/model/build_trees.py",
        ccs_file,
        seq_file,
        type_file,
        out_prefix
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ build_tree_synthetic DONE!")


if __name__ == "__main__":
    build_tree(CCS_FILE, SEQ_FILE, TYPE_FILE, OUT_PREFIX)
