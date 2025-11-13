import argparse
import subprocess

def build_tree(ccs_file, seq_file, type_file, out_prefix):
    print("[BUILD TREE]")
    print(" CCS file: ", ccs_file)
    print(" SEQ file:", seq_file)
    print(" TYPE file:", type_file)
    print(" OUT prefix:", out_prefix)

    # Gọi build_trees.py trong thư mục model/
    cmd = [
        "python3",
        "gram/model/build_trees.py",
        ccs_file,
        seq_file,
        type_file,
        out_prefix
    ]

    print("[Running] ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("DONE building tree!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccs_file", required=True)
    parser.add_argument("--seq_file", required=True)
    parser.add_argument("--type_file", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    build_tree(args.ccs_file, args.seq_file, args.type_file, args.out_prefix)
