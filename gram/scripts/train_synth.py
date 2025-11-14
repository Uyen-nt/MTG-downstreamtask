# gram/scripts/train_synth.py

import subprocess

# ========================
# FIXED PATHS (KAGGLE)
# ========================
SEQ_FILE = "gram/data/synthetic_tree.seqs"
LABEL_FILE = "gram/data/synthetic_tree.seqs"   # dùng chính synthetic để pretrain
TREE_PREFIX = "gram/data/synthetic_tree"
OUT_PREFIX = "gram/data/synth_pretrain"


def train(seq_file, label_file, tree_prefix, out_prefix):
    print("[TRAIN GRAM PRETRAIN]")

    cmd = [
        "python3",
        "gram/model/gram.py",
        seq_file,
        label_file,
        tree_prefix,
        out_prefix,
        "--embed_size", "128",
        "--rnn_size", "128",
        "--attention_size", "128",
        "--batch_size", "64",
        "--n_epochs", "20",
        "--L2", "0.001",
        "--dropout_rate", "0.5",
        "--verbose"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ train_synth DONE!")


if __name__ == "__main__":
    train(SEQ_FILE, LABEL_FILE, TREE_PREFIX, OUT_PREFIX)
