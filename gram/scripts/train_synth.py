import argparse
import subprocess

def train(seq_file, label_file, tree_prefix, out_prefix):
    print("[TRAIN PRETRAIN GRAM]")
    print(" Seq file:", seq_file)
    print(" Label file:", label_file)
    print(" Tree prefix:", tree_prefix)
    print(" Out prefix:", out_prefix)

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
        "--log_eps", "1e-8"
    ]

    print("[Running] ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("DONE training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_file", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--tree_prefix", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    train(args.seq_file, args.label_file, args.tree_prefix, args.out_prefix)
