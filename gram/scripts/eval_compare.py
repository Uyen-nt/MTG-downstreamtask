# gram/scripts/eval_compare.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
from sklearn.metrics import jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch


# ===============================
# CONFIG
# ===============================
TREE_PREFIX = "gram/data/mimic3_tree"
TEST_FILE = "gram/data/mimic.seqs"

SYNTH_MODEL = "gram/data/synth_train_best.pt"
FINETUNE_MODEL = "gram/data/finetuned.pt"


# ===============================
# UTILITIES
# ===============================
def clean_seqs(seqs):
    clean = []
    for p in seqs:
        v = [x for x in p if len(x) > 0]
        if len(v) >= 2:
            clean.append(v)
    return clean


def top_k_accuracy(y_true, y_pred, k=10):
    correct = 0
    total = 0
    for t, p in zip(y_true, y_pred):
        if len(set(t) & set(p[:k])) > 0:
            correct += 1
        total += 1
    return correct / total


# ===============================
# EVALUATE A SINGLE MODEL
# ===============================
def evaluate_model(model_file, seqs, tree_leaves, tree_anc, num_codes, max_index_tree, device):

    print(f"\n===== Evaluate model: {model_file} =====")

    model = GRAM(
        input_dim=num_codes,
        num_classes=num_codes,
        num_levels=len(tree_leaves),
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_anc,
        max_index_in_tree=max_index_tree,
        device=device,
    ).to(device)

    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    y_true_list = []
    y_pred_list = []
    jaccards = []

    for idx, p in enumerate(seqs):

        if len(p) < 2:
            continue

        x = [p[:-1]]
        y = [p[1:]]

        Xpad, _, mask, _ = pad_batch(x, num_codes, num_codes, device)
        _, Ypad, _, _ = pad_batch(y, num_codes, num_codes, device)

        if Xpad.size(0) == 0:
            continue

        with torch.no_grad():
            pred = model(Xpad, mask)

        pred = pred.squeeze(1)
        Ypad = Ypad.squeeze(1)

        last_pred = pred[-1].cpu().numpy()
        last_true = np.where(Ypad[-1].cpu().numpy() == 1)[0].tolist()

        pred_top = np.argsort(-last_pred).tolist()

        y_true_list.append(last_true)
        y_pred_list.append(pred_top)

        y_bin = np.zeros(num_codes); y_bin[last_true] = 1
        yhat_bin = np.zeros(num_codes); yhat_bin[pred_top[:10]] = 1

        jaccards.append(jaccard_score(y_bin, yhat_bin))

    result = {
        "Top5": top_k_accuracy(y_true_list, y_pred_list, k=5),
        "Top10": top_k_accuracy(y_true_list, y_pred_list, k=10),
        "Jaccard": float(np.mean(jaccards)),
    }

    return result


# ===============================
# MAIN COMPARE
# ===============================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== LOAD DATA =====")

    seqs = pickle.load(open(TEST_FILE, "rb"))
    seqs = clean_seqs(seqs)

    # Compute number of codes
    num_codes = max(max(max(v) for v in p) for p in seqs) + 1

    # Load tree
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)

    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))
    max_index_tree = max(all_idx)

    # Evaluate pretrained synthetic
    res_synth = evaluate_model(
        SYNTH_MODEL, seqs, tree_leaves, tree_anc, num_codes, max_index_tree, device
    )

    # Evaluate fine-tuned real
    res_ft = evaluate_model(
        FINETUNE_MODEL, seqs, tree_leaves, tree_anc, num_codes, max_index_tree, device
    )

    # Print comparison
    print("\n================= COMPARISON RESULT =================")
    print(f"{'Metric':<15} | {'Synthetic Pretrain':<20} | {'Fine-tuned MIMIC':<20}")
    print("-" * 65)
    print(f"{'Top-5 Acc':<15} | {res_synth['Top5']:<20.4f} | {res_ft['Top5']:<20.4f}")
    print(f"{'Top-10 Acc':<15} | {res_synth['Top10']:<20.4f} | {res_ft['Top10']:<20.4f}")
    print(f"{'Jaccard':<15} | {res_synth['Jaccard']:<20.4f} | {res_ft['Jaccard']:<20.4f}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
