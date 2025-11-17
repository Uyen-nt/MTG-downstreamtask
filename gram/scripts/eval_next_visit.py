# gram/scripts/eval_next_visit.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch


# ==========================
# CONFIG
# ==========================
MODEL_FILE = "gram/data/synth_train_best.pt"          # model pretrained trên synthetic
TREE_PREFIX = "gram/data/mimic3_tree"            # cây thật
TEST_FILE = "gram/data/mimic.seqs"               # seqs thật của mimic3


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
        t = set(t)
        topk = set(p[:k])
        if len(t & topk) > 0:
            correct += 1
        total += 1
    return correct / total


# ==========================
# EVALUATION FUNCTION
# ==========================
def evaluate():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== LOAD TEST DATA =====")

    seqs = pickle.load(open(TEST_FILE, "rb"))
    seqs = clean_seqs(seqs)

    # Load tree
    num_codes = max(max(max(v) for v in p) for p in seqs) + 1
    num_classes = num_codes

    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)

    # compute max_index_in_tree
    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types","rb"))
    all_idx.append(max(types.values()))
    max_index_in_tree = max(all_idx)

    # Load model
    model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_levels=len(tree_leaves),
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_anc,
        max_index_in_tree=max_index_in_tree,
        device=device
    ).to(device)

    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print("Model loaded.")

    # =============================
    # MAKE PREDICTION
    # =============================

    y_true_list = []
    y_pred_list = []
    jaccards = []

    for p in seqs:
        # input = visits 1..(T-1)
        # target = visits 2..T
        x = [p[:-1]]
        y = [p[1:]]

        Xpad, _, mask, _ = pad_batch(x, num_classes, num_codes, device)
        _, Ypad, _, _ = pad_batch(y, num_classes, num_codes, device)

        with torch.no_grad():
            pred = model(Xpad, mask)  # (T,1,num_classes)

        pred = pred.squeeze(1)       # (T, num_classes)
        Ypad = Ypad.squeeze(1)       # (T, num_classes)

        # Evaluate last visit only
        last_pred = pred[-1].cpu().numpy()
        last_true = np.where(Ypad[-1].cpu().numpy() == 1)[0].tolist()

        pred_top = np.argsort(-last_pred).tolist()  # descending

        y_true_list.append(last_true)
        y_pred_list.append(pred_top)

        # Jaccard
        y_binary = np.zeros(num_classes); y_binary[last_true] = 1
        yhat_binary = np.zeros(num_classes); yhat_binary[pred_top[:10]] = 1
        jaccards.append(jaccard_score(y_binary, yhat_binary))

    # =============================
    # PRINT METRICS
    # =============================
    print("===== EVALUATION METRICS =====")
    print("Top-5 Acc :", top_k_accuracy(y_true_list, y_pred_list, k=5))
    print("Top-10 Acc:", top_k_accuracy(y_true_list, y_pred_list, k=10))
    print("Jaccard :", np.mean(jaccards))

    print("Done.")


if __name__ == "__main__":
    evaluate()
