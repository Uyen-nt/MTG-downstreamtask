# gram/scripts/predict_single_patient.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np

from gram.model.gram import GRAM, load_tree, pad_batch

MODEL = "gram/data/finetuned.pt"
TREE_PREFIX = "gram/data/mimic3_tree"
MIMIC_TYPES = "gram/data/mimic3_tree.types"


def predict_next(visits):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load mapping của MIMIC3
    types = pickle.load(open(MIMIC_TYPES, "rb"))
    num_codes = max(types.values()) + 1
    num_classes = num_codes

    # Load tree
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)

    # compute max_index
    all_idx=[]
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item()); all_idx.append(A.max().item())
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

    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    # ======================================
    # FIX: Luôn giữ T = số visit - 1 >= 1
    # ======================================
    x = [visits[:-1]]      # input
    y = [visits[-1:]]      # chỉ 1 target visit (không dùng)

    Xpad, _, mask, _ = pad_batch(x, num_classes, num_codes, device)

    # Nếu T == 0 → ép thành 1 dummy visit
    if Xpad.size(0) == 0:
        Xpad = torch.zeros(1, 1, num_codes, device=device)
        mask = torch.ones(1, 1, device=device)

    with torch.no_grad():
        pred = model(Xpad, mask).squeeze(1)

    # lấy output cuối
    top10 = torch.argsort(pred[-1], dim=-1, descending=True)[:10].cpu().tolist()
    return top10



if __name__ == "__main__":
    patient = [
        [4019, 25000, 4280],
        [4280, 25000]
    ]
    print("Top-10 next diagnoses:", predict_next(patient))
