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
    """
    visits = [[code1, code2], [...], ...]
    -> Trả về top10 dự đoán cho next visit
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===========================
    # 1) Load số code thật (2879)
    # ===========================
    types = pickle.load(open(MIMIC_TYPES, "rb"))
    num_codes = max(types.values()) + 1  # = 2879
    num_classes = num_codes

    # ===========================
    # 2) Load tree của MIMIC3
    # ===========================
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)

    # Cần max_index_in_tree y hệt như lúc train
    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item()); all_idx.append(A.max().item())
    all_idx.append(max(types.values()))
    max_index_in_tree = max(all_idx)

    # ===========================
    # 3) Tạo model giống hệt khi train
    # ===========================
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

    # Load weights từ finetuned.pt
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    # ===========================
    # 4) Chuẩn bị input
    # ===========================
    x = [visits[:-1]]
    Xpad, _, mask, _ = pad_batch(x, num_classes, num_codes, device)

    # ===========================
    # 5) Dự đoán
    # ===========================
    with torch.no_grad():
        pred = model(Xpad, mask).squeeze(1)

    top10 = torch.argsort(pred[-1], dim=-1, descending=True)[:10].cpu().tolist()
    return top10


if __name__ == "__main__":
    patient = [
        [4019, 25000, 4280],
        [4280, 25000]
    ]
    print("Top-10 next diagnoses:", predict_next(patient))
