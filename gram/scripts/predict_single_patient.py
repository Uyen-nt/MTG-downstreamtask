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


def convert_icd_list_to_idx(list_icd, types):
    """Convert ICD string list → index list."""
    idx_list = []
    for icd in list_icd:
        if icd in types:
            idx_list.append(types[icd])
        else:
            print(f"[WARNING] ICD {icd} không có trong mapping → bỏ qua")
    return idx_list


def predict_next(visits_icd):
    """
    visits_icd: list các visit, mỗi visit dùng ICD9 string, ví dụ:
        [
           ["4019", "25000", "4280"],
           ["4280", "25000"]
        ]
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------
    # 1. Load mapping
    # ------------------------------------------
    types = pickle.load(open(MIMIC_TYPES, "rb"))
    num_codes = max(types.values()) + 1
    num_classes = num_codes

    idx2code = {v: k for k, v in types.items()}

    # Convert visits ICD → index
    visits = [convert_icd_list_to_idx(v, types) for v in visits_icd]

    if len(visits) < 2:
        raise ValueError("Cần ít nhất 2 visit.")

    # ------------------------------------------
    # 2. Load tree + model
    # ------------------------------------------
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)

    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    all_idx.append(max(types.values()))
    max_index_in_tree = max(all_idx)

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
        device=device,
    ).to(device)

    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    # ------------------------------------------
    # 3. Prepare history
    # ------------------------------------------
    history = visits[:-1]
    true_next = visits[-1]

    if len(history) == 1:
        history.append([])

    Xpad, _, mask, _ = pad_batch([history], num_classes, num_codes, device)

    # ------------------------------------------
    # 4. Predict
    # ------------------------------------------
    with torch.no_grad():
        pred = model(Xpad, mask).squeeze(1)

    last_pred = pred[-1].cpu().numpy()
    top10 = np.argsort(-last_pred)[:10].tolist()

    # ------------------------------------------
    # 5. Print
    # ------------------------------------------
    print("\n========= PREDICTION RESULT =========")
    print("--- HISTORY ---")
    for i, visit in enumerate(visits_icd[:-1], 1):
        print(f"Visit {i}: {visit}")

    print("\n--- TRUE NEXT VISIT ---")
    print("ICD:", visits_icd[-1])

    print("\n--- TOP-10 PREDICTIONS ---")
    for rank, idx in enumerate(top10, 1):
        icd = idx2code.get(idx, f"<idx:{idx}>")
        hit = "✓" if idx in true_next else " "
        print(f"{rank:2d}. ICD={icd:10s} | score={last_pred[idx]:.4f} {hit}")

    return top10


if __name__ == "__main__":
    patient = [
        ["4019", "25000", "4280"],
        ["4280", "25000"]
    ]

    predict_next(patient)
