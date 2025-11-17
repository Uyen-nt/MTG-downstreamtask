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
    visits: list các lần khám của bệnh nhân, mỗi lần là list index code
            ví dụ:
              [
                 [4019, 25000, 4280],
                 [4280, 25000]
              ]
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================================================
    # 1. Load mapping MIMIC3
    # =====================================================
    types = pickle.load(open(MIMIC_TYPES, "rb"))
    num_codes = max(types.values()) + 1
    num_classes = num_codes

    idx2code = {v: k for k, v in types.items()}

    # =====================================================
    # 2. Load tree + model
    # =====================================================
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
        device=device
    ).to(device)

    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    # =====================================================
    # 3. Chuẩn bị lịch sử & ground truth
    # =====================================================
    if len(visits) < 2:
        raise ValueError("Cần ít nhất 2 lần khám: 1 lịch sử và 1 ground-truth.")

    history_visits = visits[:-1]
    true_next = visits[-1]

    # -------- FIX: nếu chỉ có 1 visit history → thêm visit rỗng ----------
    if len(history_visits) == 1:
        history_visits = [history_visits[0], []]

    # Pad batch (batch = 1 bệnh nhân)
    x = [history_visits]
    Xpad, _, mask, _ = pad_batch(x, num_classes, num_codes, device)

    # An toàn: nếu vẫn T = 0
    if Xpad.size(0) == 0:
        print("⚠ Cảnh báo: T=0, tạo visit rỗng để chạy model...")
        Xpad = torch.zeros(1, 1, num_codes, device=device)
        mask = torch.ones(1, 1, device=device)

    # =====================================================
    # 4. Predict
    # =====================================================
    with torch.no_grad():
        pred = model(Xpad, mask).squeeze(1)

    last_pred = pred[-1].cpu().numpy()
    top10_idx = np.argsort(-last_pred)[:10].tolist()

    # =====================================================
    # 5. Print kết quả
    # =====================================================
    print("\n========== SINGLE PATIENT PREDICTION ==========")
    print("Device:", device)

    print("\n--- Lịch sử khám ---")
    for t, visit in enumerate(history_visits, start=1):
        icds = [idx2code.get(c, f"<idx:{c}>") for c in visit]
        print(f"Visit {t}: {icds}")

    print("\n--- Ground-truth next visit ---")
    true_icd = [idx2code.get(c, f"<idx:{c}>") for c in true_next]
    print("Mã index:", true_next)
    print("Mã ICD  :", true_icd)

    print("\n--- Top-10 dự đoán ---")
    for rank, idx in enumerate(top10_idx, start=1):
        icd = idx2code.get(idx, f"<idx:{idx}>")
        score = float(last_pred[idx])
        hit = "✓" if idx in true_next else " "
        print(f"{rank:2d}. idx={idx:5d} | ICD={icd:10s} | score={score:.4f} {hit}")

    return {
        "top10_idx": top10_idx,
        "top10_icd": [idx2code.get(i, f"<idx:{i}>") for i in top10_idx],
        "true_idx": true_next,
        "true_icd": true_icd
    }


if __name__ == "__main__":
    patient = [
        [4019, 25000, 4280],
        [4280, 25000]
    ]
    result = predict_next(patient)
    print(result)
