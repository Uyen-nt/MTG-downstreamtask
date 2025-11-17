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
    visits: list các lần khám của 1 bệnh nhân, mỗi lần là list code index
        vd: [
              [4019, 25000, 4280],   # visit 1
              [4280, 25000]          # visit 2 (ground-truth next visit)
            ]

    Hàm sẽ:
      - dùng các visits[:-1] làm input
      - visits[-1] là ground-truth next visit
      - in ra top-10 mã dự đoán + so sánh với ground-truth
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # 1. Load mapping MIMIC3
    # ------------------------------
    types = pickle.load(open(MIMIC_TYPES, "rb"))   # {icd_str: idx}
    num_codes = max(types.values()) + 1
    num_classes = num_codes

    # Tạo map ngược: idx -> icd_str
    idx2code = {v: k for k, v in types.items()}

    # ------------------------------
    # 2. Load tree + model
    # ------------------------------
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

    # ------------------------------
    # 3. Tách lịch sử & ground-truth
    # ------------------------------
    if len(visits) < 2:
        raise ValueError("Cần ít nhất 2 lần khám: 1 history, 1 ground-truth next visit.")

    history_visits = visits[:-1]   # dùng làm input
    true_next = visits[-1]         # ground-truth next visit (list mã index)

    # Pad batch (batch size = 1 bệnh nhân)
    x = [history_visits]
    Xpad, _, mask, _ = pad_batch(x, num_classes, num_codes, device)

    # Nếu vì lý do nào đó T=0 thì bỏ qua
    if Xpad.size(0) == 0:
        raise RuntimeError("Chuỗi visit sau khi pad có T=0, kiểm tra lại dữ liệu.")

    # ------------------------------
    # 4. Dự đoán
    # ------------------------------
    with torch.no_grad():
        pred = model(Xpad, mask).squeeze(1)   # (T, num_classes)

    last_pred = pred[-1].cpu().numpy()        # (num_classes,)
    top10_idx = np.argsort(-last_pred)[:10].tolist()

    # ------------------------------
    # 5. In kết quả chi tiết
    # ------------------------------
    print("========== SINGLE PATIENT PREDICTION ==========")
    print("Device:", device)
    print("\n--- Lịch sử khám (history visits) ---")
    for t, visit in enumerate(history_visits, start=1):
        codes_str = []
        for c in visit:
            icd = idx2code.get(c, f"<idx:{c}>")
            codes_str.append(icd)
        print(f"Visit {t}: {codes_str}")

    print("\n--- Ground-truth next visit ---")
    true_icd = [idx2code.get(c, f"<idx:{c}>") for c in true_next]
    print("Mã index:", true_next)
    print("Mã ICD  :", true_icd)

    print("\n--- Top-10 dự đoán cho lần khám tiếp theo ---")
    for rank, idx in enumerate(top10_idx, start=1):
        icd = idx2code.get(idx, f"<idx:{idx}>")
        hit = "✓" if idx in true_next else " "
        score = float(last_pred[idx])
        print(f"{rank:2d}. idx={idx:5d} | ICD={icd:10s} | score={score:.4f} {hit}")

    # Có thể trả về để dùng tiếp trong code khác
    return {
        "top10_idx": top10_idx,
        "top10_icd": [idx2code.get(i, f"<idx:{i}>") for i in top10_idx],
        "true_idx": true_next,
        "true_icd": true_icd
    }


if __name__ == "__main__":
    # Ví dụ: patient gồm 2 lần khám,
    # lần 1: history, lần 2: ground-truth next visit
    patient = [
        [4019, 25000, 4280],  # visit 1
        [4280, 25000]         # visit 2 (ground-truth)
    ]
    result = predict_next(patient)
