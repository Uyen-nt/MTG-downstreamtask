# gram/scripts/convert_synth_to_gram_for_tree.py

import os
import pickle
import numpy as np

# ========================
# CONFIG
# ========================
SYNTH_NPZ = "data/result/synthetic_mimic3.npz"         # output của MTGAN
CODE_MAP = "data/mimic3/encoded/code_map.pkl"          # code_map.pkl của MTGAN
OUT_SEQS = "gram/data/synthetic_tmp.seqs"              # seqs cho GRAM/tree
OUT_TYPES = "gram/data/synthetic_tmp.types"            # types cho GRAM/tree


def standardize_icd9(icd):
    """
    Chuẩn hóa ICD9 về dạng GRAM dùng:
    - Nếu bắt đầu bằng 'E': 4 ký tự + '.' + phần còn lại
    - Ngược lại: 3 ký tự + '.' + phần còn lại
    Ví dụ:
      '4280'  -> '428.0'
      '41071' -> '410.71'
      'V3001' -> 'V30.01'
      'E9317' -> 'E931.7'
    """
    icd = str(icd).strip()
    if icd == "" or icd.lower() == "nan":
        return ""

    if icd.startswith("E"):
        return icd[:4] + "." + icd[4:] if len(icd) > 4 else icd
    else:
        return icd[:3] + "." + icd[3:] if len(icd) > 3 else icd


def load_reverse_code_map(path):
    """
    MTGAN code_map.pkl: concept(string) -> index
    Ta cần: index -> concept(string) (ở đây concept là ICD9 raw / code raw)
    """
    with open(path, "rb") as f:
        code_map = pickle.load(f)
    reverse_map = {v: k for k, v in code_map.items()}
    print(f"[INFO] Loaded code_map with {len(code_map)} concepts.")
    return reverse_map


def main():
    # -------- Load synthetic npz --------
    if not os.path.exists(SYNTH_NPZ):
        raise FileNotFoundError(f"Cannot find synthetic npz at {SYNTH_NPZ}")

    data = np.load(SYNTH_NPZ, allow_pickle=True)
    if "x" not in data or "lens" not in data:
        raise ValueError("synthetic_mimic3.npz must contain 'x' and 'lens' arrays.")

    x = data["x"]       # shape: (N, T, C)
    lens = data["lens"] # shape: (N,)
    print(f"[INFO] Synthetic x shape = {x.shape}, lens shape = {lens.shape}")

    # -------- Load reverse code_map (idx -> ICD9 raw) --------
    reverse_map = load_reverse_code_map(CODE_MAP)

    # -------- Decode và build GRAM-style seqs + types --------
    types = {}          # "D_428.0" -> int ID
    next_id = 0
    seqs = []           # list[list[list[int]]] (patients -> visits -> list of codeIDs)

    n_patients = x.shape[0]
    max_T = x.shape[1]
    C = x.shape[2]

    kept_codes = set()
    lost_codes = set()

    for i in range(n_patients):
        visits = []
        T_i = int(lens[i])
        if T_i <= 0:
            continue
        T_i = min(T_i, max_T)

        for t in range(T_i):
            multi_hot = x[i, t]              # shape: (C,)
            idxs = np.where(multi_hot > 0)[0]
            visit_codes = []

            for idx in idxs:
                if idx not in reverse_map:
                    continue
                raw = str(reverse_map[idx]).strip()
                if raw == "" or raw.lower() == "nan":
                    continue

                icd_std = standardize_icd9(raw)
                if icd_std == "":
                    lost_codes.add(raw)
                    continue

                key = "D_" + icd_std
                kept_codes.add(key)

                if key not in types:
                    types[key] = next_id
                    next_id += 1

                visit_codes.append(types[key])

            if len(visit_codes) > 0:
                visits.append(visit_codes)

        # GRAM chỉ giữ bệnh nhân có ít nhất 2 lượt khám
        if len(visits) >= 2:
            seqs.append(visits)

    print(f"[INFO] #patients in synthetic (after filter len>=2): {len(seqs)}")
    print(f"[INFO] #unique ICD9 kept (GRAM-style D_xxx.xx): {len(kept_codes)}")
    print(f"[INFO] #raw codes lost/empty: {len(lost_codes)}")

    os.makedirs(os.path.dirname(OUT_SEQS), exist_ok=True)

    with open(OUT_SEQS, "wb") as f:
        pickle.dump(seqs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUT_TYPES, "wb") as f:
        pickle.dump(types, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[INFO] Saved GRAM-format synthetic seqs  ->", OUT_SEQS)
    print("[INFO] Saved GRAM-format synthetic types ->", OUT_TYPES)
    print("[DONE] convert_synth_to_gram_for_tree.py")


if __name__ == "__main__":
    main()
