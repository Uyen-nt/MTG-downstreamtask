# convert_synth_to_gram_3digit.py
# ==========================================
# Convert synthetic MTGAN → 3-digit ICD9 → GRAM tree
# ==========================================

import numpy as np
import pickle
import os

# ======================
# CONFIG
# ======================
SYNTH_NPZ = "data/result/synthetic_mimic3.npz"
CODE_MAP = "data/mimic3/encoded/code_map.pkl"
GRAM_TYPES = "gram/data/mimic3_tree.types"   # cây thật
OUT_SEQS = "gram/data/synthetic_3digit.seqs"
OUT_TYPES = "gram/data/synthetic_3digit.types"


# ========== Helper ==========
def load_reverse_map(path):
    """MTGAN code_map.pkl  (ICD9 → index) → reverse (index → ICD9)."""
    with open(path, "rb") as f:
        code_map = pickle.load(f)
    return {v: k for k, v in code_map.items()}


def icd9_to_3digit(icd):
    """Chuẩn hóa ICD9 về dạng 3-digit y như GRAM process_mimic.py."""
    icd = icd.replace(".", "")  # bỏ dấu chấm, để tách dễ hơn

    if icd.startswith("E"):
        # E + 3 số đầu
        return icd[:4]    # ví dụ E9382 → E938
    elif icd.startswith("V"):
        # V + 2 số đầu
        return icd[:3]    # ví dụ V5417 → V54
    else:
        # số thường: 3 số đầu
        return icd[:3]    # 78061 → 780


def main():
    print("=== LOAD FILES ===")

    # load synthetic
    data = np.load(SYNTH_NPZ, allow_pickle=True)
    x, lens = data["x"], data["lens"]
    print("Synthetic x:", x.shape, "lens:", lens.shape)

    # reverse MTGAN code index → ICD9
    reverse_map = load_reverse_map(CODE_MAP)
    print("reverse_map size:", len(reverse_map))

    # load GRAM real tree mapping
    real_types = pickle.load(open(GRAM_TYPES, "rb"))
    print("Real GRAM ICD9 count:", len(real_types))

    # chuẩn hóa GRAM keys: D_584 → "584"
    gram_3digit_map = {}
    for key, val in real_types.items():
        if not key.startswith("D_"): 
            continue
        raw = key[2:]            # D_584 → 584
        gram_3digit_map[raw] = val

    print("GRAM normalized ICD9:", len(gram_3digit_map))


    # ==========================================
    # Decode synthetic → ICD9 → convert 3-digit
    # ==========================================
    decoded = []
    print("\n=== Decode synthetic ICD9 to 3-digit ===")
    for i in range(len(x)):
        visits = []
        for t in range(int(lens[i])):
            idxs = np.where(x[i, t] == 1)[0]
            codes = []
            for ci in idxs:
                if ci in reverse_map:
                    raw = reverse_map[ci]   # ví dụ "78061"
                    d3 = icd9_to_3digit(raw)
                    codes.append(d3)
            visits.append(codes)
        decoded.append(visits)

    print("Decoded patients:", len(decoded))

    # ==========================================
    # Mapping synthetic 3-digit → GRAM tree id
    # ==========================================
    print("\n=== Mapping to GRAM tree ===")

    newSeqs = []
    lost_codes = set()
    kept_codes = set()

    for patient in decoded:
        new_p = []
        for visit in patient:
            mapped_v = []
            for c in visit:
                if c in gram_3digit_map:
                    mapped_v.append(gram_3digit_map[c])
                    kept_codes.add(c)
                else:
                    lost_codes.add(c)
            if len(mapped_v) > 0:
                new_p.append(mapped_v)
        if len(new_p) >= 2:
            newSeqs.append(new_p)

    print("Kept ICD9 unique:", len(kept_codes))
    print("Lost ICD9 unique:", len(lost_codes))
    print("Final patients:", len(newSeqs))


    # Save seqs
    with open(OUT_SEQS, "wb") as f:
        pickle.dump(newSeqs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save types — identical to GRAM real types
    with open(OUT_TYPES, "wb") as f:
        pickle.dump(real_types, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n=== DONE ===")
    print("Saved:", OUT_SEQS)
    print("Saved:", OUT_TYPES)


if __name__ == "__main__":
    main()
