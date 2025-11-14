# remap_synth_to_gram_real_tree.py

import numpy as np
import pickle
import os

# ========================
# USER CONFIG
# ========================
SYNTH_NPZ = "/mnt/data/result/synthetic_mimic3.npz"
CODE_MAP = "/mnt/data/mimic3/encoded/code_map.pkl"
TREE_TYPES = "gram/data/mimic3_tree.types"

OUT_SEQS = "gram/data/synthetic_remap.seqs"
OUT_TYPES = "gram/data/synthetic_remap.types"


def load_reverse_code_map(map_path):
    """MTGAN code_map.pkl: ICD9 -> index  => return index -> ICD9"""
    with open(map_path, "rb") as f:
        code_map = pickle.load(f)
    return {v: k for k, v in code_map.items()}


def standardize_icd9(icd):
    """Chuyển raw ICD9 của MTGAN thành dạng ICD9 mà GRAM build_tree dùng."""
    icd = str(icd)

    if icd.startswith("E"):
        return icd[:4] + "." + icd[4:] if len(icd) > 4 else icd
    else:
        return icd[:3] + "." + icd[3:] if len(icd) > 3 else icd


def main():

    print("=== LOAD FILES ===")

    # ---- synthetic npz ----
    data = np.load(SYNTH_NPZ, allow_pickle=True)
    x = data["x"]       # (N, T, C)
    lens = data["lens"] # (N,)
    print("Synthetic:", x.shape, lens.shape)

    # ---- MTGAN mapping ----
    reverse_map = load_reverse_code_map(CODE_MAP)
    print("Loaded MTGAN code_map:", len(reverse_map))

    # ---- GRAM tree mapping ----
    real_types = pickle.load(open(TREE_TYPES, "rb"))
    print("Loaded GRAM real types:", len(real_types))

    # Chuẩn hóa key trong real_types
    real_types_std = {}
    for k, v in real_types.items():
        if k.startswith("D_"):
            raw = k[2:]  # remove D_
            real_types_std[standardize_icd9(raw)] = v

    print("Standardized real ICD9 count:", len(real_types_std))

    # ================
    # REMAP
    # ================

    print("=== DECODE SYNTHETIC → ICD9 ===")

    decoded = []  # list of list of visits of ICD9 strings
    for i in range(len(x)):
        visits = []
        for t in range(int(lens[i])):
            idxs = np.where(x[i, t] == 1)[0]
            icd_list = []
            for idx in idxs:
                if idx in reverse_map:
                    raw = reverse_map[idx]
                    icd = standardize_icd9(raw)
                    icd_list.append(icd)
            visits.append(icd_list)
        decoded.append(visits)

    print("Decoded synthetic patients:", len(decoded))

    # ================================
    # MAP ICD9 SYNTHETIC → REAL TYPES
    # ================================

    print("=== REMAP ICD9 → GRAM TREE IDS ===")

    newSeqs = []
    lost_codes = set()
    kept_codes = set()

    for patient in decoded:
        new_patient = []
        for visit in patient:
            new_visit = []
            for icd in visit:
                if icd in real_types_std:
                    new_visit.append(real_types_std[icd])
                    kept_codes.add(icd)
                else:
                    lost_codes.add(icd)
            if len(new_visit) > 0:
                new_patient.append(new_visit)
        if len(new_patient) >= 2:
            newSeqs.append(new_patient)

    print("Patients kept:", len(newSeqs))
    print("Unique ICD9 kept:", len(kept_codes))
    print("Unique ICD9 lost:", len(lost_codes))

    # Save sequences
    with open(OUT_SEQS, "wb") as f:
        pickle.dump(newSeqs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save types (same as GRAM real)
    with open(OUT_TYPES, "wb") as f:
        pickle.dump(real_types, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n=== DONE ===")
    print("Saved remapped seqs ->", OUT_SEQS)
    print("Saved remapped types ->", OUT_TYPES)
    print("Synthetic is now 100% compatible with GRAM tree.")


if __name__ == "__main__":
    main()
