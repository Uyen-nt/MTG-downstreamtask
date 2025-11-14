# gram/scripts/convert_synth_to_gram.py

import numpy as np
import pickle
import os

SYNTH_NPZ = "data/result/synthetic_mimic3.npz"
CODE_MAP = "data/mimic3/encoded/code_map.pkl"

OUT_SEQS = "gram/data/synthetic_3digit.seqs"
OUT_TYPES = "gram/data/synthetic_3digit.types"


# Convert ICD9 → 3digit ICD9 (chuẩn GRAM & CCS)
def to_3digit(icd):
    icd = str(icd)
    if icd.startswith("E"):
        return icd[:4] + "."   # E000.
    return icd[:3] + "."       # 250. 428.


def load_reverse_code_map(map_path):
    with open(map_path, "rb") as f:
        code_map = pickle.load(f)
    return {v: k for k, v in code_map.items()}


def main():

    print("=== LOAD SYNTHETIC NPZ ===")
    data = np.load(SYNTH_NPZ, allow_pickle=True)
    x = data["x"]
    lens = data["lens"]

    print("shape:", x.shape, lens.shape)

    print("=== LOAD MTGAN code_map ===")
    revmap = load_reverse_code_map(CODE_MAP)

    print("=== DECODE → 3DIGIT ICD9 ===")
    decoded = []
    for i in range(len(x)):
        visits = []
        for t in range(int(lens[i])):
            idxs = np.where(x[i, t] == 1)[0]
            visit = []
            for idx in idxs:
                if idx in revmap:
                    icd_raw = revmap[idx]
                    icd3 = to_3digit(icd_raw)
                    visit.append("D_" + icd3)
            if visit:
                visits.append(visit)
        if len(visits) >= 2:
            decoded.append(visits)

    print("Patients:", len(decoded))

    print("=== BUILD TYPES ===")
    types = {}
    next_id = 0

    for patient in decoded:
        for visit in patient:
            for code in visit:
                if code not in types:
                    types[code] = next_id
                    next_id += 1

    print("Unique codes (3digit):", len(types))

    print("=== CONVERT ICD9 → integer IDs ===")
    seqs_int = []
    for patient in decoded:
        out_patient = []
        for visit in patient:
            out_patient.append([types[c] for c in visit])
        seqs_int.append(out_patient)

    print("=== SAVE ===")
    pickle.dump(seqs_int, open(OUT_SEQS, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(types, open(OUT_TYPES, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("DONE →", OUT_SEQS)
    print("DONE →", OUT_TYPES)


if __name__ == "__main__":
    main()
