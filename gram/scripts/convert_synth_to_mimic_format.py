# gram/scripts/convert_synth_to_mimic_format.py

import numpy as np
import pickle

SYNTH_NPZ = "data/result/synthetic_mimic3.npz"
CODE_MAP = "data/mimic3/encoded/code_map.pkl"

OUT_SEQS = "gram/data/synthetic_mimic_format.seqs"
OUT_TYPES = "gram/data/synthetic_mimic_format.types"


def to_full_icd9(code):
    """Convert raw MTGAN ICD9 → MIMIC full ICD9 used by GRAM."""
    code = str(code)

    # ---------- E CODES ----------
    if code.startswith("E"):
        if len(code) > 4:
            return code[:4] + "." + code[4:]
        return code

    # ---------- V CODES ----------
    if code.startswith("V"):
        if len(code) > 3:
            return code[:3] + "." + code[3:]
        return code

    # ---------- NUMERIC ----------
    n = len(code)

    if n == 3:
        return code + ".00"
    elif n == 4:
        return code[:3] + "." + code[3] + "0"
    elif n == 5:
        return code[:3] + "." + code[3:]
    else:
        return code


def load_reverse_code_map(path):
    with open(path, "rb") as f:
        code_map = pickle.load(f)
    return {v: k for k, v in code_map.items()}


def main():
    print("=== LOAD SYNTHETIC NPZ ===")
    data = np.load(SYNTH_NPZ, allow_pickle=True)
    x, lens = data["x"], data["lens"]

    print("Shape:", x.shape)

    print("=== LOAD MTGAN code_map ===")
    reverse_map = load_reverse_code_map(CODE_MAP)

    print("=== DECODE SYNTHETIC → FULL ICD9 ===")
    decoded = []
    for i in range(len(x)):
        visits = []
        for t in range(int(lens[i])):
            idxs = np.where(x[i, t] == 1)[0]
            visit = []
            for idx in idxs:
                if idx in reverse_map:
                    raw = reverse_map[idx]
                    full = to_full_icd9(raw)
                    visit.append("D_" + full)
            if visit:
                visits.append(visit)
        if len(visits) >= 2:
            decoded.append(visits)

    print("Patients decoded:", len(decoded))

    print("=== BUILD TYPES ===")
    types = {}
    next_id = 0
    for patient in decoded:
        for visit in patient:
            for code in visit:
                if code not in types:
                    types[code] = next_id
                    next_id += 1

    print("Unique ICD9:", len(types))

    print("=== CONVERT ICD9 → IDs ===")
    seqs_int = []
    for patient in decoded:
        out_p = []
        for visit in patient:
            out_p.append([types[c] for c in visit])
        seqs_int.append(out_p)

    print("=== SAVE ===")
    pickle.dump(seqs_int, open(OUT_SEQS, "wb"), pickle.HIGHEST_PROTOCOL)
    pickle.dump(types, open(OUT_TYPES, "wb"), pickle.HIGHEST_PROTOCOL)

    print("DONE:", OUT_SEQS)


if __name__ == "__main__":
    main()
