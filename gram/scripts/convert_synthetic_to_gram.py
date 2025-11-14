# gram/scripts/convert_synthetic_to_gram.py

import numpy as np
import pickle

# ========================
# FIXED PATHS (KAGGLE)
# ========================
NPZ_PATH = "data/result/synthetic_mimic3.npz"
CODE_MAP_PATH = "data/mimic3/encoded/code_map.pkl"
SEQ_OUT = "gram/data/synthetic.seqs"
TYPES_OUT = "gram/data/synthetic.types"


def load_reverse_code_map(path):
    """Load reverse ICD9 mapping: index → ICD9 code"""
    with open(path, "rb") as f:
        code_map = pickle.load(f)
    return {v: k for k, v in code_map.items()}


def convert(npz_path, code_map_path, seq_out, types_out):
    print(f"[Loading synthetic npz] {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    if "x" not in data or "lens" not in data:
        raise ValueError("File .npz phải chứa 'x' và 'lens'.")

    x = data["x"]
    lens = data["lens"]

    print(f"x shape = {x.shape}, lens shape = {lens.shape}")

    # Load ICD9 mapping
    print(f"[Loading code_map] {code_map_path}")
    reverse_map = load_reverse_code_map(code_map_path)

    # Giải mã synthetic về ICD9
    decoded = []
    for i in range(len(x)):
        patient_visits = []
        for t in range(int(lens[i])):
            idxs = np.where(x[i, t] == 1)[0].tolist()
            icd9 = [reverse_map[j] for j in idxs]
            patient_visits.append(icd9)
        decoded.append(patient_visits)

    # Tạo types: ICD9 → ID mới
    types = {}
    next_id = 0
    for patient in decoded:
        for visit in patient:
            for icd9 in visit:
                key = "D_" + icd9
                if key not in types:
                    types[key] = next_id
                    next_id += 1

    # Convert ICD9 → integer
    newSeqs = []
    for patient in decoded:
        out_patient = []
        for visit in patient:
            out_codes = [types["D_" + c] for c in visit]
            out_patient.append(out_codes)
        newSeqs.append(out_patient)

    # Save
    with open(seq_out, "wb") as f:
        pickle.dump(newSeqs, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(types_out, "wb") as f:
        pickle.dump(types, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ convert_synthetic_to_gram DONE!")


if __name__ == "__main__":
    convert(NPZ_PATH, CODE_MAP_PATH, SEQ_OUT, TYPES_OUT)
