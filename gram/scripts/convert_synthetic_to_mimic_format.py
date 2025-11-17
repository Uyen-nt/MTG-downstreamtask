import numpy as np
import pickle
import os
import argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

# ============================================================
# 1) Convert ICD9 code (string) về D_xxx.xx giống mimic3
#    Sử dụng đúng logic MTGAN → to_standard_icd9
# ============================================================

def to_standard_icd9(code: str):
    code = str(code)
    if code == "":
        return ""
    split_pos = 4 if code.startswith("E") else 3
    if len(code) > split_pos:
        return code[:split_pos] + "." + code[split_pos:]
    return code


def convert_to_mimic_format(code: str):
    """
    Convert:
        '45829'  → 'D_458.29'
        '7455'   → 'D_745.5'
        'V1259'  → 'D_V12.59'
        '311'    → 'D_311'
    """
    std = to_standard_icd9(code)
    return "D_" + std


# ============================================================
# 2) Convert synthetic x → seqs (list[list[list[int]]])
# ============================================================

def convert_x_to_seqs(x, lens):
    """
    Input:
        x    : numpy (N, T, V)
        lens : numpy (N,)
    
    Output:
        seqs : list of patients → visits → list of code index
    """
    N, T, V = x.shape
    seqs = []

    for n in range(N):
        patient_seq = []
        L = lens[n]
        for t in range(L):
            visit_vector = x[n, t]   # shape (V,)
            codes = np.where(visit_vector > 0)[0].tolist()
            patient_seq.append(codes)
        seqs.append(patient_seq)

    return seqs


# ============================================================
# 3) Convert code_map → synthetic.types (string → index)
# ============================================================

def build_synthetic_types(code_map):
    """
    Input:
        code_map : dict {raw_icd9_string : idx}
    
    Output:
        synthetic_types : dict {'D_458.29' : idx}
    """
    synthetic_types = {}

    for code_str, idx in code_map.items():
        new_code = convert_to_mimic_format(code_str)
        synthetic_types[new_code] = idx

    return synthetic_types


# ============================================================
# 4) Main convert function
# ============================================================

def convert_synthetic(npz_path, code_map_path, output_dir):

    print("Loading synthetic npz...")
    data = np.load(npz_path)
    x = data["x"]          # (N, T, V)
    lens = data["lens"]    # (N,)

    print("Loading code_map.pkl...")
    code_map = pickle.load(open(code_map_path, "rb"))

    print("Converting x → seqs ...")
    seqs = convert_x_to_seqs(x, lens)

    print("Converting code_map → synthetic.types ...")
    synthetic_types = build_synthetic_types(code_map)

    print("Building pids ...")
    synthetic_pids = list(range(len(seqs)))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs
    print("Saving results to:", output_dir)
    pickle.dump(synthetic_pids, open(os.path.join(output_dir, "synthetic.pids"), "wb"))
    pickle.dump(seqs, open(os.path.join(output_dir, "synthetic.seqs"), "wb"))
    pickle.dump(synthetic_types, open(os.path.join(output_dir, "synthetic.types"), "wb"))

    print("Done! Synthetic data is now in mimic3 GRAM format.")


# ============================================================
# 5) CLI interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MTGAN synthetic data → mimic3 GRAM format")

    parser.add_argument("--npz", required=True,
                        help="Path to synthetic_mimic3.npz")
    parser.add_argument("--code_map", required=True,
                        help="Path to code_map.pkl")
    parser.add_argument("--out", required=True,
                        help="Output directory to save synthetic.pids / seqs / types")

    args = parser.parse_args()

    convert_synthetic(args.npz, args.code_map, args.out)
