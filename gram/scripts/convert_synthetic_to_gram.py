# convert_synthetic_to_gram_with_icd9.py
# ------------------------------------------------------------
# Mục đích:
#   - Đọc synthetic_mimic3.npz (MTGAN output)
#   - Dùng reverse_code_map để chuyển ID → ICD9 thật
#   - Tạo synthetic.seqs (list[list[list[int]]])
#   - Tạo synthetic.types: map ICD9 string → integer
#   - Output tương thích GRAM + build_trees.py hoàn toàn
# ------------------------------------------------------------

import argparse
import numpy as np
import pickle
import sys


def load_reverse_code_map(path):
    """Load reverse ICD9 mapping: index → ICD9 code"""
    with open(path, "rb") as f:
        code_map = pickle.load(f)

    reverse_map = {v: k for k, v in code_map.items()}
    return reverse_map


def convert(npz_path, code_map_path, seq_out, types_out):
    print(f"[Loading synthetic npz] {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    if "x" not in data or "lens" not in data:
        raise ValueError("File .npz phải chứa 'x' và 'lens'.")

    x = data["x"]        # (num_patients, max_visits, num_codes)
    lens = data["lens"]  # (num_patients,)

    num_patients, max_visits, num_codes = x.shape
    print("x shape =", x.shape)
    print("lens shape =", lens.shape)

    # ------------------------------------------------------------
    # Load reverse code map: index -> ICD9 string
    # ------------------------------------------------------------
    print(f"[Loading code_map] {code_map_path}")
    reverse_map = load_reverse_code_map(code_map_path)

    # Kiểm tra một vài điểm
    sample_idx = list(reverse_map.keys())[:10]
    print("Sample ICD9 mapping:")
    for idx in sample_idx:
        print(f"  index {idx} → ICD9 {reverse_map[idx]}")

    # ------------------------------------------------------------
    # Step 1: Decode synthetic tensor thành danh sách ICD9 thật
    # ------------------------------------------------------------
    decoded_seqs_icd9 = []   # list[list[list[str]]]

    for i in range(num_patients):
        patient_visits = []
        for t in range(int(lens[i])):
            visit_codes = np.where(x[i, t] == 1)[0].tolist()
            icd9_codes = [reverse_map[cid] for cid in visit_codes]
            patient_visits.append(icd9_codes)
        decoded_seqs_icd9.append(patient_visits)

    print("\nExample patient (first visits with ICD9):")
    print(decoded_seqs_icd9[0][:2])

    # ------------------------------------------------------------
    # Step 2: Tạo synthetic.types (map ICD9 string → new integer)
    # ------------------------------------------------------------
    print("\n[Building ICD9 → integer types map]")

    types = {}
    next_id = 0

    for patient in decoded_seqs_icd9:
        for visit in patient:
            for icd9 in visit:
                key = "D_" + icd9  # GRAM dùng prefix D_
                if key not in types:
                    types[key] = next_id
                    next_id += 1

    print("Total unique ICD9 codes:", len(types))

    # ------------------------------------------------------------
    # Step 3: Convert ICD9 seqs → integer seqs theo types map
    # ------------------------------------------------------------
    newSeqs = []  # GRAM format: list[list[list[int]]]

    for patient in decoded_seqs_icd9:
        patient_int = []
        for visit in patient:
            int_codes = [types["D_" + icd9] for icd9 in visit]
            patient_int.append(int_codes)
        newSeqs.append(patient_int)

    # ------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------
    print(f"\n[Saving seqs] → {seq_out}")
    with open(seq_out, "wb") as f:
        pickle.dump(newSeqs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Saving types] → {types_out}")
    with open(types_out, "wb") as f:
        pickle.dump(types, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n✅ DONE! Synthetic đã được chuyển thành format GRAM chuẩn có ICD9 thật.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True,
                        help="Đường dẫn synthetic_mimic3.npz (MTGAN output)")
    parser.add_argument("--code_map", required=True,
                        help="Đường dẫn code_map.pkl (ICD9 mapping từ preprocessing)")
    parser.add_argument("--seq_out", required=True,
                        help="Output synthetic.seqs")
    parser.add_argument("--types_out", required=True,
                        help="Output synthetic.types")
    args = parser.parse_args()

    convert(args.npz_path, args.code_map, args.seq_out, args.types_out)
