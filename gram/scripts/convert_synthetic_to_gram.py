# convert_synthetic_to_gram.py
# -------------------------------------------------------------------
# Convert synthetic_mimic3.npz (from MTGAN) into GRAM input format:
#   - synthetic.seqs   (list[list[list[int]]])
#   - synthetic.types  ({'D_0':0, 'D_1':1, ...})
#
# Usage:
#   python convert_synthetic_to_gram.py \
#       --npz_path synthetic_mimic3.npz \
#       --seq_out synthetic.seqs \
#       --types_out synthetic.types
# -------------------------------------------------------------------

import argparse
import numpy as np
import pickle


def convert(npz_path, seq_out, types_out):
    print(f"Loading synthetic file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    if "x" not in data or "lens" not in data:
        raise ValueError("File .npz must contain both 'x' and 'lens'.")

    x = data["x"]          # shape (num_patients, max_visits, num_codes)
    lens = data["lens"]    # shape (num_patients,)
    num_patients, max_visits, num_codes = x.shape

    print("x shape:", x.shape)
    print("lens shape:", lens.shape)

    # ----------------------------------------------------------------
    # Convert x + lens → list-of-list-of-list (GRAM format)
    # ----------------------------------------------------------------
    seqs = []
    for i in range(num_patients):
        patient_seq = []
        T_i = int(lens[i])
        for t in range(T_i):
            visit_vector = x[i, t]
            codes = np.where(visit_vector == 1)[0].tolist()
            patient_seq.append(codes)
        seqs.append(patient_seq)

    print("\nExample patient 0:", seqs[0][:min(3, len(seqs[0]))])

    # ----------------------------------------------------------------
    # Build types: {'D_0':0, 'D_1':1, ...}
    # ----------------------------------------------------------------
    types = {f"D_{i}": i for i in range(num_codes)}

    print("Number of codes:", len(types))

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    print(f"Saving seqs → {seq_out}")
    with open(seq_out, "wb") as f:
        pickle.dump(seqs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving types → {types_out}")
    with open(types_out, "wb") as f:
        pickle.dump(types, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n✅ DONE: synthetic.seqs + synthetic.types created!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--seq_out", type=str, default="../data/synthetic.seqs")
    parser.add_argument("--types_out", type=str, default="../data/synthetic.types")
    args = parser.parse_args()

    convert(args.npz_path, args.seq_out, args.types_out)
