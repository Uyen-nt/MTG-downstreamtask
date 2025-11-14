# gram/scripts/map_synthetic_to_real_tree.py

import pickle

SYNTH_SEQS = "gram/data/synthetic_mimic_format.seqs"
SYNTH_TYPES = "gram/data/synthetic_mimic_format.types"

REAL_TYPES = "gram/data/mimic3_tree.types"

OUT_SEQS = "gram/data/synthetic_remapped_to_real.seqs"
OUT_TYPES = REAL_TYPES   # dùng lại TYPES của REAL TREE


def main():

    print("=== LOAD SYNTHETIC ===")
    synth_seqs = pickle.load(open(SYNTH_SEQS, "rb"))
    synth_types = pickle.load(open(SYNTH_TYPES, "rb"))

    print("=== LOAD MIMIC REAL TREE TYPES ===")
    real_types = pickle.load(open(REAL_TYPES, "rb"))

    # Chuẩn hóa key: bỏ "D_"
    real_map = {}
    for k, v in real_types.items():
        if k.startswith("D_"):
            real_map[k] = v

    print("Real ICD9 count:", len(real_map))

    print("=== REMAP SYNTHETIC → REAL ICD9 ===")

    kept = 0
    lost = 0
    out = []

    for patient in synth_seqs:
        out_p = []
        for visit in patient:
            new_v = []
            for code_id in visit:
                code_str = list(synth_types.keys())[list(synth_types.values()).index(code_id)]
                if code_str in real_map:
                    new_v.append(real_map[code_str])
                    kept += 1
                else:
                    lost += 1
            if new_v:
                out_p.append(new_v)
        if len(out_p) >= 2:
            out.append(out_p)

    print("Kept codes:", kept)
    print("Lost codes:", lost)
    print("Patients kept:", len(out))

    print("=== SAVE ===")
    pickle.dump(out, open(OUT_SEQS, "wb"), pickle.HIGHEST_PROTOCOL)
    print("Saved remapped SEQS →", OUT_SEQS)


if __name__ == "__main__":
    main()
