import pickle
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

# ============================================================
# 1) Load files
# ============================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# 2) Remap synthetic index → mimic index
# ============================================================

def build_syn_to_mimic_map(syn_types, mimic_types):

    map_syn_to_mimic = {}
    missing_codes = []

    for code, syn_idx in syn_types.items():
        if code in mimic_types:
            map_syn_to_mimic[syn_idx] = mimic_types[code]
        else:
            missing_codes.append(code)

    return map_syn_to_mimic, missing_codes


# ============================================================
# 3) Apply mapping to synthetic.seqs
# ============================================================

def remap_seqs(syn_seqs, syn_to_mimic):
    mapped = []

    for patient in syn_seqs:
        new_patient = []

        for visit in patient:
            new_visit = []
            for c in visit:
                if c in syn_to_mimic:
                    new_visit.append(syn_to_mimic[c])
                # else: code không có trong mimic → bỏ

            if len(new_visit) > 0:
                new_patient.append(new_visit)

        if len(new_patient) >= 2:     # GRAM yêu cầu >=2 visits
            mapped.append(new_patient)

    return mapped


# ============================================================
# 4) Main function
# ============================================================

def main():

    base = "/kaggle/working/MTG-downstreamtask/gram/data"

    syn_types_path = f"{base}/synthetic_converted/synthetic.types"
    syn_seqs_path  = f"{base}/synthetic_converted/synthetic.seqs"

    mimic_types_path = f"{base}/mimic.types"

    out_path = f"{base}/synthetic_converted/synthetic_mapped.seqs"

    print("Loading files...")

    syn_types = load_pickle(syn_types_path)
    syn_seqs  = load_pickle(syn_seqs_path)
    mimic_types = load_pickle(mimic_types_path)

    print(f"synthetic types: {len(syn_types)}")
    print(f"mimic types:     {len(mimic_types)}")

    # ---------------------------------------------------------
    # Build mapping syn_idx → mimic_idx
    # ---------------------------------------------------------
    syn_to_mimic, missing = build_syn_to_mimic_map(syn_types, mimic_types)

    print("\n========== SUMMARY ==========")
    print(f"Total synthetic types  : {len(syn_types)}")
    print(f"Matched types          : {len(syn_to_mimic)}")
    print(f"Missing (not in mimic) : {len(missing)}")

    if len(missing) > 0:
        print("\n⚠️ Các mã synthetic KHÔNG có trong mimic.types:")
        for c in missing[:30]:
            print("  ", c)
        if len(missing) > 30:
            print("  ... (còn nữa) ...")

    # ---------------------------------------------------------
    # Remap seqs
    # ---------------------------------------------------------
    print("\nMapping synthetic.seqs → mimic index...")

    mapped_seqs = remap_seqs(syn_seqs, syn_to_mimic)

    print(f"Original synthetic patients : {len(syn_seqs)}")
    print(f"Mapped patients (>=2 visits): {len(mapped_seqs)}")

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    with open(out_path, "wb") as f:
        pickle.dump(mapped_seqs, f)

    print("\nSaved successfully →", out_path)
    print("Now synthetic_mapped.seqs is READY for GRAM training.")


if __name__ == "__main__":
    main()
