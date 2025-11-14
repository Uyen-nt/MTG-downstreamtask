# gram/scripts/convert_real_to_gram.py

import pickle
import numpy as np

# ========================
# FIXED PATHS (KAGGLE)
# ========================
PATIENT_ADM_PATH = "data/mimic3/encoded/patient_admission.pkl"
ADMISSION_CODES_PATH = "data/mimic3/parsed/admission_codes.pkl"
CODE_MAP_PATH = "data/mimic3/encoded/code_map.pkl"      
SEQ_OUT = "gram/data/mimic3.seqs"
TYPE_OUT = "gram/data/mimic3.types"


def convert_real_to_gram(patient_admission_path, admission_codes_path,
                         code_map_path, seq_out, type_out):

    print("===== CONVERT REAL → GRAM (shared code_map) =====")

    # ----------------------------------------------------
    # Load REAL data
    # ----------------------------------------------------
    patient_adm = pickle.load(open(patient_admission_path, "rb"))
    adm_codes = pickle.load(open(admission_codes_path, "rb"))
    code_map = pickle.load(open(code_map_path, "rb"))     # ICD9 → index

    print("Loaded:")
    print("  patient_admission =", len(patient_adm))
    print("  admission_codes  =", len(adm_codes))
    print("  code_map size    =", len(code_map))

    # ----------------------------------------------------
    # Build sequences (visits) per patient
    # ----------------------------------------------------
    seqs = []
    for pid, visits in patient_adm.items():
        # visits = list of dicts → sort by time
        visits_sorted = sorted(visits, key=lambda v: v["adm_time"])

        patient_seq = []
        for v in visits_sorted:
            adm_id = v["adm_id"]

            if adm_id not in adm_codes:
                continue

            codes = adm_codes[adm_id]  # ICD9 strings: ["4280", "25000", ...]

            # Convert ICD9 → index via code_map
            visit_idx = []
            for icd in codes:
                if icd in code_map:
                    visit_idx.append(code_map[icd])
                # else: synthetic never used that code → skip

            if len(visit_idx) > 0:
                patient_seq.append(visit_idx)

        if len(patient_seq) >= 2:    # GRAM requires >=2 visits
            seqs.append(patient_seq)

    print("Total valid real patients:", len(seqs))

    # ----------------------------------------------------
    # Save seqs
    # ----------------------------------------------------
    pickle.dump(seqs, open(seq_out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # TYPES: real uses exactly SAME code ids as synthetic → use code_map
    types = {("D_" + k): v for k, v in code_map.items()}
    pickle.dump(types, open(type_out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:")
    print("  →", seq_out)
    print("  →", type_out)
    print("===== DONE =====")


if __name__ == "__main__":
    convert_real_to_gram(PATIENT_ADM_PATH,
                         ADMISSION_CODES_PATH,
                         CODE_MAP_PATH,
                         SEQ_OUT,
                         TYPE_OUT)
