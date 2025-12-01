import pickle
import numpy as np
from tqdm import tqdm

# ============================
# LOAD EXISTING ENCODED DATA
# ============================

path_base = "data/mimic3/encoded/"

print("🔹 Loading code_map.pkl ...")
code_map = pickle.load(open(path_base + "code_map.pkl", "rb"))  
# ICD → index

print("🔹 Loading codes_encoded.pkl ...")
codes = pickle.load(open(path_base + "codes_encoded.pkl", "rb"))
# HADM_ID → list index codes

print("🔹 Loading patient_admission.pkl ...")
pat2adm = pickle.load(open(path_base + "patient_admission.pkl", "rb"))
# patient → list admission IDs

# ============================
# BUILD x and lens
# ============================

num_icd = len(code_map)  # 2869
patients = list(pat2adm.keys())

lens = []
patient_visits = []

for pid in tqdm(patients):
    adms = pat2adm[pid]
    visits = []
    for adm in adms:

        # Case 1 — integer
        if isinstance(adm, int):
            HADM = adm

        # Case 2 — numpy type
        elif isinstance(adm, (np.int32, np.int64)):
            HADM = int(adm)

        # Case 3 — string number
        elif isinstance(adm, str) and adm.isnumeric():
            HADM = int(adm)

        # Case 4 — dict
        elif isinstance(adm, dict):

            # Case: {'adm_id': 194023}
            if 'adm_id' in adm:
                try:
                    HADM = int(adm['adm_id'])
                except:
                    continue

            # Case: {'194023': ...}
            else:
                key = list(adm.keys())[0]
                try:
                    HADM = int(key)
                except:
                    continue

        # Case 5 — list/tuple
        elif isinstance(adm, (list, tuple)):
            try:
                HADM = int(adm[0])
            except:
                continue

        else:
            continue

        # Add code
        if HADM in codes:
            visits.append(codes[HADM])
        else:
            visits.append([])

    patient_visits.append(visits)
    lens.append(len(visits))



lens = np.array(lens)
max_visits = max(lens)

print("✔ Total patients =", len(patients))
print("✔ Max visits     =", max_visits)
print("✔ ICD dimension  =", num_icd)

# ============================
# CREATE MULTI-HOT 3D TENSOR
# ============================

print("🔹 Building x = (N, max_visits, num_icd) ...")
N = len(patients)
x = np.zeros((N, max_visits, num_icd), dtype=np.float32)

for i, visits in tqdm(enumerate(patient_visits), total=N):
    for v, icd_ids in enumerate(visits):
        for c in icd_ids:
            x[i, v, c] = 1.0

# ============================
# SAVE TO NPZ
# ============================

print("🔹 Saving real_from_mtg.npz ...")
np.savez("data/result/real_from_mtg.npz", x=x, lens=lens)

print("🎉 DONE!")

