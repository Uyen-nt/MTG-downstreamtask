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

print("🔹 Building patient visit lists...")
for pid in tqdm(patients):
    adms = pat2adm[pid]
    visits = []
    for HADM in adms:
        if HADM in codes:
            visits.append( codes[HADM] )
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
np.savez("real_from_mtg.npz", x=x, lens=lens)

print("🎉 DONE!")

