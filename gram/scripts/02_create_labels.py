# scripts/02_create_labels.py
import pickle
import os

DATA_DIR = "data"
REAL_SEQS = "../../data/result/mimic3/real_mimic3.3digitICD9.seqs"
/data/result/mimic3/real_mimic3.3digitICD9.seqs
SYNTH_SEQS = f"{DATA_DIR}/synth_mimic3.seqs"

os.makedirs(DATA_DIR, exist_ok=True)

def create_labels(seq_file, label_file):
    seqs = pickle.load(open(seq_file, 'rb'))
    labels = []
    for patient in seqs:
        if len(patient) < 2: continue
        patient_labels = []
        for i in range(len(patient) - 1):
            next_visit = patient[i + 1]
            # Tạo nhãn: đa nhãn (multi-label)
            label = [0] * 5000  # đủ lớn
            for code in next_visit:
                if code < len(label):
                    label[code] = 1
            patient_labels.append(label[:len(next_visit)])  # cắt cho gọn
        if patient_labels:
            labels.append(patient_labels)
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f, -1)
    print(f"Saved labels → {label_file}")

create_labels(REAL_SEQS, f"{DATA_DIR}/real_mimic3.labels")
create_labels(SYNTH_SEQS, f"{DATA_DIR}/synth_mimic3.labels")
