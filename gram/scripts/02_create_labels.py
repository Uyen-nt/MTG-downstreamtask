# gram/scripts/02_create_labels.py
import pickle
import os
from pathlib import Path

# DÙNG THƯ MỤC HIỆN TẠI TRÊN KAGGLE
WORKING_DIR = Path.cwd()  # /kaggle/working/MTG-downstreamtask/
DATA_DIR = WORKING_DIR / "data"
GRAM_DATA_DIR = WORKING_DIR / "gram" / "data"

REAL_SEQS = DATA_DIR / "result" / "mimic3" / "real_mimic3.3digitICD9.seqs"
SYNTH_SEQS = GRAM_DATA_DIR / "synth_mimic3.seqs"

# Tạo thư mục nếu chưa có
os.makedirs(GRAM_DATA_DIR, exist_ok=True)

def create_labels(seq_file, label_file):
    print(f"Loading sequences from: {seq_file}")
    if not seq_file.exists():
        raise FileNotFoundError(f"File not found: {seq_file}")
    
    seqs = pickle.load(open(seq_file, 'rb'))
    labels = []
    for patient in seqs:
        if len(patient) < 2: 
            continue
        patient_labels = []
        for i in range(len(patient) - 1):
            next_visit = patient[i + 1]
            label = [0] * 5000
            for code in next_visit:
                if code < len(label):
                    label[code] = 1
            patient_labels.append(label)
        if patient_labels:
            labels.append(patient_labels)
    
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f, -1)
    print(f"Saved labels → {label_file}")

create_labels(REAL_SEQS, GRAM_DATA_DIR / "real_mimic3.labels")
create_labels(SYNTH_SEQS, GRAM_DATA_DIR / "synth_mimic3.labels")
