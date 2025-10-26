# gram/scripts/02_create_labels.py
import pickle
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # /kaggle/working/MTG-downstreamtask/
DATA_DIR = PROJECT_ROOT / "data"
GRAM_DATA_DIR = PROJECT_ROOT / "gram" / "data"

# ĐƯỜNG DẪN CHÍNH XÁC 100%
REAL_SEQS = DATA_DIR / "result" / "mimic3" / "real_mimic3.3digitICD9.seqs"
SYNTH_SEQS = GRAM_DATA_DIR / "synth_mimic3.seqs"

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
    
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f, -1)
    print(f"Saved labels → {label_file}")

create_labels(REAL_SEQS, GRAM_DATA_DIR / "real_mimic3.labels")
create_labels(SYNTH_SEQS, GRAM_DATA_DIR / "synth_mimic3.labels")



