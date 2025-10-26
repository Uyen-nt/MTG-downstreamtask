import pickle
from pathlib import Path

GRAM_DATA_DIR = Path("gram/data")
SEQS = GRAM_DATA_DIR / "tree_mimic3.seqs"
LABELS = GRAM_DATA_DIR / "tree_mimic3.labels"

def create_labels_from_seqs(seqs_path, out_path):
    print(f"Loading {seqs_path}")
    seqs = pickle.load(open(seqs_path, "rb"))
    labels = []
    for patient in seqs:
        if len(patient) < 2:
            continue
        patient_labels = []
        for i in range(len(patient)-1):
            # label = list các mã của lần khám kế tiếp
            next_visit = patient[i+1]
            patient_labels.append(list(next_visit))
        labels.append(patient_labels)
    pickle.dump(labels, open(out_path, "wb"), -1)
    print(f"Saved labels → {out_path}")

if __name__ == "__main__":
    create_labels_from_seqs(SEQS, LABELS)
