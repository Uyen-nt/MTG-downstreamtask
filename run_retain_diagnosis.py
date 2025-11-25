# run_retain_diagnosis.py
from retain_micron.utils import load_and_preprocess_synthetic
from retain_micron.model import RETAIN_Diagnosis
from retain_micron.dataset import EHRDataset, collate_fn
from retain_micron.train import train_model
from sklearn.model_selection import train_test_split

# Load data
seqs, labels, n_codes = load_and_preprocess_synthetic("data/result/synthetic_mimic3.npz")

# Train/test split
train_seqs, test_seqs, train_labels, test_labels = train_test_split(
    seqs, labels, test_size=0.2, random_state=42)

# Dataset & Loader
train_dataset = EHRDataset(train_seqs, train_labels, n_codes)
test_dataset = EHRDataset(test_seqs, test_labels, n_codes)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                           collate_fn=lambda b: collate_fn(b, n_codes))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,
                                          collate_fn=lambda b: collate_fn(b, n_codes))

# Model
model = RETAIN_Diagnosis(n_codes=n_codes, emb_size=256, dropout=0.5)

# Train
train_model(model, train_loader, test_loader, n_codes, epochs=25)
