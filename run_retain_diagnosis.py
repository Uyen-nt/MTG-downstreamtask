# run_retain_diagnosis.py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from retain_micron.utils import load_and_preprocess_synthetic
from retain_micron.model import RETAIN_Diagnosis
from retain_micron.dataset import EHRDataset, collate_fn
from retain_micron.train import train_model

# run_retain_synthetic_realnext.py
from convert_synthetic_to_realnext import convert_synthetic_to_realnext_format
from retain_micron.utils_mimic3 import load_and_preprocess_mimic3_next

def train_retain():
    synthetic_path = "data/result/synthetic_mimic3.npz"
    output_dir = "data/synthetic_realnext"
    
    convert_synthetic_to_realnext_format(synthetic_path, output_dir)

    seqs, labels, n_codes = load_and_preprocess_mimic3_next(
        train_path=os.path.join(output_dir, "train.npz"),
        test_path=os.path.join(output_dir, "test.npz")
    )
    
    # Bước 3: Train RETAIN như bình thường
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        seqs, labels, test_size=0.1, random_state=42
    )
    
    train_dataset = EHRDataset(train_seqs, train_labels)
    val_dataset = EHRDataset(val_seqs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    model = RETAIN_Diagnosis(n_codes=n_codes, emb_size=256, dropout=0.5)
    
    print("Training RETAIN on CONVERTED synthetic data...")
    train_model(model, train_loader, val_loader, epochs=20)
    
    return model

if __name__ == "__main__":
    model = train_retain()
