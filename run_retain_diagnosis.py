import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

from retain_micron.convert_synthetic_to_realnext import convert_synthetic_to_realnext
from retain_micron.utils_mimic3 import load_and_preprocess
from retain_micron.model import RETAIN_Diagnosis
from retain_micron.dataset import EHRDataset, collate_fn
from retain_micron.train import train_model

def train_retain():
    # Bước 1: Convert synthetic data sang RealNext format
    synthetic_path = "data/result/synthetic_mimic3.npz"
    output_dir = "/kaggle/working/MTG-downstreamtask/data/synthetic_realnext"
    
    print("Step 1: Converting synthetic data to RealNext format...")
    convert_synthetic_to_realnext(synthetic_path, output_dir)
    
    # Bước 2: Load converted data với hàm corrected
    print("\nStep 2: Loading converted data...")
    # Sửa: Nhận đúng số giá trị trả về
    train_seqs, train_labels, test_seqs, test_labels, n_codes = load_and_preprocess(
        train_path=os.path.join(output_dir, "train.npz"),
        test_path=os.path.join(output_dir, "test.npz")
    )
    
    print(f"Training samples: {len(train_seqs)}")
    print(f"Test samples: {len(test_seqs)}")
    
    # Bước 3: Split train thành train/val
    print("\nStep 3: Splitting training data into train/val...")
    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        train_seqs, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Final split - Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
    
    # Bước 4: Create datasets và dataloaders
    train_dataset = EHRDataset(train_seqs, train_labels)
    val_dataset = EHRDataset(val_seqs, val_labels)
    test_dataset = EHRDataset(test_seqs, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Bước 5: Initialize và train model
    print("\nStep 4: Initializing model...")
    model = RETAIN_Diagnosis(n_codes=n_codes, emb_size=256, dropout=0.5)
    
    print("Step 5: Training RETAIN on converted synthetic data...")
    train_model(model, train_loader, val_loader, epochs=20)
    
    # Evaluate trên test set
    print("\nStep 6: Evaluating on test set...")
    from retain_micron.evaluate import evaluate_topk_recall
    test_recall_10 = evaluate_topk_recall(model, test_loader, k=10)
    test_recall_20 = evaluate_topk_recall(model, test_loader, k=20)
    print(f"Test Set - Top-10 Recall: {test_recall_10:.4f} | Top-20 Recall: {test_recall_20:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_retain()
