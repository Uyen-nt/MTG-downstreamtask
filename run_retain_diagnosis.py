import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np

from retain_micron.utils import load_and_preprocess_synthetic
from retain_micron.model import RETAIN_Diagnosis
from retain_micron.dataset import EHRDataset, collate_fn
from retain_micron.train import train_model

def debug_data_structure(data_path):
    data = np.load(data_path)
    x = data['x']
    lens = data['lens']
    
    print(f"=== DATA STRUCTURE DEBUG ===")
    print(f"x shape: {x.shape}")  # (n_patients, max_visits, n_codes)
    print(f"lens shape: {lens.shape}")
    print(f"Sample patient 0:")
    print(f"  - lens[0]: {lens[0]} (real visits)")
    
    # Kiểm tra patient đầu tiên
    for j in range(min(3, lens[0])):  # 3 visits đầu
        codes = np.where(x[0, j] == 1)[0]
        print(f"  - Visit {j}: {len(codes)} codes - {codes[:5]}...")
    
    # Kiểm tra patient cuối cùng  
    last_pid = len(lens) - 1
    print(f"Sample patient {last_pid}:")
    print(f"  - lens[{last_pid}]: {lens[last_pid]} (real visits)")
    for j in range(min(3, lens[last_pid])):
        codes = np.where(x[last_pid, j] == 1)[0]
        print(f"  - Visit {j}: {len(codes)} codes - {codes[:5]}...")
    
    return x, lens

if __name__ == "__main__":
    seqs, labels, n_codes = load_and_preprocess_synthetic(
        data_path="data/result/synthetic_mimic3.npz"
    )

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        seqs, labels, test_size=0.1, random_state=42, stratify=None
    )


    
    
    # Debug cấu trúc trước
    x, lens = debug_data_structure("data/result/synthetic_mimic3.npz")
    
    # Chỉ xử lý 10 patients đầu để debug
    debug_patient_ids = list(range(min(10, len(lens))))
    print(f"\n=== DEBUG WITH {len(debug_patient_ids)} PATIENTS ===")
    
    debug_sequences = []
    debug_labels = []
    
    for pid in debug_patient_ids:
        L = int(lens[pid])
        if L < 2:
            continue
            
        patient_visits = []
        for j in range(L):
            codes = np.where(x[pid, j] == 1)[0].tolist()
            if not codes:
                codes = [x.shape[-1]]  # padding
            patient_visits.append(codes)
        
        history = patient_visits[:-1]
        label = patient_visits[-1]
        
        debug_sequences.append(history)
        debug_labels.append(label)
        
        print(f"Patient {pid}: {L} visits → {len(history)} history + 1 label")
    
    # Train với dữ liệu debug
    train_dataset = EHRDataset(debug_sequences, debug_labels)
    collate = lambda batch: collate_fn(batch, n_codes=x.shape[-1])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)
    
    # Test 1 batch
    for batch_idx, (visits, labels) in enumerate(train_loader):
        print(f"\n=== TRAINING BATCH {batch_idx} ===")
        model = RETAIN_Diagnosis(n_codes=x.shape[-1], emb_size=256, dropout=0.5)
        output = model(visits)
        print(f"✅ SUCCESS! Output shape: {output.shape}")
        break



    
    # train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    #     seqs, labels, test_size=0.1, random_state=42, stratify=None
    # )

    # train_dataset = EHRDataset(train_seqs, train_labels)
    # val_dataset   = EHRDataset(val_seqs, val_labels)

    # collate = lambda batch: collate_fn(batch, n_codes=n_codes)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    # model = RETAIN_Diagnosis(n_codes=n_codes, emb_size=256, dropout=0.5)

    # train_model(model, train_loader, val_loader, epochs=20, save_path="retain_micron/result")
