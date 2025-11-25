# retain_micron/utils.py
import numpy as np

def load_and_preprocess_synthetic(data_path="data/result/synthetic_mimic3.npz", test_size=0.2):
    """Load synthetic data và split ĐÚNG cách (by patients)"""
    data = np.load(data_path)
    x = data['x']  # (1500, 34, 2869)
    lens = data['lens']  # (1500,)
    
    n_patients = len(x)
    
    # Split by PATIENTS, not samples
    patient_ids = np.arange(n_patients)
    train_pids, test_pids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    
    def convert_patients_to_samples(patient_indices):
        sequences = []
        labels = []
        
        for pid in patient_indices:
            L = lens[pid]
            if L < 2: continue
                
            for t in range(1, L):  # Tạo samples từ patient này
                history = []
                for j in range(t):
                    codes = np.where(x[pid, j] > 0.5)[0].tolist()
                    codes = [c for c in codes if c < 2869]
                    if not codes:
                        codes = [2868]  # use last code
                    history.append(codes)
                
                label = np.where(x[pid, t] > 0.5)[0].tolist()
                label = [l for l in label if l < 2869]
                if not label:
                    label = [2868]
                    
                sequences.append(history)
                labels.append(label)
                
        return sequences, labels
    
    # Convert patients to samples
    train_seqs, train_labels = convert_patients_to_samples(train_pids)
    test_seqs, test_labels = convert_patients_to_samples(test_pids)
    
    print(f"Synthetic Data Stats:")
    print(f"  Train: {len(train_seqs)} samples from {len(train_pids)} patients")
    print(f"  Test:  {len(test_seqs)} samples from {len(test_pids)} patients")
    
    return (train_seqs, train_labels), (test_seqs, test_labels), 2869
