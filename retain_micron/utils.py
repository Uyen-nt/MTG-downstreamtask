# retain_micron/utils.py
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_synthetic(data_path, test_size=0.2):
    data = np.load(data_path)
    x = data['x']  # ĐÃ LÀ BINARY (0/1)
    lens = data['lens']
    
    n_codes = x.shape[-1]
    n_patients = len(x)
    
    print(f"Synthetic data - Binary confirmed: {np.unique(x)}")
    
    def patients_to_samples(ids):
        sequences = []
        labels = []
        
        for pid in ids:
            L = int(lens[pid])
            if L < 2:
                continue
                
            # 🔴 XÓA DÒNG NÀY - KHÔNG ĐẢO NGƯỢC!
            # patient_visits = patient_visits[::-1]   # ← XÓA!
            
            patient_visits = []
            for j in range(L):
                codes = np.where(x[pid, j] == 1)[0].tolist()  # Dùng == 1 vì là binary
                if not codes:
                    codes = [n_codes]  # padding
                patient_visits.append(codes)
            
            # ✅ THỨ TỰ ĐÚNG:
            history = patient_visits[:-1]  # visits 0 → L-2
            label = patient_visits[-1]     # visit L-1 (cuối cùng)
            
            # DEBUG: Kiểm tra thứ tự
            if pid < 2:
                print(f"Patient {pid}: {L} visits")
                print(f"  History: visits 0-{L-2}, Label: visit {L-1}")
                print(f"  Label codes: {label[:3]}...")
            
            sequences.append(history)
            labels.append(label)
            
        return sequences, labels
    
    # Split và return như cũ
    patient_ids = np.arange(n_patients)
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    
    train_seqs, train_labels = patients_to_samples(train_ids)
    test_seqs, test_labels = patients_to_samples(test_ids)
    
    print(f"Final: {len(train_seqs)} train, {len(test_seqs)} test samples")
    return (train_seqs, train_labels), (test_seqs, test_labels), n_codes
