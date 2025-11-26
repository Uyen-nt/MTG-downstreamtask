# retain_micron/utils.py
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_synthetic(data_path, test_size=0.2):
    data = np.load(data_path)
    x = data['x']  # (n_patients, max_visits, n_codes)
    lens = data['lens']
    
    n_codes = x.shape[-1]
    n_patients = len(x)
    
    print(f"Data shape: {x.shape}")
    print(f"Sample patient 0:")
    print(f"  - Number of visits: {lens[0]}")
    print(f"  - Visit 0 codes: {np.where(x[0, 0] == 1)[0]}")
    print(f"  - Visit 1 codes: {np.where(x[0, 1] == 1)[0]}")
    
    def patients_to_samples(ids):
        sequences = []
        labels = []
        
        for pid in ids:
            L = int(lens[pid])
            if L < 2:
                continue
                
            patient_visits = []
            for j in range(L):
                # 🔴 QUAN TRỌNG: Lấy codes trực tiếp, không bọc thêm list
                codes = np.where(x[pid, j] == 1)[0].tolist()
                if not codes:
                    codes = [n_codes]  # padding
                patient_visits.append(codes)  # ĐÃ LÀ list trực tiếp: [code1, code2, ...]
            
            # Kiểm tra cấu trúc
            if pid == 0:
                print(f"Patient 0 - Visit 0: {patient_visits[0][:5]}... (type: {type(patient_visits[0][0])})")
                print(f"Patient 0 - Visit 1: {patient_visits[1][:5]}...")
            
            history = patient_visits[:-1]  # visits 0 → L-2
            label = patient_visits[-1]     # visit L-1
            
            sequences.append(history)
            labels.append(label)
            
        return sequences, labels
    
    patient_ids = np.arange(n_patients)
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    
    train_seqs, train_labels = patients_to_samples(train_ids)
    test_seqs, test_labels = patients_to_samples(test_ids)
    
    # KIỂM TRA CẤU TRÚC CUỐI CÙNG
    print(f"\nFinal structure check:")
    print(f"First training sample: {len(train_seqs[0])} visits")
    print(f"First visit in first sample: {train_seqs[0][0][:3]}...")
    print(f"First label: {train_labels[0][:3]}...")
    
    return (train_seqs, train_labels), (test_seqs, test_labels), n_codes
