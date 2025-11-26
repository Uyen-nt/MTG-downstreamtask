# retain_micron/utils.py
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_synthetic(data_path, test_size=0.2):
    data = np.load(data_path)
    x = data['x']  # (n_patients, max_visits, n_codes)
    lens = data['lens']  # (n_patients,)
    
    n_codes = x.shape[-1]
    n_patients = len(x)
    
    # print(f"=== ORIGINAL DATA ===")
    # print(f"Total patients: {n_patients}")
    # print(f"Max visits per patient: {x.shape[1]}")
    # print(f"Total codes: {n_codes}")
    # print(f"Sample lens: {lens[:10]}...")
    
    def create_correct_samples(patient_ids):
        sequences = []
        labels = []
        error_count = 0
        
        for pid in patient_ids:
            L = int(lens[pid])
            if L < 2:
                error_count += 1
                continue
                
            # TẠO VISITS ĐÚNG CẤU TRÚC
            patient_visits = []
            for j in range(L):
                codes = np.where(x[pid, j] == 1)[0].tolist()
                # 🔴 QUAN TRỌNG: Đảm bảo codes là list trực tiếp, không bọc thêm list
                if isinstance(codes, list) and len(codes) > 0 and isinstance(codes[0], list):
                    print(f"ERROR: Nested list in patient {pid}, visit {j}")
                    codes = [item for sublist in codes for item in sublist]  # Flatten
                
                if not codes:
                    codes = [n_codes]  # padding
                patient_visits.append(codes)
            
            # KIỂM TRA CẤU TRÚC
            # if pid < 3:
            #     print(f"Patient {pid} - {L} visits:")
            #     for j, visit in enumerate(patient_visits):
            #         print(f"  Visit {j}: {len(visit)} codes - {visit[:3]}...")
            
            # TẠO HISTORY VÀ LABEL ĐÚNG
            history = patient_visits[:-1]  # visits 0 → L-2
            label = patient_visits[-1]     # visit L-1
            
            # 🔴 KIỂM TRA LABEL KHÔNG PHẢI LÀ LIST OF LISTS
            # if isinstance(label, list) and len(label) > 0 and isinstance(label[0], list):
            #     print(f"ERROR: Label is list of lists for patient {pid}")
            #     label = [item for sublist in label for item in sublist]  # Flatten
            
            sequences.append(history)
            labels.append(label)
        
        print(f"Errors skipped: {error_count}")
        return sequences, labels
    
    # Split patients
    patient_ids = np.arange(n_patients)
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    
    print(f"\n=== CREATING TRAINING SAMPLES ===")
    train_seqs, train_labels = create_correct_samples(train_ids)
    
    print(f"\n=== CREATING TEST SAMPLES ===")
    test_seqs, test_labels = create_correct_samples(test_ids)
    
    # KIỂM TRA CẤU TRÚC CUỐI CÙNG
    # print(f"\n=== FINAL STRUCTURE CHECK ===")
    # print(f"Training: {len(train_seqs)} samples")
    # print(f"Test: {len(test_seqs)} samples")
    
    # if train_seqs:
    #     print(f"First training sample:")
    #     print(f"  - {len(train_seqs[0])} visits in history")
    #     for j, visit in enumerate(train_seqs[0][:3]):  # 3 visits đầu
    #         print(f"  - Visit {j}: {len(visit)} codes - {visit[:3]}...")
    #     print(f"  - Label: {len(train_labels[0])} codes - {train_labels[0][:5]}...")
    #     print(f"  - Label type: {type(train_labels[0][0]) if train_labels[0] else 'empty'}")
    
    return (train_seqs, train_labels), (test_seqs, test_labels), n_codes
