# retain_micron/utils.py
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_synthetic(data_path="data/result/synthetic_mimic3.npz", test_size=0.2):
    data = np.load(data_path)
    x = data['x'].astype(np.float32)  # (N, max_visits, n_codes)
    lens = data['lens']               # (N,)

    n_codes = x.shape[-1]
    n_patients = len(x)

    # Split by patient (rất quan trọng!)
    def bin_length(l):
        if l <= 3:   return 0
        if l <= 5:   return 1
        if l <= 9:   return 2
        return 3  # 10+

    stratify_bins = np.array([bin_length(l) for l in lens])
    patient_ids = np.arange(n_patients)
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42, stratify=stratify_bins)

    def patients_to_samples(ids):
        sequences = []
        labels = []
        for pid in ids:
            L = int(lens[pid])
            if L < 2:  # cần ít nhất 2 visit để có "next"
                continue
    
            # === LẤY TẤT CẢ CÁC VISIT CỦA BỆNH NHÂN ===
            patient_visits = []
            for j in range(L):
                codes = np.where(x[pid, j] > 0.5)[0]        # dùng >0.5 cho chắc (synthetic thường 0/1)
                codes = [int(c) for c in codes]
                if not codes:
                    codes = [n_codes]                       # padding code
                patient_visits.append(codes)
    
            # === ĐẢO NGƯỢC THỨ TỰ ĐỂ ĐÚNG THỜI GIAN THỰC TẾ ===
            # Trong synthetic_mimic3.npz: index 0 = visit cuối cùng (nặng nhất)
            # Sau khi reverse → index 0 = visit đầu tiên (nhẹ nhất)
            patient_visits = patient_visits[::-1]   # ← DÒNG QUAN TRỌNG NHẤT!
    
            # Bây giờ:
            # patient_visits[0]      → visit đầu tiên (quá khứ xa)
            # patient_visits[-1]     → visit cuối cùng (mới nhất, nặng nhất) → label
            # patient_visits[:-1]    → toàn bộ lịch sử để dự đoán visit tiếp theo
    
            history = patient_visits[:-1]      # tất cả visit trừ visit cuối
            label   = patient_visits[-1]       # visit cuối cùng = next visit cần dự đoán
    
            if len(history) == 0:              # trường hợp cực hiếm
                continue
    
            sequences.append(history)
            labels.append(label)
    
        return sequences, labels

    train_seqs, train_labels = patients_to_samples(train_ids)
    test_seqs, test_labels = patients_to_samples(test_ids)

    print(f"Synthetic FIXED - Train: {len(train_seqs)} patients → samples")
    print(f"Synthetic FIXED - Test: {len(test_seqs)} patients → samples")

    return (train_seqs, train_labels), (test_seqs, test_labels), n_codes
