# retain_micron/utils.py
import numpy as np

def load_and_preprocess_synthetic(data_path="data/result/synthetic_mimic3.npz"):
    data = np.load(data_path)
    x = data['x']  # (n_patients, max_visits, n_codes)
    lens = data['lens']
    
    n_patients, max_visits, n_codes = x.shape
    print(f"Loaded: {n_patients} patients, {max_visits} max visits, {n_codes} codes")
    
    sequences = []
    labels = []
    
    for i in range(n_patients):
        L = lens[i]  # Số visits thực của patient i
        if L < 2: 
            continue  # Bỏ qua nếu chỉ có 1 visit
            
        # Tạo multiple samples từ mỗi patient
        for t in range(1, L):  # Dự đoán visit t từ visits 0..t-1
            history = []
            for j in range(t):  # Lịch sử từ visit 0 đến t-1
                # QUAN TRỌNG: Synthetic data có thể là probability, cần threshold
                visit_codes = x[i, j]  # Vector n_codes
                
                # Chuyển từ probability → binary codes
                if visit_codes.dtype == np.float32:
                    codes = np.where(visit_codes > 0.5)[0].tolist()
                else:
                    codes = np.where(visit_codes > 0)[0].tolist()
                    
                if not codes:
                    codes = [n_codes]  # padding code
                history.append(codes)
            
            # Label: visit tiếp theo (t)
            next_visit = x[i, t]
            if next_visit.dtype == np.float32:
                label = np.where(next_visit > 0.5)[0].tolist()
            else:
                label = np.where(next_visit > 0)[0].tolist()
                
            if not label:
                label = [n_codes]
                
            sequences.append(history)
            labels.append(label)
    
    print(f"Total sequences for next-visit prediction: {len(sequences)}")
    return sequences, labels, n_codes
