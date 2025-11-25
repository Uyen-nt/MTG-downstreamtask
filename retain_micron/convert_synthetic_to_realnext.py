import numpy as np
import os
from sklearn.model_selection import train_test_split

def convert_synthetic_to_realnext(synthetic_path, output_dir, test_size=0.2):
    """
    Convert synthetic data sang RealNext format CHUẨN như MIMIC-III
    """
    # Load synthetic data
    data = np.load(synthetic_path)
    x_synth = data['x']          # (1500, 34, 2869)
    lens_synth = data['lens']    # (1500,)
    
    print(f"Input: {x_synth.shape[0]} patients, max_visits={x_synth.shape[1]}")
    
    # Tìm max admission number
    max_admission_num = x_synth.shape[1]
    
    # Collect samples theo format MIMIC-III real_next
    x_list, y_list, lens_list = [], [], []
    
    for patient_idx in range(len(x_synth)):
        num_visits = lens_synth[patient_idx]
        
        if num_visits < 2:
            continue  # Skip patients with only 1 visit
            
        # Format: x = visits 0..(L-2), y = visits 1..(L-1)
        # Với L = num_visits, lens = L - 1
        history_length = num_visits - 1
        
        # Input: visits từ 0 đến L-2
        x_patient = x_synth[patient_idx, :history_length]  # (L-1, 2869)
        
        # Labels: visits từ 1 đến L-1 (next visits)
        y_patient = x_synth[patient_idx, 1:num_visits]     # (L-1, 2869)
        
        x_list.append(x_patient)
        y_list.append(y_patient)
        lens_list.append(history_length)
    
    # Pad tất cả sequences về cùng max length
    max_len = max(lens_list)
    
    x_padded = np.zeros((len(x_list), max_len, 2869), dtype=np.float32)
    y_padded = np.zeros((len(y_list), max_len, 2869), dtype=np.float32)
    
    for i in range(len(x_list)):
        actual_len = lens_list[i]
        x_padded[i, :actual_len] = x_list[i]
        y_padded[i, :actual_len] = y_list[i]
    
    lens_array = np.array(lens_list, dtype=np.int64)
    
    print(f"Generated {len(x_padded)} patient samples")
    print(f"Output shapes: x={x_padded.shape}, y={y_padded.shape}, lens={lens_array.shape}")
    
    # Split by patients
    patient_indices = np.arange(len(x_synth))
    train_pids, test_pids = train_test_split(patient_indices, test_size=test_size, random_state=42)
    
    def get_patient_samples(patient_indices):
        x_samples, y_samples, lens_samples = [], [], []
        
        for pid in patient_indices:
            num_visits = lens_synth[pid]
            if num_visits < 2:
                continue
                
            history_length = num_visits - 1
            x_patient = x_synth[pid, :history_length]
            y_patient = x_synth[pid, 1:num_visits]
            
            x_samples.append(x_patient)
            y_samples.append(y_patient)
            lens_samples.append(history_length)
        
        # Pad samples
        max_len_local = max(lens_samples) if lens_samples else 0
        x_padded_local = np.zeros((len(x_samples), max_len_local, 2869), dtype=np.float32)
        y_padded_local = np.zeros((len(y_samples), max_len_local, 2869), dtype=np.float32)
        
        for i in range(len(x_samples)):
            actual_len = lens_samples[i]
            x_padded_local[i, :actual_len] = x_samples[i]
            y_padded_local[i, :actual_len] = y_samples[i]
            
        return x_padded_local, y_padded_local, np.array(lens_samples, dtype=np.int64)
    
    # Split data
    x_train, y_train, lens_train = get_patient_samples(train_pids)
    x_test, y_test, lens_test = get_patient_samples(test_pids)
    
    print(f"Final split: Train={len(x_train)} patients, Test={len(x_test)} patients")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(os.path.join(output_dir, 'train.npz'),
             x=x_train, y=y_train, lens=lens_train)
    
    np.savez(os.path.join(output_dir, 'test.npz'),
             x=x_test, y=y_test, lens=lens_test)
    
    print(f"Saved to {output_dir}")
    
    # Analyze
    analyze_corrected_results(x_train, y_train, lens_train, x_test, y_test, lens_test)
    
    return x_train, y_train, lens_train, x_test, y_test, lens_test

def analyze_corrected_results(x_train, y_train, lens_train, x_test, y_test, lens_test):
    """Phân tích kết quả corrected conversion"""
    print("\n=== CORRECTED CONVERSION ANALYSIS ===")
    print(f"Training set: {len(x_train)} patients")
    print(f"Test set: {len(x_test)} patients")
    print(f"Input shape: {x_train.shape}")
    print(f"Label shape: {y_train.shape}")
    
    # Tính tổng số next-visit predictions
    total_train_predictions = np.sum(lens_train)
    total_test_predictions = np.sum(lens_test)
    
    print(f"Total next-visit predictions - Train: {total_train_predictions}")
    print(f"Total next-visit predictions - Test: {total_test_predictions}")
    
    # Label statistics
    train_labels_binary = (y_train > 0.5)
    test_labels_binary = (y_test > 0.5)
    
    print(f"Label sparsity - Train: {train_labels_binary.mean():.4f}")
    print(f"Avg codes per next visit - Train: {train_labels_binary.sum(axis=2).mean():.2f}")
    
    print("✓ Corrected conversion completed")
