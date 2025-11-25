import numpy as np
import os
from sklearn.model_selection import train_test_split

def convert_synthetic_to_realnext_format(synthetic_path, output_dir, test_size=0.2):
    """
    Convert synthetic patient sequences → RealNext format cho next-visit prediction
    """
    # Load synthetic data
    data = np.load(synthetic_path)
    x_synth = data['x']          # (1500, 34, 2869)
    lens_synth = data['lens']    # (1500,)
    
    print(f"Input: {x_synth.shape[0]} patients, max_visits={x_synth.shape[1]}")
    
    # Collect all samples
    x_list, y_list, lens_list = [], [], []
    
    for patient_idx in range(len(x_synth)):
        num_visits = lens_synth[patient_idx]
        
        if num_visits < 2:
            continue  # Skip patients with only 1 visit
            
        # Tạo multiple training samples từ mỗi patient
        for prediction_point in range(1, num_visits):
            # History: visits 0 to (prediction_point-1)
            history_visits = x_synth[patient_idx, :prediction_point]
            
            # Label: visit at prediction_point
            next_visit = x_synth[patient_idx, prediction_point]
            
            x_list.append(history_visits)
            y_list.append(next_visit)
            lens_list.append(prediction_point)  # Length of history
    
    # Convert to arrays
    x_array = np.array(x_list)    # (samples, max_visits, 2869)
    y_array = np.array(y_list)    # (samples, 2869)
    lens_array = np.array(lens_list)  # (samples,)
    
    print(f"Generated {len(x_array)} next-visit samples")
    
    # Split by patients để tránh data leakage
    patient_indices = np.arange(len(x_synth))
    train_pids, test_pids = train_test_split(patient_indices, test_size=test_size, random_state=42)
    
    def get_samples_from_patients(patient_indices):
        """Lấy tất cả samples từ danh sách patients"""
        x_samples, y_samples, lens_samples = [], [], []
        
        for pid in patient_indices:
            num_visits = lens_synth[pid]
            if num_visits < 2:
                continue
                
            for t in range(1, num_visits):
                history = x_synth[pid, :t]
                next_v = x_synth[pid, t]
                
                x_samples.append(history)
                y_samples.append(next_v)
                lens_samples.append(t)
                
        return np.array(x_samples), np.array(y_samples), np.array(lens_samples)
    
    # Split data
    x_train, y_train, lens_train = get_samples_from_patients(train_pids)
    x_test, y_test, lens_test = get_samples_from_patients(test_pids)
    
    print(f"Final split: Train={len(x_train)}, Test={len(x_test)}")
    
    # Save in RealNext format
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(os.path.join(output_dir, 'train.npz'),
             x=x_train.astype(np.float32),
             y=y_train.astype(np.float32), 
             lens=lens_train.astype(np.int64))
    
    np.savez(os.path.join(output_dir, 'test.npz'),
             x=x_test.astype(np.float32),
             y=y_test.astype(np.float32),
             lens=lens_test.astype(np.int64))
    
    print(f"Saved RealNext format to {output_dir}")
    return x_train, y_train, lens_train, x_test, y_test, lens_test

def analyze_converted_data(original_path, converted_dir):
    """Phân tích chất lượng converted data"""
    # Load original synthetic
    orig_data = np.load(original_path)
    x_orig, lens_orig = orig_data['x'], orig_data['lens']
    
    # Load converted
    train_data = np.load(os.path.join(converted_dir, 'train.npz'))
    x_train, y_train, lens_train = train_data['x'], train_data['y'], train_data['lens']
    
    print("=== CONVERSION ANALYSIS ===")
    print(f"Original: {x_orig.shape[0]} patients, {np.sum(lens_orig)} total visits")
    print(f"Converted: {x_train.shape[0]} training samples")
    print(f"Avg history length: {np.mean(lens_train):.2f}")
    
    # Analyze label sparsity
    label_sparsity = (y_train > 0.5).mean()
    print(f"Label sparsity: {label_sparsity:.4f}")
    
    # Analyze temporal patterns
    avg_codes_per_visit = [(x_train[i, :lens_train[i]] > 0.5).sum() / lens_train[i] 
                          for i in range(len(x_train))]
    print(f"Avg codes per visit: {np.mean(avg_codes_per_visit):.2f}")
    
    return True
