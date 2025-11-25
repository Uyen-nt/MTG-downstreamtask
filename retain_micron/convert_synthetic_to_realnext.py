import numpy as np
import os
from sklearn.model_selection import train_test_split

def convert_synthetic_to_realnext_format(synthetic_path, output_dir, test_size=0.2):
    """
    Convert synthetic patient sequences → RealNext format với padding
    """
    # Load synthetic data
    data = np.load(synthetic_path)
    x_synth = data['x']          # (1500, 34, 2869)
    lens_synth = data['lens']    # (1500,)
    
    print(f"Input: {x_synth.shape[0]} patients, max_visits={x_synth.shape[1]}")
    
    # Tìm max history length cần thiết
    max_history_length = 0
    for patient_idx in range(len(x_synth)):
        num_visits = lens_synth[patient_idx]
        if num_visits < 2:
            continue
        max_history_length = max(max_history_length, num_visits - 1)
    
    print(f"Max history length needed: {max_history_length}")
    
    # Collect all samples với padding
    x_list, y_list, lens_list = [], [], []
    
    for patient_idx in range(len(x_synth)):
        num_visits = lens_synth[patient_idx]
        
        if num_visits < 2:
            continue  # Skip patients with only 1 visit
            
        # Tạo multiple training samples từ mỗi patient
        for prediction_point in range(1, num_visits):
            # History: visits 0 to (prediction_point-1)
            history_length = prediction_point
            history_visits = x_synth[patient_idx, :prediction_point]  # (history_length, 2869)
            
            # Pad history to max_history_length
            padded_history = np.zeros((max_history_length, 2869))
            padded_history[:history_length] = history_visits
            
            # Label: visit at prediction_point
            next_visit = x_synth[patient_idx, prediction_point]
            
            x_list.append(padded_history)
            y_list.append(next_visit)
            lens_list.append(history_length)  # Actual history length before padding
    
    # Convert to arrays
    x_array = np.array(x_list)    # (samples, max_history_length, 2869)
    y_array = np.array(y_list)    # (samples, 2869)
    lens_array = np.array(lens_list)  # (samples,)
    
    print(f"Generated {len(x_array)} next-visit samples")
    print(f"Output shapes: x={x_array.shape}, y={y_array.shape}, lens={lens_array.shape}")
    
    # Split by patients để tránh data leakage
    patient_indices = np.arange(len(x_synth))
    train_pids, test_pids = train_test_split(patient_indices, test_size=test_size, random_state=42)
    
    def get_samples_from_patients(patient_indices):
        """Lấy tất cả samples từ danh sách patients với padding"""
        x_samples, y_samples, lens_samples = [], [], []
        
        for pid in patient_indices:
            num_visits = lens_synth[pid]
            if num_visits < 2:
                continue
                
            for t in range(1, num_visits):
                history_length = t
                history = x_synth[pid, :t]  # (t, 2869)
                
                # Pad history
                padded_history = np.zeros((max_history_length, 2869))
                padded_history[:history_length] = history
                
                next_visit = x_synth[pid, t]
                
                x_samples.append(padded_history)
                y_samples.append(next_visit)
                lens_samples.append(history_length)
                
        return np.array(x_samples), np.array(y_samples), np.array(lens_samples)
    
    # Split data
    x_train, y_train, lens_train = get_samples_from_patients(train_pids)
    x_test, y_test, lens_test = get_samples_from_patients(test_pids)
    
    print(f"Final split: Train={len(x_train)}, Test={len(x_test)}")
    print(f"Train shapes: x={x_train.shape}, y={y_train.shape}, lens={lens_train.shape}")
    
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
    
    # Analyze results
    analyze_conversion_results(x_train, y_train, lens_train, x_test, y_test, lens_test)
    
    return x_train, y_train, lens_train, x_test, y_test, lens_test

def analyze_conversion_results(x_train, y_train, lens_train, x_test, y_test, lens_test):
    """Phân tích kết quả conversion"""
    print("\n=== CONVERSION ANALYSIS ===")
    print(f"Training set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")
    print(f"Input shape: {x_train.shape}")
    print(f"Label shape: {y_train.shape}")
    
    # Label statistics
    train_labels_binary = (y_train > 0.5)
    test_labels_binary = (y_test > 0.5)
    
    print(f"Train label sparsity: {train_labels_binary.mean():.4f}")
    print(f"Test label sparsity: {test_labels_binary.mean():.4f}")
    print(f"Avg codes per label - Train: {train_labels_binary.sum(axis=1).mean():.2f}")
    print(f"Avg codes per label - Test: {test_labels_binary.sum(axis=1).mean():.2f}")
    
    # History length distribution
    print(f"History length - Min: {lens_train.min()}, Max: {lens_train.max()}, Mean: {lens_train.mean():.2f}")
    
    # Check data validity
    assert not np.isnan(x_train).any(), "NaN values in training data"
    assert not np.isnan(y_train).any(), "NaN values in training labels"
    assert not np.isnan(x_test).any(), "NaN values in test data"
    assert not np.isnan(y_test).any(), "NaN values in test labels"
    
    print("✓ Data validation passed - no NaN values")
