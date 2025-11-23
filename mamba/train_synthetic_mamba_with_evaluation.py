import torch
import numpy as np
import time
from synthetic_data_loader import create_synthetic_data_loader
from mamba_synthetic import SyntheticEHRMamba, SyntheticMambaTrainer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import os
from datetime import datetime

def evaluate_predictions(predictions, targets, threshold=0.5):
    """
    Evaluate multi-label predictions
    """
    # Convert to binary predictions
    binary_preds = (predictions > threshold).float()
    
    # Flatten for metrics
    preds_flat = binary_preds.reshape(-1).cpu().numpy()
    targets_flat = targets.reshape(-1).cpu().numpy()
    
    # Calculate metrics
    precision = precision_score(targets_flat, preds_flat, zero_division=0)
    recall = recall_score(targets_flat, preds_flat, zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, zero_division=0)
    
    # ROC-AUC (need to handle case when only one class present)
    try:
        roc_auc = roc_auc_score(targets_flat, predictions.reshape(-1).cpu().numpy())
    except:
        roc_auc = 0.0
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'roc_auc': roc_auc
    }

def train_synthetic_mamba_with_evaluation():
    print("🚀 Starting Mamba Training on Synthetic EHR Data...")
    
    # Configuration
    config = {
        'data_path': 'data/result/synthetic_mimic3.npz',  
        'code_map_path': 'data/mimic3/encoded/code_map.pkl',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 50,
        'max_seq_len': 30,
    }
    
    # Load data với code mapping
    print("📥 Loading synthetic data with code mapping...")
    train_loader, code_num, index_to_code = create_synthetic_data_loader(
        config['data_path'], 
        config['code_map_path'],
        batch_size=config['batch_size']
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   Code dimension: {code_num}")
    print(f"   Number of batches: {len(train_loader)}")
    
    # Model configuration
    model_config = {
        'code_num': code_num,
        'd_model': 256,
        'n_layer': 4, 
        'd_state': 64,
        'd_conv': 4,
    }
    
    # Initialize model
    print("🤖 Initializing Mamba model...")
    model = SyntheticEHRMamba(**model_config)
    trainer = SyntheticMambaTrainer(model, learning_rate=config['learning_rate'])
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("🎯 Starting training...")
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (x, lens) in enumerate(train_loader):
            loss = trainer.train_step((x, lens))
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'  Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        
        # Evaluation every 5 epochs
        if epoch % 5 == 0 or epoch == config['epochs'] - 1:
            model.eval()
            val_metrics = {
                'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0
            }
            val_batches = 0
            
            for x, lens in train_loader:
                loss, predictions, targets = trainer.evaluate_batch((x, lens))
                batch_metrics = evaluate_predictions(predictions, targets)
                
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
                val_batches += 1
                
                if val_batches >= 10:  # Evaluate on 10 batches
                    break
            
            # Average metrics
            for key in val_metrics:
                val_metrics[key] /= val_batches
            
            print(f'✅ Epoch {epoch} completed in {epoch_time:.2f}s')
            print(f'   📊 Training Loss: {avg_loss:.4f}')
            print(f'   🎯 Validation Metrics:')
            print(f'      Precision: {val_metrics["precision"]:.4f}')
            print(f'      Recall: {val_metrics["recall"]:.4f}')
            print(f'      F1-Score: {val_metrics["f1"]:.4f}')
            print(f'      ROC-AUC: {val_metrics["roc_auc"]:.4f}')
        else:
            print(f'✅ Epoch {epoch} completed in {epoch_time:.2f}s. Average Loss: {avg_loss:.4f}')
    
    # Save model với metadata
    
    result_dir = 'result'  # thư mục con trong mamba/
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(result_dir, f"synthetic_mamba_{timestamp}.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'code_num': code_num,
        'code_map': index_to_code
    }, model_save_path)
    
    print("="*60)
    print("HOÀN TẤT TRAINING!")
    print(f"Model đã lưu:")
    print(f"   → {model_save_path}")
    print("Vào thư mục mamba/result để xem tất cả các checkpoint!")
    print("="*60)

    return model, train_loader

def demonstrate_predictions(model, train_loader):
    """Demonstrate predictions with actual ICD9 codes"""
    print("\n" + "="*60)
    print("🧪 DEMONSTRATING PREDICTIONS WITH ICD9 CODES")
    print("="*60)
    
    model.eval()
    
    # Get a batch of data
    for x, lens in train_loader:
        # Take first patient
        patient_sequence = x[0]  # (seq_len, code_num)
        actual_seq_len = lens[0]
        
        print(f"\n📋 Patient Sequence (first {actual_seq_len} visits):")
        
        # Decode actual visits
        for visit_idx in range(actual_seq_len):
            visit_codes = []
            for code_idx in range(train_loader.dataset.code_num):
                if patient_sequence[visit_idx, code_idx] == 1:
                    code = train_loader.dataset.index_to_code[code_idx]
                    visit_codes.append(code)
            print(f"   Visit {visit_idx+1}: {', '.join(visit_codes[:5])}{'...' if len(visit_codes) > 5 else ''}")
        
        # Predict next visit
        print(f"\n🔮 PREDICTING NEXT VISIT...")
        
        # Use all but last visit to predict the last one
        input_sequence = patient_sequence[:actual_seq_len-1]  # All but last
        actual_next_visit = patient_sequence[actual_seq_len-1]  # The last one
        
        with torch.no_grad():
            # Add batch dimension
            input_tensor = input_sequence.unsqueeze(0)
            prediction_probs = model(input_tensor)[0, -1]  # Get prediction for next visit
        
        # Get predicted ICD9 codes
        predicted_codes = train_loader.dataset.get_icd9_codes(prediction_probs, threshold=0.3)
        
        # Get actual ICD9 codes for next visit
        actual_codes = []
        for code_idx in range(train_loader.dataset.code_num):
            if actual_next_visit[code_idx] == 1:
                code = train_loader.dataset.index_to_code[code_idx]
                actual_codes.append(code)
        
        print(f"   Actual next visit codes: {', '.join(actual_codes[:8])}{'...' if len(actual_codes) > 8 else ''}")
        print(f"\n   Predicted codes (threshold > 0.3):")
        for code, prob in predicted_codes[:10]:  # Show top 10 predictions
            print(f"      {code}: {prob:.3f}")
        
        # Calculate accuracy for this prediction
        correct_predictions = set([code for code, _ in predicted_codes[:len(actual_codes)]]) & set(actual_codes)
        accuracy = len(correct_predictions) / len(actual_codes) if actual_codes else 0
        
        print(f"\n   📊 Prediction Accuracy: {accuracy:.2%} ({len(correct_predictions)}/{len(actual_codes)} codes correct)")
        
        break  # Only demonstrate for first patient

if __name__ == "__main__":
    # Train model
    trained_model, train_loader = train_synthetic_mamba_with_evaluation()
    
    # Demonstrate predictions with ICD9 codes
    demonstrate_predictions(trained_model, train_loader)
