# gram/scripts/eval_compare.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch
from gram.scripts.train_synth import evaluate_model, calculate_metrics, top_k_accuracy

# ===============================
# CONFIG
# ===============================
TREE_PREFIX = "gram/data/mimic3_tree"
TEST_FILE = "gram/data/mimic_test.seqs"  # Use the saved test set

SYNTH_MODEL = "gram/data/synth_train_best.pt"
FINETUNE_MODEL = "gram/data/finetuned_best.pt"

# ===============================
# COMPREHENSIVE EVALUATION
# ===============================

def load_test_data():
    """Load test data for evaluation"""
    try:
        test_seqs = pickle.load(open(TEST_FILE, "rb"))
        print(f"Loaded test set: {len(test_seqs)} patients")
        return test_seqs
    except:
        print("Test set not found, using full dataset...")
        full_seqs = pickle.load(open("gram/data/mimic.seqs", "rb"))
        # Clean and take subset for testing
        clean_seqs = []
        for p in full_seqs:
            v = [x for x in p if len(x) > 0]
            if len(v) >= 2:
                clean_seqs.append(v)
        return clean_seqs[:1000]  # Use subset for faster evaluation

def get_original_num_codes(model_path):
    """Lấy số lượng codes gốc từ checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Tìm output layer weight để xác định num_codes gốc
        for key in checkpoint.keys():
            if 'out.weight' in key:
                original_num_codes = checkpoint[key].shape[0]
                print(f"Original num_codes in {model_path}: {original_num_codes}")
                return original_num_codes
    except Exception as e:
        print(f"Error getting original num_codes: {e}")
    return None

def create_compatible_model(model_path, current_num_codes, tree_leaves, tree_anc, max_index_tree, device):
    """Tạo model tương thích với số lượng codes hiện tại"""
    try:
        # Lấy số lượng codes gốc từ checkpoint
        original_num_codes = get_original_num_codes(model_path)
        if original_num_codes is None:
            return None
            
        # Tạo model với số lượng codes gốc
        model = GRAM(
            input_dim=original_num_codes,
            num_classes=original_num_codes,
            num_levels=len(tree_leaves),
            emb_dim=128,
            att_dim=128,
            hidden_dim=128,
            tree_leaves=tree_leaves,
            tree_ancestors=tree_anc,
            max_index_in_tree=max_index_tree,
            device=device,
        ).to(device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        # Nếu số lượng codes khác nhau, cần điều chỉnh output layer
        if original_num_codes != current_num_codes:
            print(f"⚠️  Adjusting model: {original_num_codes} → {current_num_codes} codes")
            
            # Tạo output layer mới với kích thước phù hợp
            new_out_weight = torch.zeros(current_num_codes, 128, device=device)
            new_out_bias = torch.zeros(current_num_codes, device=device)
            
            # Copy weights từ checkpoint (chỉ copy những codes có sẵn)
            min_codes = min(original_num_codes, current_num_codes)
            new_out_weight[:min_codes] = checkpoint['out.weight'][:min_codes]
            new_out_bias[:min_codes] = checkpoint['out.bias'][:min_codes]
            
            # Cập nhật checkpoint
            checkpoint['out.weight'] = new_out_weight
            checkpoint['out.bias'] = new_out_bias
            
            print(f"✅ Adjusted output layer: {min_codes} codes preserved")
        
        # Load state dict đã điều chỉnh
        model.load_state_dict(checkpoint)
        return model
        
    except Exception as e:
        print(f"Error creating compatible model: {e}")
        return None

def evaluate_all_models():
    """Evaluate both pretrained and fine-tuned models"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== COMPREHENSIVE MODEL EVALUATION =====")
    
    # Load test data
    test_seqs = load_test_data()
    
    # Compute num_codes từ test data
    current_num_codes = max(max(max(v) for v in p) for p in test_seqs) + 1
    
    # Load tree với số lượng codes lớn hơn để đảm bảo coverage
    max_num_codes = max(current_num_codes, 2879)  # Dùng số lớn nhất
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, max_num_codes, device)
    
    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))
    max_index_tree = max(all_idx)
    
    print(f"Current num_codes: {current_num_codes}")
    print(f"Test patients: {len(test_seqs)}")
    
    # Evaluate both models
    results = {}
    
    for model_name, model_path in [("Synthetic Pretrained", SYNTH_MODEL), 
                                   ("Fine-tuned MIMIC", FINETUNE_MODEL)]:
        
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        # Tạo model tương thích
        model = create_compatible_model(model_path, current_num_codes, tree_leaves, tree_anc, max_index_tree, device)
        
        if model is None:
            print(f"❌ Failed to load {model_name}")
            continue
        
        print(f"✅ Model loaded successfully")
        
        # Evaluate - sử dụng current_num_codes cho evaluation
        metrics = evaluate_model(model, test_seqs, current_num_codes, current_num_codes, device)
        results[model_name] = metrics
        
        # Print results
        print(f"Loss:          {metrics['loss']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")
        print(f"Top-10 Accuracy: {metrics['top10_acc']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1-Score:      {metrics['f1']:.4f}")
        print(f"Jaccard:       {metrics['jaccard']:.4f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    
    if len(results) == 2:
        synth_metrics = results["Synthetic Pretrained"]
        ft_metrics = results["Fine-tuned MIMIC"]
        
        print(f"{'Metric':<15} | {'Synthetic Pretrain':<20} | {'Fine-tuned MIMIC':<20} | {'Improvement':<15}")
        print("-" * 85)
        
        metrics_list = [
            ('Top-5 Acc', synth_metrics['top5_acc'], ft_metrics['top5_acc']),
            ('Top-10 Acc', synth_metrics['top10_acc'], ft_metrics['top10_acc']),
            ('Precision', synth_metrics['precision'], ft_metrics['precision']),
            ('Recall', synth_metrics['recall'], ft_metrics['recall']),
            ('F1-Score', synth_metrics['f1'], ft_metrics['f1']),
            ('Jaccard', synth_metrics['jaccard'], ft_metrics['jaccard']),
            ('Loss', synth_metrics['loss'], ft_metrics['loss']),
        ]
        
        for name, synth_val, ft_val in metrics_list:
            if name == 'Loss':
                improvement = synth_val - ft_val  # Lower loss is better
            else:
                improvement = ft_val - synth_val  # Higher is better
            
            print(f"{name:<15} | {synth_val:<20.4f} | {ft_val:<20.4f} | {improvement:>14.4f}")
        
        # Calculate overall improvement
        key_metrics = ['top5_acc', 'top10_acc', 'f1']
        avg_improvement = np.mean([ft_metrics[m] - synth_metrics[m] for m in key_metrics])
        print(f"\n📈 Average improvement on key metrics: {avg_improvement:.4f}")
    
    elif len(results) == 1:
        model_name = list(results.keys())[0]
        print(f"Only {model_name} available for evaluation")
    
    else:
        print("No models available for evaluation")
    
    print("="*80)

# Alternative evaluation method cho trường hợp vẫn lỗi
def simple_evaluation():
    """Phương pháp đánh giá đơn giản hơn"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== SIMPLE EVALUATION =====")
    
    # Load test data
    test_seqs = load_test_data()
    current_num_codes = max(max(max(v) for v in p) for p in test_seqs) + 1
    
    print(f"Current num_codes: {current_num_codes}")
    
    for model_name, model_path in [("Synthetic Pretrained", SYNTH_MODEL), 
                                   ("Fine-tuned MIMIC", FINETUNE_MODEL)]:
        
        print(f"\nEvaluating: {model_name}")
        
        try:
            # Thử load trực tiếp
            checkpoint = torch.load(model_path, map_location=device)
            original_num_codes = checkpoint['out.weight'].shape[0]
            
            print(f"Original: {original_num_codes}, Current: {current_num_codes}")
            
            if original_num_codes != current_num_codes:
                print(f"⚠️  Size mismatch. Creating compatible model...")
                
                # Tạo model với original size
                tree_leaves, tree_anc = load_tree(TREE_PREFIX, original_num_codes, device)
                all_idx = []
                for L, A in zip(tree_leaves, tree_anc):
                    all_idx.append(L.max().item())
                    all_idx.append(A.max().item())
                types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
                all_idx.append(max(types.values()))
                max_index_tree = max(all_idx)
                
                model = GRAM(
                    input_dim=original_num_codes,
                    num_classes=original_num_codes,
                    num_levels=len(tree_leaves),
                    emb_dim=128,
                    att_dim=128,
                    hidden_dim=128,
                    tree_leaves=tree_leaves,
                    tree_ancestors=tree_anc,
                    max_index_in_tree=max_index_tree,
                    device=device,
                ).to(device)
                
                model.load_state_dict(checkpoint)
                print(f"✅ Model loaded with original size {original_num_codes}")
                
                # Đánh giá với test data đã filter
                filtered_test_seqs = []
                for patient in test_seqs:
                    filtered_patient = []
                    for visit in patient:
                        filtered_visit = [code for code in visit if code < original_num_codes]
                        if filtered_visit:  # Chỉ thêm visit không rỗng
                            filtered_patient.append(filtered_visit)
                    if len(filtered_patient) >= 2:  # Chỉ thêm patient có ít nhất 2 visits
                        filtered_test_seqs.append(filtered_patient)
                
                print(f"Filtered test set: {len(filtered_test_seqs)} patients (from {len(test_seqs)})")
                
                if filtered_test_seqs:
                    metrics = evaluate_model(model, filtered_test_seqs, original_num_codes, original_num_codes, device)
                    print(f"Top-10 Accuracy: {metrics['top10_acc']:.4f}")
                    print(f"F1-Score: {metrics['f1']:.4f}")
                else:
                    print("❌ No valid patients after filtering")
                    
            else:
                print("✅ Sizes match, proceeding with normal evaluation...")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Thử phương pháp chính trước
    evaluate_all_models()
    
    # Nếu vẫn lỗi, thử phương pháp đơn giản
    print("\n" + "="*80)
    print("ALTERNATIVE EVALUATION")
    print("="*80)
    simple_evaluation()
