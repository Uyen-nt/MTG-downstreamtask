# gram/scripts/compare_simple.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch

# ===============================
# CONFIG
# ===============================
TREE_PREFIX = "gram/data/mimic3_tree"
TEST_FILE = "gram/data/mimic_test.seqs"

MODELS = {
    "Fine-tuned": "gram/data/finetuned_best.pt",
    "Real Data Trained": "gram/data/real_train_best.pt"
}

# ===============================
# SIMPLE EVALUATION FUNCTIONS
# ===============================

def load_test_data():
    """Load test data for evaluation"""
    try:
        test_seqs = pickle.load(open(TEST_FILE, "rb"))
        print(f"✅ Loaded test set: {len(test_seqs)} patients")
        return test_seqs
    except:
        print("❌ Test set not found")
        return []

def create_compatible_model(model_path, current_num_codes, tree_leaves, tree_anc, max_index_tree, device):
    """Tạo model tương thích với số lượng codes hiện tại"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        original_num_codes = checkpoint['out.weight'].shape[0]
        print(f"Original num_codes: {original_num_codes}")
        
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
        print(f"✅ Model loaded successfully")
        
        return model, original_num_codes
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def filter_test_seqs(test_seqs, max_code):
    """Filter test sequences để chỉ giữ codes trong phạm vi model"""
    filtered_seqs = []
    
    for patient in test_seqs:
        filtered_patient = []
        for visit in patient:
            filtered_visit = [code for code in visit if code < max_code]
            if filtered_visit:
                filtered_patient.append(filtered_visit)
        if len(filtered_patient) >= 2:
            filtered_seqs.append(filtered_patient)
    
    print(f"Filtered test set: {len(filtered_seqs)} patients")
    return filtered_seqs

def evaluate_model_simple(model, test_seqs, num_codes, num_classes, device):
    """Đánh giá model đơn giản - chỉ tính metrics"""
    model.eval()
    total_loss = 0
    all_true = []
    all_pred = []
    batch_count = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for i in range(0, len(test_seqs), 32):
            x_batch = [p[:-1] for p in test_seqs[i:i+32]]
            y_batch = [p[1:] for p in test_seqs[i:i+32]]
            
            if not x_batch:
                continue
                
            try:
                x, _, mask, _ = pad_batch(x_batch, num_classes, num_codes, device)
                _, y, _, _ = pad_batch(y_batch, num_classes, num_codes, device)
                
                # Skip if sequence length is 0
                if x.size(0) == 0:
                    continue
                    
                pred = model(x, mask)
                y_labels = y.argmax(dim=-1)
                
                loss_per_step = loss_fn(pred.permute(0, 2, 1), y_labels)
                loss_masked = loss_per_step * mask
                batch_loss = loss_masked.sum() / mask.sum()
                total_loss += batch_loss.item()
                batch_count += 1
                
                # Collect predictions
                for b in range(pred.size(1)):
                    last_idx = mask[:, b].sum().int() - 1
                    if last_idx >= 0:
                        last_pred = pred[last_idx, b].cpu().numpy()
                        last_true = y[last_idx, b].cpu().numpy()
                        
                        pred_top10 = np.argsort(-last_pred)[:10]
                        true_codes = np.where(last_true == 1)[0]
                        
                        all_true.append(true_codes)
                        all_pred.append(pred_top10)
                        
            except Exception as e:
                print(f"⚠️  Skipping batch due to error: {e}")
                continue
    
    # Calculate metrics
    metrics = calculate_metrics_simple(all_true, all_pred, num_codes)
    metrics['loss'] = total_loss / batch_count if batch_count > 0 else float('inf')
    metrics['num_patients'] = len(all_true)
    
    return metrics

def calculate_metrics_simple(true_list, pred_list, num_codes):
    """Tính toán metrics đơn giản"""
    if not true_list:
        return {
            'top1_acc': 0, 'top3_acc': 0, 'top5_acc': 0, 'top10_acc': 0,
            'precision': 0, 'recall': 0, 'f1': 0, 'jaccard': 0
        }
    
    # Top-k accuracy
    top1_acc = top_k_accuracy_simple(true_list, [p[:1] for p in pred_list], k=1)
    top3_acc = top_k_accuracy_simple(true_list, [p[:3] for p in pred_list], k=3)
    top5_acc = top_k_accuracy_simple(true_list, [p[:5] for p in pred_list], k=5)
    top10_acc = top_k_accuracy_simple(true_list, pred_list, k=10)
    
    # Binary metrics
    all_true_binary = []
    all_pred_binary = []
    
    for true_codes, pred_codes in zip(true_list, pred_list):
        true_binary = np.zeros(num_codes)
        pred_binary = np.zeros(num_codes)
        
        true_binary[list(true_codes)] = 1
        pred_binary[list(pred_codes)] = 1
        
        all_true_binary.append(true_binary)
        all_pred_binary.append(pred_binary)
    
    if all_true_binary:
        all_true_binary = np.array(all_true_binary)
        all_pred_binary = np.array(all_pred_binary)
        
        precision = precision_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
        recall = recall_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
        f1 = f1_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
        jaccard = jaccard_score(all_true_binary, all_pred_binary, average='micro')
    else:
        precision = recall = f1 = jaccard = 0
    
    return {
        'top1_acc': top1_acc,
        'top3_acc': top3_acc, 
        'top5_acc': top5_acc,
        'top10_acc': top10_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard
    }

def top_k_accuracy_simple(true_list, pred_list, k=10):
    """Tính top-k accuracy đơn giản"""
    correct = 0
    total = 0
    
    for true_codes, pred_codes in zip(true_list, pred_list):
        if len(set(true_codes) & set(pred_codes[:k])) > 0:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

# ===============================
# MAIN COMPARISON FUNCTION
# ===============================

def compare_models_simple():
    """So sánh đơn giản - chỉ in kết quả metrics"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("🤖 SO SÁNH METRICS: FINE-TUNED vs REAL DATA TRAINED")
    print("=" * 60)
    
    # Load test data
    test_seqs = load_test_data()
    if not test_seqs:
        return
    
    # Compute num_codes từ test data
    current_num_codes = max(max(max(v) for v in p) for p in test_seqs) + 1
    
    # Load tree với số lượng codes lớn hơn
    max_num_codes = max(current_num_codes, 2879)
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, max_num_codes, device)
    
    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))
    max_index_tree = max(all_idx)
    
    print(f"Test patients: {len(test_seqs)}")
    print(f"Current num_codes: {current_num_codes}")
    print()
    
    # Evaluate both models
    results = {}
    
    for model_name, model_path in MODELS.items():
        print(f"{'='*40}")
        print(f"📊 {model_name}")
        print(f"{'='*40}")
        
        # Tạo model tương thích
        model_result = create_compatible_model(model_path, current_num_codes, tree_leaves, tree_anc, max_index_tree, device)
        
        if model_result is None:
            continue
            
        model, original_num_codes = model_result
        
        # Filter test data
        filtered_test_seqs = filter_test_seqs(test_seqs, original_num_codes)
        
        if not filtered_test_seqs:
            print("❌ No valid patients")
            continue
        
        # Simple evaluation
        print("Evaluating...")
        metrics = evaluate_model_simple(model, filtered_test_seqs, original_num_codes, original_num_codes, device)
        results[model_name] = metrics
        
        # Print metrics only
        print(f"Loss:          {metrics['loss']:.4f}")
        print(f"Top-1 Acc:     {metrics['top1_acc']:.4f}")
        print(f"Top-3 Acc:     {metrics['top3_acc']:.4f}")
        print(f"Top-5 Acc:     {metrics['top5_acc']:.4f}")
        print(f"Top-10 Acc:    {metrics['top10_acc']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1-Score:      {metrics['f1']:.4f}")
        print(f"Jaccard:       {metrics['jaccard']:.4f}")
        print(f"Patients:      {metrics['num_patients']}")
        print()
    
    # Print comparison table
    if len(results) >= 2:
        print_comparison_table_simple(results)

def print_comparison_table_simple(results):
    """In bảng so sánh đơn giản"""
    print("\n" + "=" * 80)
    print("📊 COMPARISON TABLE")
    print("=" * 80)
    
    ft_metrics = results["Fine-tuned"]
    real_metrics = results["Real Data Trained"]
    
    # Metrics to compare
    metrics_list = [
        ('Loss', ft_metrics['loss'], real_metrics['loss'], 'lower'),
        ('Top-1 Acc', ft_metrics['top1_acc'], real_metrics['top1_acc'], 'higher'),
        ('Top-3 Acc', ft_metrics['top3_acc'], real_metrics['top3_acc'], 'higher'),
        ('Top-5 Acc', ft_metrics['top5_acc'], real_metrics['top5_acc'], 'higher'),
        ('Top-10 Acc', ft_metrics['top10_acc'], real_metrics['top10_acc'], 'higher'),
        ('Precision', ft_metrics['precision'], real_metrics['precision'], 'higher'),
        ('Recall', ft_metrics['recall'], real_metrics['recall'], 'higher'),
        ('F1-Score', ft_metrics['f1'], real_metrics['f1'], 'higher'),
        ('Jaccard', ft_metrics['jaccard'], real_metrics['jaccard'], 'higher'),
    ]
    
    # Print table
    print(f"{'Metric':<12} | {'Fine-tuned':<10} | {'Real Data':<10} | {'Diff':<8} | {'Better':<6}")
    print("-" * 60)
    
    ft_wins = 0
    real_wins = 0
    
    for name, ft_val, real_val, better_type in metrics_list:
        if better_type == 'lower':
            diff = ft_val - real_val
            better = "Real" if real_val < ft_val else "Fine-tuned" if ft_val < real_val else "Equal"
        else:
            diff = real_val - ft_val
            better = "Real" if real_val > ft_val else "Fine-tuned" if ft_val > real_val else "Equal"
        
        print(f"{name:<12} | {ft_val:<10.4f} | {real_val:<10.4f} | {diff:>7.4f} | {better:<6}")
        
        if better == "Fine-tuned":
            ft_wins += 1
        elif better == "Real":
            real_wins += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    print(f"Fine-tuned wins: {ft_wins} metrics")
    print(f"Real Data wins:  {real_wins} metrics")
    print(f"Equal:           {len(metrics_list) - ft_wins - real_wins} metrics")

if __name__ == "__main__":
    compare_models_simple()
