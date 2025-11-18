# gram/scripts/compare_finetune_vs_real.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, roc_auc_score
from sklearn.preprocessing import label_binarize

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
# COMPREHENSIVE EVALUATION FUNCTIONS
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

def evaluate_model_comprehensive(model, test_seqs, num_codes, num_classes, device):
    """Đánh giá model toàn diện với AUC và các metrics nâng cao"""
    model.eval()
    total_loss = 0
    all_true = []
    all_pred = []
    all_pred_probs = []
    batch_count = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for i in range(0, len(test_seqs), 32):
            x_batch = [p[:-1] for p in test_seqs[i:i+32]]
            y_batch = [p[1:] for p in test_seqs[i:i+32]]
            
            if not x_batch:
                continue
                
            x, _, mask, _ = pad_batch(x_batch, num_classes, num_codes, device)
            _, y, _, _ = pad_batch(y_batch, num_classes, num_codes, device)
            
            pred = model(x, mask)
            y_labels = y.argmax(dim=-1)
            
            # Calculate loss
            loss_per_step = loss_fn(pred.permute(0, 2, 1), y_labels)
            loss_masked = loss_per_step * mask
            batch_loss = loss_masked.sum() / mask.sum()
            total_loss += batch_loss.item()
            batch_count += 1
            
            # Collect predictions for comprehensive analysis
            for b in range(pred.size(1)):
                last_idx = mask[:, b].sum().int() - 1
                if last_idx >= 0:
                    last_pred = pred[last_idx, b].cpu().numpy()
                    last_true = y[last_idx, b].cpu().numpy()
                    
                    # Top-k predictions
                    pred_top1 = np.argsort(-last_pred)[:1]
                    pred_top3 = np.argsort(-last_pred)[:3]
                    pred_top5 = np.argsort(-last_pred)[:5]
                    pred_top10 = np.argsort(-last_pred)[:10]
                    true_codes = np.where(last_true == 1)[0]
                    
                    all_true.append(true_codes)
                    all_pred.append({
                        'top1': pred_top1,
                        'top3': pred_top3,
                        'top5': pred_top5,
                        'top10': pred_top10,
                        'probabilities': last_pred
                    })
                    all_pred_probs.append(last_pred)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(all_true, all_pred, all_pred_probs, num_codes)
    metrics['loss'] = total_loss / batch_count if batch_count > 0 else float('inf')
    metrics['num_patients'] = len(all_true)
    
    return metrics

def calculate_comprehensive_metrics(true_list, pred_list, pred_probs_list, num_codes):
    """Tính toán comprehensive metrics với AUC"""
    if not true_list:
        return {
            'top1_acc': 0, 'top3_acc': 0, 'top5_acc': 0, 'top10_acc': 0,
            'precision': 0, 'recall': 0, 'f1': 0, 'jaccard': 0,
            'auc_micro': 0, 'auc_macro': 0, 'avg_true_codes': 0,
            'avg_pred_codes': 0, 'coverage': 0
        }
    
    # Top-k accuracy for different k values
    top1_acc = top_k_accuracy(true_list, [p['top1'] for p in pred_list], k=1)
    top3_acc = top_k_accuracy(true_list, [p['top3'] for p in pred_list], k=3)
    top5_acc = top_k_accuracy(true_list, [p['top5'] for p in pred_list], k=5)
    top10_acc = top_k_accuracy(true_list, [p['top10'] for p in pred_list], k=10)
    
    # Binary metrics
    all_true_binary = []
    all_pred_binary = []
    all_pred_probs_binary = []
    
    avg_true_codes = 0
    avg_pred_codes = 0
    
    for true_codes, pred_dict, pred_probs in zip(true_list, pred_list, pred_probs_list):
        true_binary = np.zeros(num_codes)
        pred_binary = np.zeros(num_codes)
        
        true_binary[list(true_codes)] = 1
        # Use top-10 for binary metrics
        pred_binary[list(pred_dict['top10'])] = 1
        
        all_true_binary.append(true_binary)
        all_pred_binary.append(pred_binary)
        all_pred_probs_binary.append(pred_probs)
        
        avg_true_codes += len(true_codes)
        avg_pred_codes += len(pred_dict['top10'])
    
    all_true_binary = np.array(all_true_binary)
    all_pred_binary = np.array(all_pred_binary)
    all_pred_probs_binary = np.array(all_pred_probs_binary)
    
    # Calculate standard metrics
    precision = precision_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    recall = recall_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    f1 = f1_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    jaccard = jaccard_score(all_true_binary, all_pred_binary, average='micro')
    
    # Calculate AUC (có thể tốn nhiều bộ nhớ với số classes lớn)
    auc_micro, auc_macro = calculate_auc_safe(all_true_binary, all_pred_probs_binary, num_codes)
    
    # Coverage: percentage of unique codes predicted
    unique_true_codes = len(np.unique(np.where(all_true_binary == 1)[1]))
    unique_pred_codes = len(np.unique(np.where(all_pred_binary == 1)[1]))
    coverage = unique_pred_codes / unique_true_codes if unique_true_codes > 0 else 0
    
    return {
        'top1_acc': top1_acc,
        'top3_acc': top3_acc, 
        'top5_acc': top5_acc,
        'top10_acc': top10_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'auc_micro': auc_micro,
        'auc_macro': auc_macro,
        'avg_true_codes': avg_true_codes / len(true_list),
        'avg_pred_codes': avg_pred_codes / len(pred_list),
        'coverage': coverage
    }

def calculate_auc_safe(true_binary, pred_probs, num_codes, max_classes=1000):
    """Tính AUC an toàn với số classes lớn"""
    try:
        # Giới hạn số classes để tính AUC (tránh memory issues)
        if num_codes > max_classes:
            # Chọn các classes phổ biến nhất
            class_frequency = np.sum(true_binary, axis=0)
            top_classes = np.argsort(-class_frequency)[:max_classes]
            
            true_binary_limited = true_binary[:, top_classes]
            pred_probs_limited = pred_probs[:, top_classes]
            
            auc_micro = roc_auc_score(true_binary_limited, pred_probs_limited, average='micro')
            auc_macro = roc_auc_score(true_binary_limited, pred_probs_limited, average='macro')
        else:
            auc_micro = roc_auc_score(true_binary, pred_probs, average='micro')
            auc_macro = roc_auc_score(true_binary, pred_probs, average='macro')
        
        return auc_micro, auc_macro
    except Exception as e:
        print(f"⚠️  AUC calculation error: {e}")
        return 0.0, 0.0

def top_k_accuracy(true_list, pred_list, k=10):
    """Tính top-k accuracy"""
    correct = 0
    total = 0
    
    for true_codes, pred_codes in zip(true_list, pred_list):
        if len(set(true_codes) & set(pred_codes[:k])) > 0:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def analyze_prediction_patterns(model, test_seqs, num_codes, num_classes, device, sample_size=5):
    """Phân tích patterns dự đoán chi tiết"""
    model.eval()
    
    print(f"\n🔍 PHÂN TÍCH PATTERNS DỰ ĐOÁN (mẫu {sample_size} patients)")
    print("=" * 90)
    
    sample_seqs = test_seqs[:sample_size]
    analysis_results = []
    
    with torch.no_grad():
        for i, patient in enumerate(sample_seqs):
            if len(patient) < 2:
                continue
                
            history = patient[:-1]
            true_next = patient[-1]
            
            x, _, mask, _ = pad_batch([history], num_classes, num_codes, device)
            pred = model(x, mask).squeeze(1)
            
            last_pred = pred[-1].cpu().numpy()
            top5_pred = np.argsort(-last_pred)[:5]
            top10_pred = np.argsort(-last_pred)[:10]
            
            # Tính confidence scores
            top1_confidence = last_pred[top5_pred[0]]
            top5_avg_confidence = np.mean(last_pred[top5_pred])
            
            # Tính độ chính xác
            hit_top5 = len(set(true_next) & set(top5_pred)) > 0
            hit_top10 = len(set(true_next) & set(top10_pred)) > 0
            num_hits = len(set(true_next) & set(top10_pred))
            precision_at_k = num_hits / len(top10_pred) if top10_pred.any() else 0
            
            analysis_results.append({
                'patient_id': i,
                'history_len': len(history),
                'true_codes_count': len(true_next),
                'hit_top5': hit_top5,
                'hit_top10': hit_top10,
                'num_hits': num_hits,
                'precision_at_10': precision_at_k,
                'top1_confidence': top1_confidence,
                'top5_avg_confidence': top5_avg_confidence
            })
            
            # Print detailed analysis for each patient
            print(f"\nPatient {i}:")
            print(f"  Lịch sử: {len(history)} visits")
            print(f"  Mã thực tế: {true_next}")
            print(f"  Top-5 dự đoán: {top5_pred.tolist()}")
            print(f"  Confidence: {top1_confidence:.4f} (top1), {top5_avg_confidence:.4f} (top5 avg)")
            print(f"  Hit Top-5: {'✅' if hit_top5 else '❌'}, Hit Top-10: {'✅' if hit_top10 else '❌'}")
            print(f"  Precision@10: {precision_at_k:.1%}")
    
    # Summary statistics
    if analysis_results:
        hit_top5_rate = np.mean([r['hit_top5'] for r in analysis_results])
        hit_top10_rate = np.mean([r['hit_top10'] for r in analysis_results])
        avg_precision = np.mean([r['precision_at_10'] for r in analysis_results])
        avg_confidence = np.mean([r['top1_confidence'] for r in analysis_results])
        
        print(f"\n📊 TỔNG HỢP MẪU:")
        print(f"  Top-5 Accuracy: {hit_top5_rate:.1%}")
        print(f"  Top-10 Accuracy: {hit_top10_rate:.1%}")
        print(f"  Average Precision@10: {avg_precision:.1%}")
        print(f"  Average Top-1 Confidence: {avg_confidence:.4f}")

# ===============================
# MAIN COMPARISON FUNCTION
# ===============================

def compare_finetune_vs_real():
    """So sánh chuyên sâu Fine-tuned vs Real Data Trained"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print("🤖 SO SÁNH CHUYÊN SÂU: FINE-TUNED vs REAL DATA TRAINED")
    print("=" * 100)
    
    # Load test data
    test_seqs = load_test_data()
    if not test_seqs:
        return
    
    # Compute num_codes
    num_codes = max(max(max(v) for v in p) for p in test_seqs) + 1
    
    # Load tree
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)
    
    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))
    max_index_tree = max(all_idx)
    
    print(f"📊 Dataset Info:")
    print(f"  • Số patients test: {len(test_seqs)}")
    print(f"  • Số mã bệnh: {num_codes}")
    print(f"  • Device: {device}")
    print()
    
    # Evaluate both models
    results = {}
    
    for model_name, model_path in MODELS.items():
        print(f"{'='*50}")
        print(f"Đánh giá: {model_name}")
        print(f"{'='*50}")
        
        # Create model
        model = GRAM(
            input_dim=num_codes,
            num_classes=num_codes,
            num_levels=len(tree_leaves),
            emb_dim=128,
            att_dim=128,
            hidden_dim=128,
            tree_leaves=tree_leaves,
            tree_ancestors=tree_anc,
            max_index_in_tree=max_index_tree,
            device=device,
        ).to(device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            continue
        
        # Comprehensive evaluation
        print("Đang đánh giá model...")
        metrics = evaluate_model_comprehensive(model, test_seqs, num_codes, num_codes, device)
        results[model_name] = metrics
        
        # Print detailed results
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  • Loss:              {metrics['loss']:.4f}")
        print(f"  • Top-1 Accuracy:    {metrics['top1_acc']:.4f}")
        print(f"  • Top-3 Accuracy:    {metrics['top3_acc']:.4f}")
        print(f"  • Top-5 Accuracy:    {metrics['top5_acc']:.4f}")
        print(f"  • Top-10 Accuracy:   {metrics['top10_acc']:.4f}")
        print(f"  • Precision:         {metrics['precision']:.4f}")
        print(f"  • Recall:            {metrics['recall']:.4f}")
        print(f"  • F1-Score:          {metrics['f1']:.4f}")
        print(f"  • Jaccard:           {metrics['jaccard']:.4f}")
        print(f"  • AUC Micro:         {metrics['auc_micro']:.4f}")
        print(f"  • AUC Macro:         {metrics['auc_macro']:.4f}")
        print(f"  • Coverage:          {metrics['coverage']:.1%}")
        print(f"  • Avg True Codes:    {metrics['avg_true_codes']:.2f}")
        print(f"  • Avg Pred Codes:    {metrics['avg_pred_codes']:.2f}")
        
        # Detailed analysis
        analyze_prediction_patterns(model, test_seqs, num_codes, num_codes, device)
    
    # Print comprehensive comparison table
    print_comparison_table(results)

def print_comparison_table(results):
    """In bảng so sánh chi tiết"""
    print("\n" + "=" * 120)
    print("📊 BẢNG SO SÁNH TOÀN DIỆN: FINE-TUNED vs REAL DATA TRAINED")
    print("=" * 120)
    
    if len(results) < 2:
        print("❌ Không đủ models để so sánh")
        return
    
    # Metrics categories
    accuracy_metrics = ['top1_acc', 'top3_acc', 'top5_acc', 'top10_acc']
    classification_metrics = ['precision', 'recall', 'f1', 'jaccard']
    advanced_metrics = ['auc_micro', 'auc_macro', 'coverage']
    efficiency_metrics = ['loss', 'avg_true_codes', 'avg_pred_codes']
    
    model_names = list(results.keys())
    ft_metrics = results["Fine-tuned"]
    real_metrics = results["Real Data Trained"]
    
    # Print comparison header
    header = f"{'Metric':<20} | {'Fine-tuned':<12} | {'Real Data':<12} | {'Difference':<12} | {'% Change':<10} | {'Better':<8}"
    print(header)
    print("-" * len(header))
    
    # Compare all metrics
    all_metrics = accuracy_metrics + classification_metrics + advanced_metrics + efficiency_metrics
    
    for metric in all_metrics:
        ft_val = ft_metrics[metric]
        real_val = real_metrics[metric]
        
        # Calculate difference and percentage change
        if metric == 'loss':
            # Lower loss is better
            difference = ft_val - real_val
            better = "Real" if real_val < ft_val else "Fine-tuned"
        else:
            # Higher is better
            difference = real_val - ft_val
            better = "Real" if real_val > ft_val else "Fine-tuned"
        
        # Calculate percentage change
        if ft_val != 0:
            pct_change = (difference / ft_val) * 100
        else:
            pct_change = 0
        
        # Format values based on metric type
        if metric == 'loss':
            ft_str = f"{ft_val:.4f}"
            real_str = f"{real_val:.4f}"
            diff_str = f"{difference:+.4f}"
        elif metric == 'coverage':
            ft_str = f"{ft_val:.1%}"
            real_str = f"{real_val:.1%}"
            diff_str = f"{difference:+.1%}"
        else:
            ft_str = f"{ft_val:.4f}"
            real_str = f"{real_val:.4f}"
            diff_str = f"{difference:+.4f}"
        
        print(f"{metric:<20} | {ft_str:<12} | {real_str:<12} | {diff_str:<12} | {pct_change:>9.1f}% | {better:<8}")
    
    # Summary analysis
    print("\n" + "=" * 120)
    print("🎯 PHÂN TÍCH TỔNG QUAN")
    print("=" * 120)
    
    # Count wins for each model
    ft_wins = 0
    real_wins = 0
    
    for metric in all_metrics:
        ft_val = ft_metrics[metric]
        real_val = real_metrics[metric]
        
        if metric == 'loss':
            if real_val < ft_val:
                real_wins += 1
            elif ft_val < real_val:
                ft_wins += 1
        else:
            if real_val > ft_val:
                real_wins += 1
            elif ft_val > real_val:
                ft_wins += 1
    
    print(f"Fine-tuned wins: {ft_wins} metrics")
    print(f"Real Data wins: {real_wins} metrics")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # Top-10 Accuracy comparison
    top10_diff = real_metrics['top10_acc'] - ft_metrics['top10_acc']
    if abs(top10_diff) > 0.01:
        better_model = "Real Data" if top10_diff > 0 else "Fine-tuned"
        print(f"  • {better_model} có Top-10 Accuracy cao hơn {abs(top10_diff):.3f}")
    
    # F1-Score comparison
    f1_diff = real_metrics['f1'] - ft_metrics['f1']
    if abs(f1_diff) > 0.01:
        better_model = "Real Data" if f1_diff > 0 else "Fine-tuned"
        print(f"  • {better_model} có F1-Score cao hơn {abs(f1_diff):.3f}")
    
    # Loss comparison
    loss_diff = ft_metrics['loss'] - real_metrics['loss']
    if abs(loss_diff) > 0.1:
        better_model = "Real Data" if loss_diff > 0 else "Fine-tuned"
        print(f"  • {better_model} có Loss thấp hơn {abs(loss_diff):.3f}")
    
    # Coverage comparison
    coverage_diff = real_metrics['coverage'] - ft_metrics['coverage']
    if abs(coverage_diff) > 0.05:
        better_model = "Real Data" if coverage_diff > 0 else "Fine-tuned"
        print(f"  • {better_model} có Coverage tốt hơn {abs(coverage_diff):.1%}")

if __name__ == "__main__":
    compare_finetune_vs_real()
