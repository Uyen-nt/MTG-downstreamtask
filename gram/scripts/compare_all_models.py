# gram/scripts/compare_all_models.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch

# ===============================
# CONFIG - ALL MODELS
# ===============================
TREE_PREFIX = "gram/data/mimic3_tree"
TEST_FILE = "gram/data/mimic_test.seqs"

MODELS = {
    "Synthetic Pretrained": "gram/data/synth_train_best.pt",
    "Fine-tuned MIMIC": "gram/data/finetuned_best.pt", 
    "Real Data Trained": "gram/data/real_train_best.pt"
}

# ===============================
# EVALUATION FUNCTIONS
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
    """Đánh giá model toàn diện trên test set"""
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
            
            # Collect predictions for analysis
            for b in range(pred.size(1)):
                last_idx = mask[:, b].sum().int() - 1
                if last_idx >= 0:
                    last_pred = pred[last_idx, b].cpu().numpy()
                    last_true = y[last_idx, b].cpu().numpy()
                    
                    # Top-k predictions
                    pred_top5 = np.argsort(-last_pred)[:5]
                    pred_top10 = np.argsort(-last_pred)[:10]
                    pred_top20 = np.argsort(-last_pred)[:20]
                    true_codes = np.where(last_true == 1)[0]
                    
                    all_true.append(true_codes)
                    all_pred.append({
                        'top5': pred_top5,
                        'top10': pred_top10, 
                        'top20': pred_top20,
                        'probabilities': last_pred
                    })
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(all_true, all_pred, num_codes)
    metrics['loss'] = total_loss / batch_count if batch_count > 0 else float('inf')
    metrics['num_patients'] = len(all_true)
    
    return metrics

def calculate_comprehensive_metrics(true_list, pred_list, num_codes):
    """Tính toán comprehensive metrics"""
    if not true_list:
        return {
            'top1_acc': 0, 'top3_acc': 0, 'top5_acc': 0, 'top10_acc': 0, 'top20_acc': 0,
            'precision': 0, 'recall': 0, 'f1': 0, 'jaccard': 0,
            'avg_true_codes': 0, 'avg_pred_codes': 0, 'coverage': 0
        }
    
    # Top-k accuracy for different k values
    top1_acc = top_k_accuracy(true_list, [p['top5'][:1] for p in pred_list], k=1)
    top3_acc = top_k_accuracy(true_list, [p['top5'][:3] for p in pred_list], k=3)
    top5_acc = top_k_accuracy(true_list, [p['top5'] for p in pred_list], k=5)
    top10_acc = top_k_accuracy(true_list, [p['top10'] for p in pred_list], k=10)
    top20_acc = top_k_accuracy(true_list, [p['top20'] for p in pred_list], k=20)
    
    # Binary metrics
    all_true_binary = []
    all_pred_binary = []
    
    avg_true_codes = 0
    avg_pred_codes = 0
    
    for true_codes, pred_dict in zip(true_list, pred_list):
        true_binary = np.zeros(num_codes)
        pred_binary = np.zeros(num_codes)
        
        true_binary[list(true_codes)] = 1
        # Use top-10 for binary metrics
        pred_binary[list(pred_dict['top10'])] = 1
        
        all_true_binary.append(true_binary)
        all_pred_binary.append(pred_binary)
        
        avg_true_codes += len(true_codes)
        avg_pred_codes += len(pred_dict['top10'])
    
    all_true_binary = np.array(all_true_binary)
    all_pred_binary = np.array(all_pred_binary)
    
    precision = precision_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    recall = recall_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    f1 = f1_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    jaccard = jaccard_score(all_true_binary, all_pred_binary, average='micro')
    
    # Coverage: percentage of unique codes predicted
    unique_true_codes = len(np.unique(np.where(all_true_binary == 1)[1]))
    unique_pred_codes = len(np.unique(np.where(all_pred_binary == 1)[1]))
    coverage = unique_pred_codes / unique_true_codes if unique_true_codes > 0 else 0
    
    return {
        'top1_acc': top1_acc,
        'top3_acc': top3_acc, 
        'top5_acc': top5_acc,
        'top10_acc': top10_acc,
        'top20_acc': top20_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'avg_true_codes': avg_true_codes / len(true_list),
        'avg_pred_codes': avg_pred_codes / len(pred_list),
        'coverage': coverage
    }

def top_k_accuracy(true_list, pred_list, k=10):
    """Tính top-k accuracy"""
    correct = 0
    total = 0
    
    for true_codes, pred_codes in zip(true_list, pred_list):
        if len(set(true_codes) & set(pred_codes[:k])) > 0:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def analyze_prediction_quality(model, test_seqs, num_codes, num_classes, device, sample_size=10):
    """Phân tích chất lượng dự đoán chi tiết"""
    model.eval()
    
    print(f"\n🔍 PHÂN TÍCH CHẤT LƯỢNG DỰ ĐOÁN (mẫu {sample_size} patients)")
    print("=" * 80)
    
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
            top10_pred = np.argsort(-last_pred)[:10]
            top5_pred = top10_pred[:5]
            
            # Calculate metrics for this patient
            hit_5 = len(set(true_next) & set(top5_pred)) > 0
            hit_10 = len(set(true_next) & set(top10_pred)) > 0
            num_hits = len(set(true_next) & set(top10_pred))
            
            analysis_results.append({
                'patient_id': i,
                'history_len': len(history),
                'true_codes_count': len(true_next),
                'hit_top5': hit_5,
                'hit_top10': hit_10,
                'num_hits': num_hits,
                'top1_prob': last_pred[top10_pred[0]]
            })
            
            if i < 3:  # Print detailed analysis for first 3 patients
                print(f"\nPatient {i}:")
                print(f"  Lịch sử: {len(history)} visits")
                print(f"  Mã thực tế: {true_next}")
                print(f"  Top-5 dự đoán: {top5_pred.tolist()}")
                print(f"  Hit Top-5: {'✅' if hit_5 else '❌'}, Hit Top-10: {'✅' if hit_10 else '❌'}")
                print(f"  Số mã đúng: {num_hits}/{len(true_next)}")
    
    # Summary statistics
    if analysis_results:
        hit_top5_rate = np.mean([r['hit_top5'] for r in analysis_results])
        hit_top10_rate = np.mean([r['hit_top10'] for r in analysis_results])
        avg_hits = np.mean([r['num_hits'] for r in analysis_results])
        avg_true_codes = np.mean([r['true_codes_count'] for r in analysis_results])
        
        print(f"\n📊 TỔNG HỢP MẪU:")
        print(f"  Top-5 Accuracy: {hit_top5_rate:.1%}")
        print(f"  Top-10 Accuracy: {hit_top10_rate:.1%}")
        print(f"  Số mã đúng trung bình: {avg_hits:.2f}/{avg_true_codes:.2f}")

# ===============================
# MAIN COMPARISON FUNCTION
# ===============================

def compare_all_models():
    """So sánh toàn diện tất cả models"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("🤖 SO SÁNH TOÀN DIỆN: SYNTHETIC vs FINE-TUNED vs REAL DATA TRAINED")
    print("=" * 80)
    
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
    
    # Evaluate all models
    results = {}
    
    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Đánh giá: {model_name}")
        print(f"{'='*60}")
        
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
        metrics = evaluate_model_comprehensive(model, test_seqs, num_codes, num_codes, device)
        results[model_name] = metrics
        
        # Print results
        print(f"📈 Performance Metrics:")
        print(f"  • Loss:          {metrics['loss']:.4f}")
        print(f"  • Top-1 Acc:     {metrics['top1_acc']:.4f}")
        print(f"  • Top-3 Acc:     {metrics['top3_acc']:.4f}")
        print(f"  • Top-5 Acc:     {metrics['top5_acc']:.4f}")
        print(f"  • Top-10 Acc:    {metrics['top10_acc']:.4f}")
        print(f"  • Top-20 Acc:    {metrics['top20_acc']:.4f}")
        print(f"  • F1-Score:      {metrics['f1']:.4f}")
        print(f"  • Precision:     {metrics['precision']:.4f}")
        print(f"  • Recall:        {metrics['recall']:.4f}")
        print(f"  • Jaccard:       {metrics['jaccard']:.4f}")
        print(f"  • Coverage:      {metrics['coverage']:.1%}")
        
        # Detailed analysis for each model
        analyze_prediction_quality(model, test_seqs, num_codes, num_codes, device)
    
    # Print comprehensive comparison table
    print_comparison_table(results)

def print_comparison_table(results):
    """In bảng so sánh đẹp"""
    print("\n" + "=" * 100)
    print("📊 BẢNG SO SÁNH TOÀN DIỆN")
    print("=" * 100)
    
    if len(results) < 2:
        print("❌ Không đủ models để so sánh")
        return
    
    # Create comparison dataframe
    metrics_to_show = [
        'top1_acc', 'top3_acc', 'top5_acc', 'top10_acc', 'top20_acc',
        'f1', 'precision', 'recall', 'jaccard', 'loss', 'coverage'
    ]
    
    # Print header
    header = f"{'Metric':<15} " + "".join([f"| {name:<20} " for name in results.keys()])
    print(header)
    print("-" * len(header))
    
    # Print metrics
    for metric in metrics_to_show:
        row = f"{metric:<15} "
        for model_name in results.keys():
            value = results[model_name][metric]
            if metric == 'loss':
                row += f"| {value:<20.4f} "
            elif metric == 'coverage':
                row += f"| {value:<19.1%} "
            else:
                row += f"| {value:<20.4f} "
        print(row)
    
    # Calculate improvements
    if "Synthetic Pretrained" in results and "Fine-tuned MIMIC" in results:
        print("\n" + "=" * 100)
        print("📈 PHÂN TÍCH HIỆU QUẢ FINE-TUNING")
        print("=" * 100)
        
        synth = results["Synthetic Pretrained"]
        ft = results["Fine-tuned MIMIC"]
        
        key_metrics = ['top5_acc', 'top10_acc', 'f1', 'loss']
        improvements = {}
        
        for metric in key_metrics:
            if metric == 'loss':
                # Lower loss is better
                improvement = synth[metric] - ft[metric]
            else:
                # Higher is better  
                improvement = ft[metric] - synth[metric]
            improvements[metric] = improvement
        
        print(f"{'Metric':<15} | {'Synthetic':<10} | {'Fine-tuned':<10} | {'Improvement':<12} | {'% Change':<10}")
        print("-" * 70)
        
        for metric in key_metrics:
            synth_val = synth[metric]
            ft_val = ft[metric]
            improvement = improvements[metric]
            
            if metric == 'loss':
                pct_change = (improvement / synth_val) * 100 if synth_val != 0 else 0
            else:
                pct_change = (improvement / synth_val) * 100 if synth_val != 0 else 0
            
            print(f"{metric:<15} | {synth_val:10.4f} | {ft_val:10.4f} | {improvement:>11.4f} | {pct_change:>9.1f}%")
    
    # Compare Real Data vs Others
    if "Real Data Trained" in results:
        print("\n" + "=" * 100)
        print("🎯 REAL DATA TRAINED vs OTHERS")
        print("=" * 100)
        
        real = results["Real Data Trained"]
        
        for other_name in ["Synthetic Pretrained", "Fine-tuned MIMIC"]:
            if other_name in results:
                other = results[other_name]
                print(f"\nVS {other_name}:")
                print(f"  • Top-10 Acc: {real['top10_acc']:.4f} vs {other['top10_acc']:.4f} → {('✅' if real['top10_acc'] > other['top10_acc'] else '❌')}")
                print(f"  • F1-Score:   {real['f1']:.4f} vs {other['f1']:.4f} → {('✅' if real['f1'] > other['f1'] else '❌')}")
                print(f"  • Loss:       {real['loss']:.4f} vs {other['loss']:.4f} → {('✅' if real['loss'] < other['loss'] else '❌')}")

if __name__ == "__main__":
    compare_all_models()
