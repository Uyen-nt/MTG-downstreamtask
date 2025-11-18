# gram/scripts/compare_finetune_vs_real_fixed.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, roc_auc_score

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
# COMPATIBILITY FUNCTIONS
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
        # Load checkpoint để lấy số lượng codes gốc
        checkpoint = torch.load(model_path, map_location='cpu')
        original_num_codes = checkpoint['out.weight'].shape[0]
        print(f"Original num_codes in {model_path}: {original_num_codes}")
        
        # LUÔN tạo model với original_num_codes và filter test data
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
        
        # Load state dict gốc
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded with original size {original_num_codes}")
        
        return model, original_num_codes
        
    except Exception as e:
        print(f"Error creating compatible model: {e}")
        return None, None

def filter_test_seqs(test_seqs, max_code):
    """Filter test sequences để chỉ giữ codes trong phạm vi model"""
    filtered_seqs = []
    
    for patient in test_seqs:
        filtered_patient = []
        for visit in patient:
            filtered_visit = [code for code in visit if code < max_code]
            if filtered_visit:  # Chỉ thêm visit không rỗng
                filtered_patient.append(filtered_visit)
        if len(filtered_patient) >= 2:  # Chỉ thêm patient có ít nhất 2 visits
            filtered_seqs.append(filtered_patient)
    
    print(f"Filtered test set: {len(filtered_seqs)} patients (from {len(test_seqs)})")
    return filtered_seqs

# ===============================
# COMPREHENSIVE EVALUATION FUNCTIONS
# ===============================

def evaluate_model_comprehensive(model, test_seqs, num_codes, num_classes, device):
    """Đánh giá model toàn diện"""
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
            
            # Collect predictions
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
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(all_true, all_pred, num_codes)
    metrics['loss'] = total_loss / batch_count if batch_count > 0 else float('inf')
    metrics['num_patients'] = len(all_true)
    
    return metrics

def calculate_comprehensive_metrics(true_list, pred_list, num_codes):
    """Tính toán comprehensive metrics"""
    if not true_list:
        return {
            'top1_acc': 0, 'top3_acc': 0, 'top5_acc': 0, 'top10_acc': 0,
            'precision': 0, 'recall': 0, 'f1': 0, 'jaccard': 0,
            'avg_true_codes': 0, 'avg_pred_codes': 0, 'coverage': 0
        }
    
    # Top-k accuracy
    top1_acc = top_k_accuracy(true_list, [p['top1'] for p in pred_list], k=1)
    top3_acc = top_k_accuracy(true_list, [p['top3'] for p in pred_list], k=3)
    top5_acc = top_k_accuracy(true_list, [p['top5'] for p in pred_list], k=5)
    top10_acc = top_k_accuracy(true_list, [p['top10'] for p in pred_list], k=10)
    
    # Binary metrics
    all_true_binary = []
    all_pred_binary = []
    
    avg_true_codes = 0
    avg_pred_codes = 0
    
    for true_codes, pred_dict in zip(true_list, pred_list):
        true_binary = np.zeros(num_codes)
        pred_binary = np.zeros(num_codes)
        
        true_binary[list(true_codes)] = 1
        pred_binary[list(pred_dict['top10'])] = 1
        
        all_true_binary.append(true_binary)
        all_pred_binary.append(pred_binary)
        
        avg_true_codes += len(true_codes)
        avg_pred_codes += len(pred_dict['top10'])
    
    if all_true_binary:
        all_true_binary = np.array(all_true_binary)
        all_pred_binary = np.array(all_pred_binary)
        
        precision = precision_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
        recall = recall_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
        f1 = f1_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
        jaccard = jaccard_score(all_true_binary, all_pred_binary, average='micro')
    else:
        precision = recall = f1 = jaccard = 0
    
    # Coverage
    unique_true_codes = len(np.unique(np.where(all_true_binary == 1)[1])) if all_true_binary.any() else 0
    unique_pred_codes = len(np.unique(np.where(all_pred_binary == 1)[1])) if all_pred_binary.any() else 0
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

def analyze_prediction_patterns(model, test_seqs, num_codes, num_classes, device, sample_size=5):
    """Phân tích patterns dự đoán chi tiết"""
    model.eval()
    
    print(f"\n🔍 PHÂN TÍCH PATTERNS DỰ ĐOÁN (mẫu {sample_size} patients)")
    print("=" * 80)
    
    sample_seqs = test_seqs[:sample_size]
    
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
            
            # Tính độ chính xác
            hit_top5 = len(set(true_next) & set(top5_pred)) > 0
            hit_top10 = len(set(true_next) & set(top10_pred)) > 0
            num_hits = len(set(true_next) & set(top10_pred))
            
            print(f"\nPatient {i}:")
            print(f"  Lịch sử: {len(history)} visits")
            print(f"  Mã thực tế: {true_next}")
            print(f"  Top-5 dự đoán: {top5_pred.tolist()}")
            print(f"  Hit Top-5: {'✅' if hit_top5 else '❌'}, Hit Top-10: {'✅' if hit_top10 else '❌'}")
            print(f"  Số mã đúng: {num_hits}/{len(true_next)}")

# ===============================
# MAIN COMPARISON FUNCTION
# ===============================

def compare_finetune_vs_real():
    """So sánh chuyên sâu Fine-tuned vs Real Data Trained"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("🤖 SO SÁNH CHUYÊN SÂU: FINE-TUNED vs REAL DATA TRAINED")
    print("=" * 80)
    
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
    
    print(f"📊 Dataset Info:")
    print(f"  • Số patients test: {len(test_seqs)}")
    print(f"  • Current num_codes: {current_num_codes}")
    print(f"  • Device: {device}")
    print()
    
    # Evaluate both models
    results = {}
    
    for model_name, model_path in MODELS.items():
        print(f"{'='*50}")
        print(f"Đánh giá: {model_name}")
        print(f"{'='*50}")
        
        # Tạo model tương thích
        model_result = create_compatible_model(model_path, current_num_codes, tree_leaves, tree_anc, max_index_tree, device)
        
        if model_result is None:
            print(f"❌ Failed to load {model_name}")
            continue
            
        model, original_num_codes = model_result
        
        # Filter test data để phù hợp với model
        filtered_test_seqs = filter_test_seqs(test_seqs, original_num_codes)
        
        if not filtered_test_seqs:
            print("❌ No valid patients after filtering")
            continue
        
        # Comprehensive evaluation
        print("Đang đánh giá model...")
        metrics = evaluate_model_comprehensive(model, filtered_test_seqs, original_num_codes, original_num_codes, device)
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
        print(f"  • Coverage:          {metrics['coverage']:.1%}")
        print(f"  • Avg True Codes:    {metrics['avg_true_codes']:.2f}")
        print(f"  • Avg Pred Codes:    {metrics['avg_pred_codes']:.2f}")
        
        # Detailed analysis
        analyze_prediction_patterns(model, filtered_test_seqs, original_num_codes, original_num_codes, device)
    
    # Print comprehensive comparison table
    if len(results) >= 2:
        print_comparison_table(results)
    else:
        print("\n❌ Không đủ models để so sánh")

def print_comparison_table(results):
    """In bảng so sánh chi tiết"""
    print("\n" + "=" * 100)
    print("📊 BẢNG SO SÁNH TOÀN DIỆN: FINE-TUNED vs REAL DATA TRAINED")
    print("=" * 100)
    
    ft_metrics = results["Fine-tuned"]
    real_metrics = results["Real Data Trained"]
    
    # Metrics to compare
    metrics_list = [
        ('Top-1 Acc', ft_metrics['top1_acc'], real_metrics['top1_acc']),
        ('Top-3 Acc', ft_metrics['top3_acc'], real_metrics['top3_acc']),
        ('Top-5 Acc', ft_metrics['top5_acc'], real_metrics['top5_acc']),
        ('Top-10 Acc', ft_metrics['top10_acc'], real_metrics['top10_acc']),
        ('Precision', ft_metrics['precision'], real_metrics['precision']),
        ('Recall', ft_metrics['recall'], real_metrics['recall']),
        ('F1-Score', ft_metrics['f1'], real_metrics['f1']),
        ('Jaccard', ft_metrics['jaccard'], real_metrics['jaccard']),
        ('Coverage', ft_metrics['coverage'], real_metrics['coverage']),
        ('Loss', ft_metrics['loss'], real_metrics['loss']),
    ]
    
    # Print header
    print(f"{'Metric':<15} | {'Fine-tuned':<12} | {'Real Data':<12} | {'Difference':<12} | {'% Change':<10} | {'Better':<8}")
    print("-" * 85)
    
    ft_wins = 0
    real_wins = 0
    
    for name, ft_val, real_val in metrics_list:
        if name == 'Loss':
            # Lower loss is better
            difference = ft_val - real_val
            better = "Real" if real_val < ft_val else "Fine-tuned"
            pct_change = (difference / ft_val) * 100 if ft_val != 0 else 0
        else:
            # Higher is better
            difference = real_val - ft_val
            better = "Real" if real_val > ft_val else "Fine-tuned"
            pct_change = (difference / ft_val) * 100 if ft_val != 0 else 0
        
        # Format values
        if name == 'Coverage':
            ft_str = f"{ft_val:.1%}"
            real_str = f"{real_val:.1%}"
            diff_str = f"{difference:+.1%}"
        elif name == 'Loss':
            ft_str = f"{ft_val:.4f}"
            real_str = f"{real_val:.4f}"
            diff_str = f"{difference:+.4f}"
        else:
            ft_str = f"{ft_val:.4f}"
            real_str = f"{real_val:.4f}"
            diff_str = f"{difference:+.4f}"
        
        print(f"{name:<15} | {ft_str:<12} | {real_str:<12} | {diff_str:<12} | {pct_change:>9.1f}% | {better:<8}")
        
        # Count wins
        if better == "Fine-tuned":
            ft_wins += 1
        else:
            real_wins += 1
    
    # Summary
    print("\n" + "=" * 100)
    print("🎯 TỔNG KẾT")
    print("=" * 100)
    print(f"Fine-tuned wins: {ft_wins} metrics")
    print(f"Real Data wins: {real_wins} metrics")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    if real_metrics['top10_acc'] > ft_metrics['top10_acc']:
        improvement = real_metrics['top10_acc'] - ft_metrics['top10_acc']
        print(f"  • Real Data có Top-10 Accuracy cao hơn {improvement:.3f}")
    elif ft_metrics['top10_acc'] > real_metrics['top10_acc']:
        improvement = ft_metrics['top10_acc'] - real_metrics['top10_acc']
        print(f"  • Fine-tuned có Top-10 Accuracy cao hơn {improvement:.3f}")
    
    if real_metrics['f1'] > ft_metrics['f1']:
        improvement = real_metrics['f1'] - ft_metrics['f1']
        print(f"  • Real Data có F1-Score cao hơn {improvement:.3f}")
    elif ft_metrics['f1'] > real_metrics['f1']:
        improvement = ft_metrics['f1'] - real_metrics['f1']
        print(f"  • Fine-tuned có F1-Score cao hơn {improvement:.3f}")

if __name__ == "__main__":
    compare_finetune_vs_real()
