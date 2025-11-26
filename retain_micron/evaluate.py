import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def evaluate_topk_recall(model, loader, k=10):
    """Top-K Recall - Giữ nguyên như cũ"""
    model.eval()
    total_hits = 0
    total_codes = 0

    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            topk_indices = torch.topk(logits, k=k, dim=1)[1]
            pred_codes_batch = topk_indices.cpu().tolist()

            for i in range(labels.size(0)):
                pred_codes = pred_codes_batch[i]
                true_codes = torch.where(labels[i] > 0.5)[0].cpu().tolist()
                
                hits = len(set(pred_codes) & set(true_codes))
                total_hits += hits
                total_codes += len(true_codes) if true_codes else 1

    return total_hits / total_codes if total_codes > 0 else 0

def evaluate_comprehensive(model, loader, k_list=[5, 10, 20, 30]):
    """Đánh giá toàn diện với nhiều metric"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            # Lưu logits và labels để tính các metric
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            # Top-K predictions
            for k in k_list:
                topk_indices = torch.topk(logits, k=k, dim=1)[1]
                pred_codes_batch = topk_indices.cpu().tolist()
                all_predictions.append((k, pred_codes_batch, labels.cpu()))

    # Tính các metric
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    results = {}
    
    # Top-K Recall
    for k in k_list:
        recall_k = evaluate_topk_recall(model, loader, k=k)
        results[f'recall@{k}'] = recall_k
    
    # Precision, Recall, F1 cho threshold-based predictions
    predictions_binary = (torch.sigmoid(all_logits_tensor) > 0.5).float()
    
    # Micro averages
    try:
        results['precision_micro'] = precision_score(
            all_labels_tensor.view(-1).numpy(), 
            predictions_binary.view(-1).numpy(), 
            average='micro', zero_division=0
        )
        results['recall_micro'] = recall_score(
            all_labels_tensor.view(-1).numpy(),
            predictions_binary.view(-1).numpy(),
            average='micro', zero_division=0
        )
        results['f1_micro'] = f1_score(
            all_labels_tensor.view(-1).numpy(),
            predictions_binary.view(-1).numpy(),
            average='micro', zero_division=0
        )
    except:
        results['precision_micro'] = 0
        results['recall_micro'] = 0
        results['f1_micro'] = 0
    
    # Macro averages
    try:
        results['precision_macro'] = precision_score(
            all_labels_tensor.numpy(),
            predictions_binary.numpy(),
            average='macro', zero_division=0
        )
        results['recall_macro'] = recall_score(
            all_labels_tensor.numpy(),
            predictions_binary.numpy(),
            average='macro', zero_division=0
        )
        results['f1_macro'] = f1_score(
            all_labels_tensor.numpy(),
            predictions_binary.numpy(),
            average='macro', zero_division=0
        )
    except:
        results['precision_macro'] = 0
        results['recall_macro'] = 0
        results['f1_macro'] = 0
    
    # AUC-ROC (có thể không tính được nếu chỉ có 1 class)
    try:
        results['auc_roc_micro'] = roc_auc_score(
            all_labels_tensor.numpy(),
            torch.sigmoid(all_logits_tensor).numpy(),
            average='micro'
        )
    except:
        results['auc_roc_micro'] = 0
    
    return results

def print_evaluation_results(results):
    """In kết quả đánh giá đẹp mắt"""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    print("\n🎯 TOP-K RECALL METRICS:")
    for key, value in results.items():
        if key.startswith('recall@'):
            print(f"   {key.upper():<12}: {value:.4f}")
    
    print("\n📈 PRECISION-RECALL-F1 METRICS:")
    print(f"   {'METRIC':<15} {'MICRO':<8} {'MACRO':<8}")
    print(f"   {'-'*15} {'-'*8} {'-'*8}")
    print(f"   {'Precision':<15} {results['precision_micro']:.4f}   {results['precision_macro']:.4f}")
    print(f"   {'Recall':<15} {results['recall_micro']:.4f}   {results['recall_macro']:.4f}")
    print(f"   {'F1-Score':<15} {results['f1_micro']:.4f}   {results['f1_macro']:.4f}")
    
    print(f"\n📊 AUC-ROC SCORE:")
    print(f"   Micro AUC-ROC: {results['auc_roc_micro']:.4f}")
    
    print("="*60)

# =============================================================================
# CÁC HÀM PHÂN TÍCH CHI TIẾT MỚI THÊM
# =============================================================================

def debug_predictions_distribution(model, loader, n_codes):
    """Phân tích phân phối predictions để hiểu model behavior"""
    model.eval()
    all_predictions = []
    code_prediction_counts = torch.zeros(n_codes)
    total_samples = 0
    
    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)
            probs = torch.sigmoid(logits)
            
            # Count how many codes have prob > 0.5 per sample
            high_prob_codes = (probs > 0.5).sum(dim=1)
            all_predictions.extend(high_prob_codes.cpu().tolist())
            
            # Count predictions per code
            code_prediction_counts += (probs > 0.5).sum(dim=0).cpu()
            total_samples += labels.shape[0]
    
    # Phân tích phân phối
    avg_codes_predicted = np.mean(all_predictions)
    max_codes_predicted = np.max(all_predictions)
    min_codes_predicted = np.min(all_predictions)
    
    # Top predicted codes
    top_predicted = torch.topk(code_prediction_counts, min(10, n_codes))
    
    print(f"\n🔍 PREDICTION DISTRIBUTION ANALYSIS:")
    print(f"   Avg codes predicted per sample: {avg_codes_predicted:.2f}")
    print(f"   Min-Max codes predicted: {min_codes_predicted}-{max_codes_predicted}")
    print(f"   Unique codes predicted: {(code_prediction_counts > 0).sum().item()}/{n_codes}")
    print(f"   Top {len(top_predicted.indices)} most predicted codes: {top_predicted.indices.tolist()}")
    
    return {
        'avg_codes_per_sample': avg_codes_predicted,
        'unique_codes_predicted': (code_prediction_counts > 0).sum().item(),
        'top_predicted_codes': top_predicted.indices.tolist()
    }

def evaluate_improved(model, loader, k_list=[5, 10, 20, 30], n_codes=2869):
    """Evaluation cải tiến với phân tích chi tiết hơn"""
    model.eval()
    
    all_logits = []
    all_labels = []
    all_pred_probs = []
    
    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)
            probs = torch.sigmoid(logits)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_pred_probs.append(probs.cpu())
    
    # Concatenate all
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_probs_tensor = torch.cat(all_pred_probs, dim=0)
    
    results = {}
    
    # 1. Top-K Recall (giữ nguyên)
    for k in k_list:
        results[f'recall@{k}'] = evaluate_topk_recall(model, loader, k=k)
    
    # 2. Adaptive Thresholding (thay vì fixed 0.5)
    thresholds = [0.1, 0.3, 0.5, 0.7]
    for threshold in thresholds:
        preds_binary = (all_probs_tensor > threshold).float()
        
        # Micro metrics
        try:
            precision_micro = precision_score(
                all_labels_tensor.view(-1).numpy(),
                preds_binary.view(-1).numpy(),
                average='micro', zero_division=0
            )
            recall_micro = recall_score(
                all_labels_tensor.view(-1).numpy(),
                preds_binary.view(-1).numpy(),
                average='micro', zero_division=0
            )
            f1_micro = f1_score(
                all_labels_tensor.view(-1).numpy(),
                preds_binary.view(-1).numpy(),
                average='micro', zero_division=0
            )
        except:
            precision_micro, recall_micro, f1_micro = 0, 0, 0
        
        results[f'precision_micro_thresh{threshold}'] = precision_micro
        results[f'recall_micro_thresh{threshold}'] = recall_micro
        results[f'f1_micro_thresh{threshold}'] = f1_micro
    
    # 3. Code-wise analysis
    code_metrics = {
        'codes_with_predictions': 0,
        'codes_with_high_recall': 0,
        'avg_codes_per_sample': 0,
        'median_code_recall': 0
    }
    
    # Count how many codes are actually predicted (at threshold 0.5)
    avg_predicted_codes = (all_probs_tensor > 0.5).sum(dim=1).float().mean()
    code_metrics['avg_codes_per_sample'] = avg_predicted_codes.item()
    
    # Count how many codes have any predictions
    codes_with_any_pred = (all_probs_tensor > 0.5).sum(dim=0) > 0
    code_metrics['codes_with_predictions'] = codes_with_any_pred.sum().item()
    
    # Calculate code-wise recall
    code_recalls = []
    for code_idx in range(n_codes):
        true_pos = ((all_probs_tensor[:, code_idx] > 0.5) & (all_labels_tensor[:, code_idx] > 0)).sum()
        actual_pos = (all_labels_tensor[:, code_idx] > 0).sum()
        if actual_pos > 0:
            recall = true_pos / actual_pos
            code_recalls.append(recall.item())
    
    if code_recalls:
        code_metrics['median_code_recall'] = np.median(code_recalls)
        code_metrics['codes_with_high_recall'] = sum(1 for r in code_recalls if r > 0.5)
        code_metrics['min_code_recall'] = np.min(code_recalls)
        code_metrics['max_code_recall'] = np.max(code_recalls)
    else:
        code_metrics.update({'median_code_recall': 0, 'codes_with_high_recall': 0, 
                           'min_code_recall': 0, 'max_code_recall': 0})
    
    results['code_analysis'] = code_metrics
    
    # 4. Additional metrics
    results['total_samples'] = all_labels_tensor.shape[0]
    results['avg_true_codes_per_sample'] = all_labels_tensor.sum(dim=1).float().mean().item()
    
    return results

def print_detailed_analysis(results, n_codes=2869):
    """In phân tích chi tiết"""
    print("\n" + "="*70)
    print("🔍 DETAILED MODEL ANALYSIS")
    print("="*70)
    
    print("\n🎯 TOP-K RECALL:")
    for k in [5, 10, 20, 30]:
        key = f'recall@{k}'
        if key in results:
            print(f"   Recall@{k}: {results[key]:.4f}")
    
    print("\n📊 PREDICTION DISTRIBUTION:")
    code_analysis = results.get('code_analysis', {})
    print(f"   Avg codes predicted per sample: {code_analysis.get('avg_codes_per_sample', 0):.2f}")
    print(f"   Avg true codes per sample: {results.get('avg_true_codes_per_sample', 0):.2f}")
    print(f"   Unique codes predicted: {code_analysis.get('codes_with_predictions', 0)}/{n_codes}")
    print(f"   Codes with recall > 0.5: {code_analysis.get('codes_with_high_recall', 0)}")
    print(f"   Median code-wise recall: {code_analysis.get('median_code_recall', 0):.4f}")
    print(f"   Code recall range: [{code_analysis.get('min_code_recall', 0):.4f}, {code_analysis.get('max_code_recall', 0):.4f}]")
    
    print("\n🎚️  THRESHOLD SENSITIVITY:")
    print(f"   {'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print(f"   {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        prec = results.get(f'precision_micro_thresh{threshold}', 0)
        rec = results.get(f'recall_micro_thresh{threshold}', 0)
        f1 = results.get(f'f1_micro_thresh{threshold}', 0)
        print(f"   {threshold:<10} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    
    print(f"\n📈 SAMPLE STATS:")
    print(f"   Total samples evaluated: {results.get('total_samples', 0)}")
    
    print("="*70)

def calculate_class_weights(loader, n_codes, smooth=1.0):
    """Tính weights cho các classes hiếm - dùng cho weighted loss"""
    code_counts = torch.zeros(n_codes)
    total_samples = 0
    
    for _, labels in loader:
        code_counts += labels.sum(dim=0)
        total_samples += labels.shape[0]
    
    # Inverse frequency weighting với smoothing
    weights = total_samples / (code_counts + smooth)
    
    print(f"\n⚖️  CLASS WEIGHTS ANALYSIS:")
    print(f"   Most frequent code count: {code_counts.max().item()}")
    print(f"   Least frequent code count: {code_counts[code_counts > 0].min().item()}")
    print(f"   Unique codes in data: {(code_counts > 0).sum().item()}/{n_codes}")
    print(f"   Weight range: [{weights.min().item():.2f}, {weights.max().item():.2f}]")
    
    return weights

def run_complete_evaluation(model, train_loader, val_loader, n_codes):
    """Chạy đánh giá hoàn chỉnh với tất cả phân tích"""
    
    print("🚀 STARTING COMPLETE MODEL EVALUATION")
    print("="*70)
    
    # 1. Phân tích phân phối predictions
    dist_analysis = debug_predictions_distribution(model, val_loader, n_codes)
    
    # 2. Đánh giá toàn diện
    comprehensive_results = evaluate_comprehensive(model, val_loader)
    print_evaluation_results(comprehensive_results)
    
    # 3. Phân tích chi tiết
    detailed_results = evaluate_improved(model, val_loader, n_codes=n_codes)
    print_detailed_analysis(detailed_results, n_codes)
    
    # 4. Tính class weights (cho training improvement)
    class_weights = calculate_class_weights(train_loader, n_codes)
    
    # 5. Tóm tắt đánh giá
    print("\n📋 EVALUATION SUMMARY:")
    recall_30 = comprehensive_results.get('recall@30', 0)
    auc_roc = comprehensive_results.get('auc_roc_micro', 0)
    unique_predicted = dist_analysis.get('unique_codes_predicted', 0)
    
    print(f"   ✅ Top-30 Recall: {recall_30:.4f}")
    print(f"   ✅ AUC-ROC: {auc_roc:.4f}")
    print(f"   ⚠️  Unique codes predicted: {unique_predicted}/{n_codes} ({unique_predicted/n_codes*100:.1f}%)")
    
    if unique_predicted < n_codes * 0.1:  # Dưới 10% codes được predict
        print(f"   🚨 ISSUE: Model only predicts {unique_predicted} unique codes")
        print(f"   💡 SUGGESTION: Use class weighting or focal loss")
    
    return {
        'comprehensive': comprehensive_results,
        'detailed': detailed_results,
        'distribution': dist_analysis,
        'class_weights': class_weights
    }
