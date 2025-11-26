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
