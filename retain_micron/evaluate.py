# retain_micron/evaluate.py
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Top-K Recall
# =============================================================================
def evaluate_topk_recall(model, loader, k=30):
    model.eval()
    total_hits = 0
    total_codes = 0
    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            topk_preds = torch.topk(logits, k=k, dim=1).indices
            true_codes = [set(torch.where(label > 0.5)[0].cpu().tolist()) for label in labels]

            for pred_set, true_set in zip(topk_preds.cpu().tolist(), true_codes):
                pred_set = set(pred_set)
                total_hits += len(pred_set & true_set)
                total_codes += len(true_set) if true_set else 1

    return total_hits / total_codes if total_codes > 0 else 0.0


# =============================================================================
# 2. Phân tích phân phối dự đoán
# =============================================================================
def debug_predictions_distribution(model, loader, n_codes):
    model.eval()
    code_counts = torch.zeros(n_codes)
    per_sample_counts = []

    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()

            per_sample_counts.extend(predicted.sum(dim=1).cpu().tolist())
            code_counts += predicted.sum(dim=0).cpu()

    avg_per_sample = np.mean(per_sample_counts)
    min_per_sample = np.min(per_sample_counts)
    max_per_sample = np.max(per_sample_counts)
    unique_predicted = int((code_counts > 0).sum().item())
    top_codes = torch.topk(code_counts, min(10, unique_predicted)).indices.tolist()

    print(f"\nPREDICTION DISTRIBUTION ANALYSIS:")
    print(f"   Avg codes predicted per sample : {avg_per_sample:.2f}")
    print(f"   Min–Max codes predicted        : {min_per_sample}–{max_per_sample}")
    print(f"   Unique codes predicted         : {unique_predicted}/{n_codes} ({unique_predicted/n_codes*100:.1f}%)")
    print(f"   Top 10 most predicted codes    : {top_codes}")

    return {
        'avg_codes_per_sample': avg_per_sample,
        'unique_codes_predicted': unique_predicted,
        'top_predicted_codes': top_codes
    }


# =============================================================================
# 3. Đánh giá toàn diện + Threshold sensitivity
# =============================================================================
def evaluate_detailed(model, loader, n_codes, k_list=[5, 10, 20, 30]):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    probs_tensor = torch.sigmoid(logits_tensor)

    results = {}

    # Top-K Recall
    for k in k_list:
        results[f'recall@{k}'] = evaluate_topk_recall(model, loader, k=k)

    # Threshold sensitivity
    thresholds = [0.1, 0.3, 0.5, 0.7]
    print(f"\nTHRESHOLD SENSITIVITY (micro):")
    print(f" {'Thresh':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-" * 34)
    for th in thresholds:
        pred = (probs_tensor > th).float()
        p = precision_score(labels_tensor.view(-1), pred.view(-1), average='micro', zero_division=0)
        r = recall_score(labels_tensor.view(-1), pred.view(-1), average='micro', zero_division=0)
        f1 = f1_score(labels_tensor.view(-1), pred.view(-1), average='micro', zero_division=0)
        results[f'prec_th{th}'] = p
        results[f'rec_th{th}'] = r
        results[f'f1_th{th}'] = f1
        print(f" {th:<8} {p:.4f}   {r:.4f}   {f1:.4f}")

    # AUC-ROC
    try:
        auc_micro = roc_auc_score(labels_tensor.numpy(), probs_tensor.numpy(), average='micro')
        results['auc_micro'] = auc_micro
        print(f"\nMicro AUC-ROC: {auc_micro:.4f}")
    except:
        results['auc_micro'] = 0.0

    # Avg true codes
    avg_true = labels_tensor.sum(dim=1).float().mean().item()
    results['avg_true_codes'] = avg_true

    return results


# =============================================================================
# 4. In kết quả cuối cùng đẹp nhất (dùng trong run_retain_diagnosis.py)
# =============================================================================
def print_final_evaluation(model, val_loader, n_codes):
    print("\n" + "="*70)
    print("COMPLETE FINAL EVALUATION")
    print("="*70)

    # Phân phối dự đoán
    dist = debug_predictions_distribution(model, val_loader, n_codes)

    # Top-K Recall
    recall30 = evaluate_topk_recall(model, val_loader, k=30)
    recall20 = evaluate_topk_recall(model, val_loader, k=20)
    recall10 = evaluate_topk_recall(model, val_loader, k=10)
    recall5  = evaluate_topk_recall(model, val_loader, k=5)

    print(f"\nTOP-K RECALL:")
    print(f"   Recall@5  : {recall5:.4f}")
    print(f"   Recall@10 : {recall10:.4f}")
    print(f"   Recall@20 : {recall20:.4f}")
    print(f"   Recall@30 : {recall30:.4f}")

    # Chi tiết threshold + AUC
    detailed = evaluate_detailed(model, val_loader, n_codes, k_list=[30])

    print(f"\nFINAL SUMMARY:")
    print(f"   → Recall@30          : {recall30:.4f}")
    print(f"   → Unique codes       : {dist['unique_codes_predicted']}/{n_codes}")
    print(f"   → Avg predicted/sample: {dist['avg_codes_per_sample']:.2f}")
    print(f"   → Micro AUC-ROC      : {detailed.get('auc_micro', 0):.4f}")

    if dist['unique_codes_predicted'] < 100:
        print("   → Model đang bị collapse – cần giảm bias hoặc dùng Focal Loss mạnh hơn")
    elif dist['avg_codes_per_sample'] > 50:
        print("   → Model predict quá nhiều – tăng gamma Focal Loss hoặc giảm bias")
    else:
        print("   → Model đang ở trạng thái TỐT!")

    print("="*70)


# =============================================================================
# 5. Tính class weights
# =============================================================================
def calculate_class_weights(loader, n_codes, smooth=1.0):
    counts = torch.zeros(n_codes)
    total = 0
    for _, labels in loader:
        counts += labels.sum(dim=0)
        total += labels.size(0)
    weights = total / (counts + smooth)

    print(f"\nCLASS WEIGHTS ANALYSIS:")
    print(f"   Most frequent code : {counts.max().item():.0f}")
    print(f"   Least frequent code: {counts[counts>0].min().item():.0f}")
    print(f"   Unique codes in data: {(counts>0).sum().item()}/{n_codes}")
    print(f"   Weight range       : [{weights.min():.1f}, {weights.max():.1f}]")

    return weights
