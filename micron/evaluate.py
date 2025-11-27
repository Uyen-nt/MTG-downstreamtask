import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, hamming_loss, jaccard_score, coverage_error, label_ranking_loss

def evaluate_all_metrics(model, loader, n_codes, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits[0]).squeeze(0)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels[0].cpu().numpy())

    all_logits = np.array(all_logits)   # (num_samples, n_codes)
    all_labels = np.array(all_labels)   # (num_samples, n_codes)

    # Sigmoid
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()

    # Apply threshold
    preds = (probs > threshold).astype(int)

    metrics = {}

    # Micro + Macro metrics
    metrics["precision_micro"] = precision_score(all_labels, preds, average='micro', zero_division=0)
    metrics["recall_micro"]    = recall_score(all_labels, preds, average='micro', zero_division=0)
    metrics["f1_micro"]        = f1_score(all_labels, preds, average='micro', zero_division=0)

    metrics["precision_macro"] = precision_score(all_labels, preds, average='macro', zero_division=0)
    metrics["recall_macro"]    = recall_score(all_labels, preds, average='macro', zero_division=0)
    metrics["f1_macro"]        = f1_score(all_labels, preds, average='macro', zero_division=0)

    # Jaccard Similarity
    metrics["jaccard"] = jaccard_score(all_labels, preds, average='samples')

    # Hamming Loss
    metrics["hamming_loss"] = hamming_loss(all_labels, preds)

    # ROC-AUC
    try:
        metrics["auc_micro"] = roc_auc_score(all_labels, probs, average='micro')
        metrics["auc_macro"] = roc_auc_score(all_labels, probs, average='macro')
    except:
        metrics["auc_micro"] = 0
        metrics["auc_macro"] = 0

    # Ranking metrics
    metrics["coverage_error"] = coverage_error(all_labels, probs)
    metrics["label_ranking_loss"] = label_ranking_loss(all_labels, probs)

    # Complexity of prediction
    metrics["avg_predicted_codes"] = preds.sum(axis=1).mean()
    metrics["avg_true_codes"] = all_labels.sum(axis=1).mean()

    return metrics


def print_metrics(metrics):
    print("\n======= MODEL EVALUATION METRICS =======\n")
    for k, v in metrics.items():
        print(f"{k:20s} : {v:.4f}")
    print("\n========================================\n")
