# retain_micron/evaluate.py
import torch
import numpy as np

def evaluate_topk_recall(model, loader, n_codes, k=10):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for visits, labels in loader:
            visits = [v for v in visits]
            logits = model(visits, n_codes)
            pred_codes = torch.topk(logits, k=k)[1].cpu().numpy().tolist()
            true_codes = torch.where(labels > 0.5)[1].cpu().numpy().tolist()
            
            all_preds.append(pred_codes)
            all_trues.append(true_codes)

    recalls = []
    for p, t in zip(all_preds, all_trues):
        hits = len(set(p) & set(t))
        recalls.append(hits / max(1, len(t)))

    return np.mean(recalls)
