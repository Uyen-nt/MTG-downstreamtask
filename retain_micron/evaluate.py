# retain_micron/evaluate.py
import torch
import numpy as np

def evaluate_topk_recall(model, loader, k=10):
    model.eval()
    all_hits = 0
    all_total = 0

    with torch.no_grad():
        for visits, labels in loader:
            visits = [v for v in visits]
            logits = model(visits)                               # (n_codes,)
            pred_codes = torch.topk(logits, k)[1].cpu().numpy()

            # Duyệt từng bệnh nhân trong batch
            for i in range(labels.size(0)):
                true_codes = torch.where(labels[i] > 0.5)[0].cpu().numpy()
                hits = len(set(pred_codes) & set(true_codes))
                all_hits += hits
                all_total += max(1, len(true_codes))

    return all_hits / all_total
