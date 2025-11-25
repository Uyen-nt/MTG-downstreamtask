# retain_micron/evaluate.py
import torch
import numpy as np

def evaluate_topk_recall(model, loader, k=10):
    model.eval()
    total_hits = 0
    total_codes = 0

    with torch.no_grad():
        for visits, labels in loader:
            visits = [v for v in visits]           # list of patients
            logits = model(visits)                 # (n_codes,)
            pred_codes = torch.topk(logits, k)[1].cpu().numpy().tolist()

            # Duyệt từng bệnh nhân trong batch
            for i in range(labels.size(0)):
                true_codes = torch.where(labels[i] > 0.5)[0].cpu().numpy().tolist()
                hits = len(set(pred_codes) & set(true_codes))
                total_hits += hits
                total_codes += max(1, len(true_codes))

    return total_hits / total_codes if total_codes > 0 else 0
