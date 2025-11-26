# retain_micron/evaluate.py
import torch
import numpy as np

def evaluate_topk_recall(model, loader, k=10):
    model.eval()
    total_hits = 0
    total_codes = 0

    with torch.no_grad():
        for visits, labels in loader:
            logits = model(visits)                              # (B, n_codes) hoặc (n_codes,)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)                    # đảm bảo (B, n_codes)

            # Lấy top-k cho TỪNG bệnh nhân trong batch
            topk_indices = torch.topk(logits, k=k, dim=1)[1]    # (B, k)
            pred_codes_batch = topk_indices.cpu().tolist()      # → list of list int

            for i in range(labels.size(0)):
                pred_codes = pred_codes_batch[i]                # list int
                true_codes = torch.where(labels[i] > 0.5)[0]
                true_codes = true_codes[true_codes < n_codes].cpu().tolist()
                
                hits = len(set(pred_codes) & set(true_codes))
                total_hits += hits
                total_codes += len(true_codes) if len(true_codes) > 0 else 1

    return total_hits / total_codes
