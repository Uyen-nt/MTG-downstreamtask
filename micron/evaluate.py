import torch

def eval_recall_k(model, loader, k=30):
    model.eval()
    total_hits = 0
    total_codes = 0

    with torch.no_grad():
        for visits, labels in loader:
            visits = visits[0]
            labels = labels[0]

            logits = model(visits).squeeze(0)
            topk = torch.topk(logits, k=k).indices.cpu().tolist()

            true = set(torch.where(labels > 0.5)[0].cpu().tolist())

            total_hits += len(set(topk) & true)
            total_codes += len(true)

    recall = total_hits / total_codes if total_codes else 0
    print(f"Recall@{k} = {recall:.4f}")
    return recall
