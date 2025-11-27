import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1-pt)**self.gamma * bce
        return focal.mean()


def micron_loss(logits, labels, dcm, lambda_reg=0.01):
    # BCE loss
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    prob = torch.sigmoid(logits)   # (codes,)

    # flatten to pairwise
    prob_pair = torch.outer(prob, prob)   # shape (C,C)

    # mask chỉ apply regularization
    # với các cặp có co-occurrence thật
    mask = (dcm > 0).float()

    struct_loss = torch.sum(((prob_pair - dcm)**2) * mask) / (mask.sum() + 1e-8)

    return bce + lambda_reg * struct_loss

