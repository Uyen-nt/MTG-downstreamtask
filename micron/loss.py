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


def micron_loss(logits, labels, dcm, dcm_force=1.0, lambda_reg=0.002):

    bce = F.binary_cross_entropy_with_logits(logits, labels)

    prob = torch.sigmoid(logits)
    prob_pair = torch.outer(prob, prob)

    # chỉ apply regularizer lên những cặp có co-occurrence thật
    mask = (dcm > 0).float()

    struct_loss = torch.sum(((prob_pair - dcm)**2) * mask) / (mask.sum() + 1e-8)

    return bce + dcm_force * lambda_reg * struct_loss

