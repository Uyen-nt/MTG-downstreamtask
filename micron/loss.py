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


def micron_loss(logits, labels, dcm, lambda_reg=0.02):
    # BCE
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    # probability
    prob = torch.sigmoid(logits)
    prob_pair = prob.t() @ prob  # shape (vocab, vocab)

    # alignment loss
    struct_loss = torch.mean((prob_pair - dcm)**2)

    return bce + lambda_reg * struct_loss * 0.02

