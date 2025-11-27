import torch
import torch.nn as nn
import torch.nn.functional as F

def micron_loss(logits, labels, dcm, dcm_force=1.0, class_weights=None, lambda_reg=0.002):
    if class_weights is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, weight=class_weights, reduction='none')
        bce = bce.mean()
    else:
        bce = F.binary_cross_entropy_with_logits(logits, labels)

    prob = torch.sigmoid(logits)
    prob_pair = torch.outer(prob, prob)

    # chỉ apply regularizer lên những cặp có co-occurrence thật
    mask = (dcm > 0).float()

    struct_loss = torch.sum(((prob_pair - dcm)**2) * mask) / (mask.sum() + 1e-8)

    return bce + dcm_force * lambda_reg * struct_loss

