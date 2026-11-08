"""Equivalent gamma-zero focal loss without the fused kernel's 0 * inf backward.

SAM3's presence_gamma defaults to zero. The supplied Triton derivative computes
(1-p_t)**(gamma-1) even when gamma is zero; saturated correct predictions then
produce NaN. For gamma=0 the focal loss is exactly alpha-weighted BCE. Preserve
the original reduction/normalization and leave all gamma>0 calls untouched.
"""
from functools import wraps

import torch
import torch.nn.functional as F


def gamma_zero_focal_loss(inputs, targets, num_boxes, alpha=0.25,
                          loss_on_multimask=False, reduce=True):
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        logits = inputs.float()
        labels = targets.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        if alpha >= 0:
            loss = loss * (alpha * labels + (1 - alpha) * (1 - labels))
        if not reduce:
            return loss
        if loss_on_multimask:
            if loss.ndim != 4:
                raise ValueError('Multimask focal loss expects [N,M,H,W]')
            return loss.flatten(2).mean(-1) / num_boxes
        return loss.mean(1).sum() / num_boxes


def enable_stable_presence_loss():
    import sam3.train.loss.loss_fns as loss_fns
    original = loss_fns.sigmoid_focal_loss
    if getattr(original, '_stable_gamma_zero', False):
        return

    @wraps(original)
    def safe_focal(inputs, targets, num_boxes, alpha=0.25, gamma=2,
                   loss_on_multimask=False, reduce=True, triton=True):
        if float(gamma) == 0.0:
            if triton and not 0 <= alpha <= 1:
                raise RuntimeError(f'Alpha should be in [0,1], got {alpha}')
            return gamma_zero_focal_loss(inputs, targets, num_boxes, alpha,
                                         loss_on_multimask, reduce)
        return original(inputs, targets, num_boxes, alpha, gamma,
                        loss_on_multimask, reduce, triton)

    safe_focal._stable_gamma_zero = True
    loss_fns.sigmoid_focal_loss = safe_focal
