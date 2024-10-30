from mmengine.registry import MODELS
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


def size_weighted_loss(pred_bboxes: Tensor, gt_bboxes: Tensor, small_threshold=32*32, medium_threshold=96*96) -> Tensor:

    epsilon = 1e-6
    widths = torch.clamp(gt_bboxes[:, 2], min=epsilon)
    heights = torch.clamp(gt_bboxes[:, 3], min=epsilon)
    areas = widths * heights

    scale_weights = torch.ones_like(areas)
    scale_weights[areas < small_threshold] = 3.0
    scale_weights[(areas >= small_threshold) & (areas < medium_threshold)] = 1.5

    loss = torch.abs(pred_bboxes - gt_bboxes)
    weighted_loss = scale_weights.unsqueeze(1) * loss

    return weighted_loss.mean()

@MODELS.register_module()
class SizeWeightedLoss(nn.Module):
    def __init__(self, 
                 base_loss_fn: size_weighted_loss, 
                 small_threshold: int = 32*32, 
                 medium_threshold: int = 96*96, 
                 reduction: str = 'mean', 
                 ) -> None:
        super(SizeWeightedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold
        self.reduction = reduction

    def forward(self,
                pred_bboxes: Tensor,
                gt_bboxes: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        
        if weight is not None and not torch.any(weight > 0):
            return (pred_bboxes * weight).sum()

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        losses = self.base_loss_fn(pred_bboxes, gt_bboxes, self.small_threshold, self.medium_threshold)

        return losses
