import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection
    
    Original paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted probabilities [N, C] or [N]
            target: Target classes [N] or binary target [N]
            
        Returns:
            loss: Focal loss
        """
        if pred.dim() > 1:
            # Multi-class: use cross entropy first
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-ce_loss)
        else:
            # Binary case
            pred = torch.sigmoid(pred)
            pt = torch.where(target == 1, pred, 1 - pred)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha term
        if pred.dim() > 1:
            alpha_term = torch.ones_like(pt) * self.alpha
        else:
            alpha_term = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Calculate loss
        loss = alpha_term * focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class BoxRegressionLoss(nn.Module):
    """
    Smooth L1 Loss for bounding box regression
    """
    def __init__(
        self,
        beta: float = 1.0 / 9.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted box deltas [N, 4] or [batch_size, num_anchors*4, H, W]
            target: Target box deltas [N, 4] or [batch_size, num_anchors*4, H, W]
            weights: Optional weights [N] or [batch_size, num_anchors, H, W]
            
        Returns:
            loss: Smooth L1 loss
        """
        diff = torch.abs(pred - target)
        
        # Apply smooth L1 formula
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Apply weights if provided
        if weights is not None:
            if weights.dim() == 1:
                # [N] weights for [N, 4] boxes
                loss = loss * weights.unsqueeze(1)
            elif weights.dim() == 4 and pred.dim() == 4:
                # [B, A, H, W] weights for [B, A*4, H, W] boxes
                batch_size, num_anchors, height, width = weights.shape
                weights = weights.reshape(batch_size, num_anchors, 1, height, width)
                weights = weights.expand(batch_size, num_anchors, 4, height, width)
                weights = weights.reshape(batch_size, num_anchors*4, height, width)
                loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-balanced loss for addressing extreme class imbalance
    
    Based on "Class-Balanced Loss Based on Effective Number of Samples"
    https://arxiv.org/abs/1901.05555
    """
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        # Effective number of samples
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class).float())
        
        # Class weights
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)
        
        self.class_weights = weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pred: Predicted logits [N, C]
            target: Target classes [N]
            
        Returns:
            loss: Class-balanced loss
        """
        # Move class weights to the same device as predictions
        weights = self.class_weights.to(pred.device)
        
        # Get weights for each target
        target_weights = weights[target]
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Apply focal term
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        
        # Apply class weights
        loss = focal_term * ce_loss * target_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss