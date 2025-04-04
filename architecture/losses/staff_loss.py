import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class StafflineDetectionLoss(nn.Module):
    """
    Loss for staffline detection, combining heatmap and offset losses
    """
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        heatmap_pred: torch.Tensor,
        heatmap_target: torch.Tensor,
        offset_pred: torch.Tensor,
        offset_target: torch.Tensor,
        heatmap_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            heatmap_pred: Predicted heatmaps [B, 5, H, W]
            heatmap_target: Target heatmaps [B, 5, H, W]
            offset_pred: Predicted offsets [B, 10, H, W] (y-offset and height for each line)
            offset_target: Target offsets [B, 10, H, W]
            heatmap_mask: Optional mask for heatmap loss [B, 5, H, W]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Apply focal loss for heatmap prediction
        heatmap_loss = self.focal_heatmap_loss(heatmap_pred, heatmap_target, heatmap_mask)
        
        # Apply smooth L1 loss for offset prediction
        # Only apply to regions where stafflines exist
        if heatmap_mask is None:
            heatmap_mask = (heatmap_target > 0.5).float()
        
        # Expand mask for y-offset and height
        offset_mask = torch.cat([heatmap_mask, heatmap_mask], dim=1)
        
        # Compute offset loss only where stafflines exist
        offset_loss = F.smooth_l1_loss(
            offset_pred * offset_mask,
            offset_target * offset_mask,
            reduction='sum'
        )
        
        # Normalize by the number of staffline pixels
        num_staffline_pixels = torch.clamp(heatmap_mask.sum(), min=1.0)
        offset_loss = offset_loss / (2.0 * num_staffline_pixels)  # 2 values per line
        
        # Combined loss
        total_loss = self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss
        
        return total_loss, {
            'heatmap_loss': heatmap_loss,
            'offset_loss': offset_loss
        }
    
    def focal_heatmap_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Focal loss for heatmap prediction
        
        Args:
            pred: Predicted heatmaps [B, 5, H, W]
            target: Target heatmaps [B, 5, H, W]
            mask: Optional mask [B, 5, H, W]
            
        Returns:
            loss: Focal loss for heatmaps
        """
        # Apply sigmoid if predictions are not already in [0, 1]
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        
        # Focal loss parameters
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        
        # Compute binary cross entropy
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        # Positive samples get alpha weight, negative samples get (1-alpha)
        pos_loss = torch.log(pred + 1e-8) * torch.pow(1 - pred, gamma) * pos_inds
        neg_loss = torch.log(1 - pred + 1e-8) * torch.pow(pred, gamma) * neg_inds
        
        pos_loss = pos_loss * alpha
        neg_loss = neg_loss * (1 - alpha)
        
        # Apply mask if provided
        if mask is not None:
            pos_loss = pos_loss * mask
            neg_loss = neg_loss * mask
        
        # Sum all pixels
        num_pos = pos_inds.sum() if pos_inds.sum() > 0 else 1
        pos_loss = -pos_loss.sum() / num_pos
        neg_loss = -neg_loss.sum() / num_pos
        
        return pos_loss + neg_loss


class StaffCompletenessLoss(nn.Module):
    """
    Loss for ensuring staves have exactly 5 stafflines
    """
    def __init__(
        self,
        line_spacing_weight: float = 1.0,
        completeness_weight: float = 2.0
    ):
        super().__init__()
        
        self.line_spacing_weight = line_spacing_weight
        self.completeness_weight = completeness_weight
    
    def forward(
        self,
        stafflines: List[Dict],
        target_stafflines: List[Dict]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            stafflines: Predicted stafflines as a list of dictionaries
            target_stafflines: Target stafflines as a list of dictionaries
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        batch_size = len(stafflines)
        device = next(iter(stafflines[0]['staffs'][0]['stafflines'][0].values())).device if stafflines[0]['staffs'] else 'cpu'
        
        spacing_losses = []
        completeness_losses = []
        
        for b in range(batch_size):
            # 1. Check if staff has 5 lines
            pred_staffs = stafflines[b]['staffs']
            target_staffs = target_stafflines[b]['staffs']
            
            # Track number of staves detected vs. target
            pred_num_staffs = len(pred_staffs)
            target_num_staffs = len(target_staffs)
            
            # Completeness loss: penalize if number of staves doesn't match
            if pred_num_staffs != target_num_staffs:
                completeness_loss = torch.tensor(
                    abs(pred_num_staffs - target_num_staffs) / max(target_num_staffs, 1),
                    device=device
                )
            else:
                completeness_loss = torch.tensor(0.0, device=device)
            
            # 2. Check staffline spacing consistency
            spacing_loss = torch.tensor(0.0, device=device)
            
            if pred_staffs:
                for staff in pred_staffs:
                    lines = staff['stafflines']
                    
                    # Check number of stafflines
                    if len(lines) != 5:
                        # Penalize incomplete staffs
                        spacing_loss += abs(len(lines) - 5) / 5
                        continue
                    
                    # Sort stafflines by y-position
                    lines = sorted(lines, key=lambda x: x['y1'])
                    
                    # Calculate spacings between consecutive lines
                    spacings = []
                    for i in range(1, len(lines)):
                        spacing = lines[i]['y1'] - lines[i-1]['y1']
                        spacings.append(spacing)
                    
                    if spacings:
                        # Calculate consistency of spacing
                        mean_spacing = sum(spacings) / len(spacings)
                        deviations = [(s - mean_spacing) ** 2 for s in spacings]
                        std_spacing = (sum(deviations) / len(deviations)) ** 0.5 
                        
                        # Normalize by mean spacing
                        rel_std = std_spacing / (mean_spacing + 1e-8)
                        
                        # Add to spacing loss (penalize inconsistent spacing)
                        spacing_loss += min(rel_std, 1.0)  # Cap at 1.0
                
                # Average over all staves
                if pred_staffs:
                    spacing_loss = spacing_loss / len(pred_staffs)
            
            spacing_losses.append(spacing_loss)
            completeness_losses.append(completeness_loss)
        
        # Average losses across batch
        spacing_loss = torch.stack(spacing_losses).mean()
        completeness_loss = torch.stack(completeness_losses).mean()
        
        # Combined loss
        total_loss = (
            self.line_spacing_weight * spacing_loss + 
            self.completeness_weight * completeness_loss
        )
        
        return total_loss, {
            'spacing_loss': spacing_loss,
            'completeness_loss': completeness_loss
        }