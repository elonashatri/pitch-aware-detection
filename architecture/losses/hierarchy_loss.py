import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

from .detection_loss import FocalLoss, BoxRegressionLoss
from .staff_loss import StafflineDetectionLoss, StaffCompletenessLoss
from .relationship_loss import RelationshipLoss, RelationshipConsistencyLoss

class HierarchicalDetectionLoss(nn.Module):
    """
    Combined loss for hierarchical music notation detection
    """
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        class_weights: Optional[torch.Tensor] = None,
        lambda_det: float = 1.0,
        lambda_staff: float = 2.0,
        lambda_rel: float = 0.5
    ):
        super().__init__()
        
        self.lambda_det = lambda_det
        self.lambda_staff = lambda_staff
        self.lambda_rel = lambda_rel
        
        # Standard detection losses
        self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.box_loss = BoxRegressionLoss(beta=0.1)
        
        # Staffline-specific losses
        self.staffline_loss = StafflineDetectionLoss()
        self.staff_completeness_loss = StaffCompletenessLoss()
        
        # Relationship losses
        self.relationship_loss = RelationshipLoss()
        self.relationship_consistency_loss = RelationshipConsistencyLoss(class_names)
        
        # Class weights for handling imbalance
        self.class_weights = class_weights
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            predictions: Dictionary of predictions from the model
            targets: Dictionary of ground truth targets
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # 1. Standard detection losses for musical elements
        detection_loss = 0.0
        
        for scale in ['macro', 'mid', 'micro']:
            if scale in predictions and scale in targets:
                cls_pred = predictions[scale]['cls']
                cls_target = targets[scale]['cls']
                box_pred = predictions[scale]['box']
                box_target = targets[scale]['box']
                obj_weights = targets[scale]['weights'] if 'weights' in targets[scale] else None
                
                # Apply class weights if provided
                cls_loss = self.cls_loss(cls_pred, cls_target)
                box_loss = self.box_loss(box_pred, box_target, obj_weights)
                
                scale_loss = cls_loss + box_loss
                detection_loss += scale_loss
                
                # Add to loss dict
                loss_dict[f'{scale}_cls_loss'] = cls_loss
                loss_dict[f'{scale}_box_loss'] = box_loss
        
        # Normalize by number of scales
        num_scales = sum(1 for scale in ['macro', 'mid', 'micro'] if scale in predictions)
        if num_scales > 0:
            detection_loss /= num_scales
        
        # 2. Staffline detection loss
        staffline_loss = 0.0
        
        if 'staffline' in predictions and 'staffline' in targets:
            heatmap_pred = predictions['staffline']['heatmap']
            heatmap_target = targets['staffline']['heatmap']
            offset_pred = predictions['staffline']['offset']
            offset_target = targets['staffline']['offset']
            heatmap_mask = targets['staffline']['mask'] if 'mask' in targets['staffline'] else None
            
            sl_loss, sl_loss_dict = self.staffline_loss(
                heatmap_pred, heatmap_target, offset_pred, offset_target, heatmap_mask
            )
            
            staffline_loss += sl_loss
            loss_dict.update({f'staffline_{k}': v for k, v in sl_loss_dict.items()})
        
        # 3. Staff completeness loss
        staff_completeness_loss = 0.0
        
        if 'staffs' in predictions and 'staffs' in targets:
            sc_loss, sc_loss_dict = self.staff_completeness_loss(
                predictions['staffs'], targets['staffs']
            )
            
            staff_completeness_loss += sc_loss
            loss_dict.update({f'staff_{k}': v for k, v in sc_loss_dict.items()})
        
        # 4. Relationship losses
        relationship_loss = 0.0
        
        if 'relationships' in predictions and 'relationships' in targets:
            edge_scores = predictions['relationships']['edge_scores']
            edge_targets = targets['relationships']['edge_targets']
            node_features = predictions['relationships']['node_features']
            node_targets = targets['relationships']['node_targets']
            edge_indices = targets['relationships']['edge_indices']
            
            rel_loss, rel_loss_dict = self.relationship_loss(
                edge_scores, edge_targets, node_features, node_targets, edge_indices
            )
            
            # Consistency loss for relationships
            node_classes = predictions['relationships']['node_classes']
            cons_loss = self.relationship_consistency_loss(
                node_classes, edge_scores, edge_indices
            )
            
            relationship_loss += rel_loss + cons_loss
            loss_dict.update({f'rel_{k}': v for k, v in rel_loss_dict.items()})
            loss_dict['rel_consistency_loss'] = cons_loss
        
        # Combined loss
        total_loss = (
            self.lambda_det * detection_loss +
            self.lambda_staff * (staffline_loss + staff_completeness_loss) +
            self.lambda_rel * relationship_loss
        )
        
        loss_dict['detection_loss'] = detection_loss
        loss_dict['staffline_loss'] = staffline_loss
        loss_dict['staff_completeness_loss'] = staff_completeness_loss
        loss_dict['relationship_loss'] = relationship_loss
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict