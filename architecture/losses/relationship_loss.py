import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class RelationshipLoss(nn.Module):
    """
    Loss for musical element relationships, combining classification and consistency losses
    """
    def __init__(
        self,
        relationship_weight: float = 1.0,
        consistency_weight: float = 0.5,
        pos_weight: float = 2.0
    ):
        super().__init__()
        
        self.relationship_weight = relationship_weight
        self.consistency_weight = consistency_weight
        self.pos_weight = pos_weight
    
    def forward(
        self,
        edge_scores: torch.Tensor,
        edge_targets: torch.Tensor,
        node_features: torch.Tensor,
        node_targets: torch.Tensor,
        edge_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            edge_scores: Predicted edge scores [E, 1]
            edge_targets: Target edge labels [E]
            node_features: Node features [N, C]
            node_targets: Node target classes [N]
            edge_indices: Edge indices [2, E]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Relationship classification loss
        relationship_loss = F.binary_cross_entropy_with_logits(
            edge_scores.squeeze(-1),
            edge_targets.float(),
            pos_weight=torch.tensor(self.pos_weight, device=edge_scores.device)
        )
        
        # Relationship consistency loss - penalize when connected nodes have incompatible classes
        consistency_loss = self.compute_consistency_loss(
            node_features, node_targets, edge_indices, edge_targets
        )
        
        # Combined loss
        total_loss = (
            self.relationship_weight * relationship_loss +
            self.consistency_weight * consistency_loss
        )
        
        return total_loss, {
            'relationship_loss': relationship_loss,
            'consistency_loss': consistency_loss
        }
    
    def compute_consistency_loss(
        self,
        node_features: torch.Tensor,
        node_targets: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss between connected nodes
        
        Args:
            node_features: Node features [N, C]
            node_targets: Node target classes [N]
            edge_indices: Edge indices [2, E]
            edge_targets: Edge target labels [E]
            
        Returns:
            consistency_loss: Loss penalizing incompatible connections
        """
        # Only consider positive edges (true relationships)
        pos_edges = edge_targets > 0.5
        if not pos_edges.any():
            return torch.tensor(0.0, device=node_features.device)
        
        # Get source and destination nodes for positive edges
        src_idx, dst_idx = edge_indices[:, pos_edges]
        
        # Get node features and targets
        src_features = node_features[src_idx]
        dst_features = node_features[dst_idx]
        
        src_targets = node_targets[src_idx]
        dst_targets = node_targets[dst_idx]
        
        # Compute feature similarity
        feature_sim = F.cosine_similarity(src_features, dst_features, dim=1)
        
        # Compute target compatibility based on music notation rules
        # For simplicity, we'll use same/different class as a proxy for compatibility
        target_compatibility = (src_targets == dst_targets).float()
        
        # Consistency loss: features should be similar when targets are compatible
        consistency_loss = F.binary_cross_entropy(
            (feature_sim + 1) / 2,  # Map from [-1, 1] to [0, 1]
            target_compatibility
        )
        
        return consistency_loss


class RelationshipConsistencyLoss(nn.Module):
    """
    Loss for ensuring music notation relationship consistency
    
    This enforces domain-specific constraints such as:
    - Noteheads should connect to stems
    - Stems should connect to beams or flags
    - etc.
    """
    def __init__(
        self,
        class_names: List[str],
        relationship_rules: Dict[str, List[str]] = None
    ):
        super().__init__()
        
        self.class_names = class_names
        
        # Default relationship rules if not provided
        if relationship_rules is None:
            relationship_rules = {
                'noteheadBlack': ['stem', 'beam', 'kStaffLine', 'slur', 'tie', 'accidentalSharp', 'accidentalFlat'],
                'noteheadHalf': ['stem', 'kStaffLine', 'slur', 'tie', 'accidentalSharp', 'accidentalFlat'],
                'stem': ['noteheadBlack', 'noteheadHalf', 'beam', 'flag8thUp', 'flag8thDown'],
                'beam': ['stem', 'noteheadBlack'],
                'slur': ['noteheadBlack', 'noteheadHalf'],
                'tie': ['noteheadBlack', 'noteheadHalf'],
                'accidentalSharp': ['noteheadBlack', 'noteheadHalf', 'kStaffLine'],
                'accidentalFlat': ['noteheadBlack', 'noteheadHalf', 'kStaffLine'],
                'flag8thUp': ['stem'],
                'flag8thDown': ['stem']
            }
        
        self.relationship_rules = relationship_rules
        
        # Create inverse mapping from class names to indices
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Create adjacency matrix of valid relationships
        self.valid_adjacency = torch.zeros(len(class_names), len(class_names))
        
        for src_class, dst_classes in relationship_rules.items():
            if src_class in self.class_to_idx:
                src_idx = self.class_to_idx[src_class]
                
                for dst_class in dst_classes:
                    if dst_class in self.class_to_idx:
                        dst_idx = self.class_to_idx[dst_class]
                        self.valid_adjacency[src_idx, dst_idx] = 1.0
    
    def forward(
        self,
        node_classes: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_classes: Predicted node classes [N]
            edge_scores: Predicted edge scores [E, 1]
            edge_indices: Edge indices [2, E]
            
        Returns:
            consistency_loss: Loss penalizing invalid relationships
        """
        device = node_classes.device
        self.valid_adjacency = self.valid_adjacency.to(device)
        
        # Get edge classes
        src_idx, dst_idx = edge_indices
        src_classes = node_classes[src_idx]
        dst_classes = node_classes[dst_idx]
        
        # Check if relationships are valid according to rules
        valid_edges = torch.zeros(edge_indices.shape[1], device=device)
        
        for i, (src_class, dst_class) in enumerate(zip(src_classes, dst_classes)):
            if src_class < self.valid_adjacency.shape[0] and dst_class < self.valid_adjacency.shape[1]:
                valid_edges[i] = self.valid_adjacency[src_class, dst_class]
        
        # Edge scores should be high for valid edges and low for invalid edges
        edge_probs = torch.sigmoid(edge_scores).squeeze(-1)
        
        # Binary cross entropy with valid_edges as target
        consistency_loss = F.binary_cross_entropy(
            edge_probs,
            valid_edges
        )
        
        return consistency_loss