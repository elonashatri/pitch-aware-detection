import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class RelationshipModule(nn.Module):
    """
    Graph Neural Network module for modeling relationships between musical elements
    """
    def __init__(
        self,
        node_feat_dim: int = 256,
        edge_feat_dim: int = 128,
        hidden_dim: int = 256,
        num_iterations: int = 3
    ):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        # Initial node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing modules
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_iterations)
        ])
        
        # Edge predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # Node classifier (refines node class predictions)
        self.node_classifier = nn.Linear(hidden_dim, node_feat_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        node_boxes: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the relationship module
        
        Args:
            node_features: Node features [N, node_feat_dim]
            node_boxes: Node bounding boxes [N, 4] in (x1, y1, x2, y2) format
            edge_indices: Edge indices [2, E]
            edge_features: Optional edge features [E, edge_feat_dim]
            
        Returns:
            updated_node_features: Updated node features [N, node_feat_dim]
            edge_scores: Predicted relationship scores [E, 1]
        """
        # Encode node features
        node_hidden = self.node_encoder(node_features)
        
        # If no edge features provided, create from node positions
        if edge_features is None:
            edge_features = self.create_edge_features(node_boxes, edge_indices)
        
        # Encode edge features
        edge_hidden = self.edge_encoder(edge_features)
        
        # Message passing iterations
        for message_layer in self.message_layers:
            node_hidden = message_layer(node_hidden, edge_indices, edge_hidden)
        
        # Predict relationship scores
        edge_scores = self.predict_edge_scores(node_hidden, edge_indices)
        
        # Update node features
        updated_node_features = self.node_classifier(node_hidden)
        
        return updated_node_features, edge_scores
    
    def create_edge_features(
        self,
        node_boxes: torch.Tensor,
        edge_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Create edge features from node positions
        
        Args:
            node_boxes: Node bounding boxes [N, 4] in (x1, y1, x2, y2) format
            edge_indices: Edge indices [2, E]
            
        Returns:
            edge_features: Edge features [E, edge_feat_dim]
        """
        src_idx, dst_idx = edge_indices
        
        # Convert boxes to center format
        boxes_cent = torch.zeros_like(node_boxes)
        boxes_cent[:, 0] = (node_boxes[:, 0] + node_boxes[:, 2]) / 2  # center_x
        boxes_cent[:, 1] = (node_boxes[:, 1] + node_boxes[:, 3]) / 2  # center_y
        boxes_cent[:, 2] = node_boxes[:, 2] - node_boxes[:, 0]        # width
        boxes_cent[:, 3] = node_boxes[:, 3] - node_boxes[:, 1]        # height
        
        # Get box features for source and destination nodes
        src_boxes = boxes_cent[src_idx]
        dst_boxes = boxes_cent[dst_idx]
        
        # Compute geometric features
        # 1. Relative positions (x, y)
        rel_pos = dst_boxes[:, :2] - src_boxes[:, :2]
        
        # 2. Log-normalized distance
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        log_dist = torch.log(distance + 1)
        
        # 3. Angle between centers
        angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]).unsqueeze(1)
        
        # 4. Area ratio
        src_area = src_boxes[:, 2] * src_boxes[:, 3]
        dst_area = dst_boxes[:, 2] * dst_boxes[:, 3]
        area_ratio = torch.log(dst_area / (src_area + 1e-8) + 1e-8).unsqueeze(1)
        
        # 5. IoU (may be 0 for non-overlapping boxes)
        iou = self.box_iou(node_boxes[src_idx], node_boxes[dst_idx]).unsqueeze(1)
        
        # 6. Relative scale
        width_ratio = torch.log(dst_boxes[:, 2] / (src_boxes[:, 2] + 1e-8) + 1e-8).unsqueeze(1)
        height_ratio = torch.log(dst_boxes[:, 3] / (src_boxes[:, 3] + 1e-8) + 1e-8).unsqueeze(1)
        
        # Concatenate all features
        edge_features = torch.cat([
            rel_pos, log_dist, torch.sin(angle), torch.cos(angle),
            area_ratio, iou, width_ratio, height_ratio
        ], dim=1)
        
        # Add positional encoding for better embedding
        pos_enc = self.positional_encoding(edge_features.shape[0], self.edge_feat_dim - edge_features.shape[1])
        pos_enc = pos_enc.to(edge_features.device)
        
        # Concatenate with positional encoding to reach edge_feat_dim
        edge_features = torch.cat([edge_features, pos_enc], dim=1)
        
        return edge_features
    
    def predict_edge_scores(
        self,
        node_hidden: torch.Tensor,
        edge_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict relationship scores between nodes
        
        Args:
            node_hidden: Node embeddings [N, hidden_dim]
            edge_indices: Edge indices [2, E]
            
        Returns:
            edge_scores: Predicted relationship scores [E, 1]
        """
        src_idx, dst_idx = edge_indices
        
        # Get node features for source and destination
        src_features = node_hidden[src_idx]
        dst_features = node_hidden[dst_idx]
        
        # Concatenate features
        edge_features = torch.cat([src_features, dst_features], dim=1)
        
        # Predict scores
        edge_scores = torch.sigmoid(self.edge_predictor(edge_features))
        
        return edge_scores
    
    @staticmethod
    def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between boxes
        
        Args:
            boxes1: First set of boxes [N, 4] in (x1, y1, x2, y2) format
            boxes2: Second set of boxes [N, 4] in (x1, y1, x2, y2) format
            
        Returns:
            iou: IoU values [N]
        """
        # Area of boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Intersection coordinates
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        
        # Width and height of intersection
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]
        
        # IoU
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-8)
        
        return iou
    
    @staticmethod
    def positional_encoding(n_positions: int, dim: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding
        
        Args:
            n_positions: Number of positions
            dim: Dimensionality of the encoding
            
        Returns:
            pos_enc: Positional encoding [n_positions, dim]
        """
        if dim % 2 != 0:
            dim += 1  # Ensure dim is even
            
        position = torch.arange(n_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pos_enc = torch.zeros(n_positions, dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc


class MessagePassingLayer(nn.Module):
    """
    Message passing layer for graph neural network
    """
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim)
        )
        
        # Node update
        self.node_update = nn.GRUCell(node_dim, node_dim)
        
        # Gate for edge importance
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: Node features [N, node_dim]
            edge_indices: Edge indices [2, E]
            edge_features: Edge features [E, edge_dim]
            
        Returns:
            updated_node_features: Updated node features [N, node_dim]
        """
        src_idx, dst_idx = edge_indices
        
        # Compute messages
        src_features = node_features[src_idx]
        dst_features = node_features[dst_idx]
        
        # Concatenate source, destination and edge features
        edge_inputs = torch.cat([src_features, dst_features, edge_features], dim=1)
        
        # Compute messages
        messages = self.message_mlp(edge_inputs)
        
        # Edge importance gating
        edge_importance = self.edge_gate(edge_features)
        messages = messages * edge_importance
        
        # Aggregate messages for each node
        aggregated_messages = torch.zeros_like(node_features)
        aggregated_messages.index_add_(0, dst_idx, messages)
        
        # Update node features
        updated_features = self.node_update(aggregated_messages, node_features)
        
        return updated_features