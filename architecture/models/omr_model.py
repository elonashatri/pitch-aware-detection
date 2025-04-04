import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from .backbones import get_backbone
from .system_detector import SystemDetector, StaffDetector
from .staffline_detector import StafflineDetector
from .element_detector import MusicElementDetector
from .relationship import RelationshipModule

class HierarchicalOMRModel(nn.Module):
    """
    Hierarchical model for Optical Music Recognition
    """
    def __init__(
        self,
        config: Dict
    ):
        super().__init__()
        
        self.config = config
        
        # Create backbone
        self.backbone = get_backbone(
            config['backbone']['name'],
            pretrained=config['backbone']['pretrained'],
            out_channels=config['backbone']['out_channels']
        )
        
        # System detector
        self.system_detector = SystemDetector(
            in_channels=config['backbone']['out_channels'],
            hidden_channels=config['system_detector']['hidden_channels'],
            num_anchors=len(config['system_detector']['prior_aspect_ratio']),
            prior_aspect_ratio=config['system_detector']['prior_aspect_ratio']
        )
        
        # Staff detector
        self.staff_detector = StaffDetector(
            in_channels=config['backbone']['out_channels'],
            hidden_channels=config['staff_detector']['hidden_channels'],
            num_anchors=len(config['staff_detector']['prior_aspect_ratio']),
            prior_aspect_ratio=config['staff_detector']['prior_aspect_ratio']
        )
        
        # Staffline detector
        self.staffline_detector = StafflineDetector(
            in_channels=config['backbone']['out_channels'],
            hidden_channels=config['staffline_detector']['hidden_channels'],
            groups=config['staffline_detector']['groups']
        )
        
        # Element detector
        self.element_detector = MusicElementDetector(
            in_channels_list=[
                config['backbone']['out_channels'],
                config['backbone']['out_channels'],
                config['backbone']['out_channels']
            ],
            hidden_channels=config['element_detector']['hidden_channels'],
            class_groups=config['element_detector']['class_groups'],
            anchors_config=config['element_detector']['anchors_config']
        )
        
        # Relationship module
        self.relationship_module = RelationshipModule(
            node_feat_dim=config['relationship']['node_feat_dim'],
            edge_feat_dim=config['relationship']['edge_feat_dim'],
            hidden_dim=config['relationship']['hidden_dim'],
            num_iterations=config['relationship']['num_iterations']
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Dict] = None
    ) -> Dict:
        """
        Forward pass
        
        Args:
            images: Input images [B, C, H, W]
            targets: Optional targets for training
            
        Returns:
            Dictionary of outputs including detections and relationships
        """
        # Extract features
        features = self.backbone(images)
        
        # System detection
        system_cls, system_reg = self.system_detector(features['p5'])
        
        # Staff detection
        staff_cls, staff_reg = self.staff_detector(features['p4'])
        
        # Staffline detection
        staffline_heatmaps, staffline_offsets = self.staffline_detector(features['p3'])
        
        # Element detection
        element_results = self.element_detector(features)
        
        # Collect outputs
        outputs = {
            'system_cls': system_cls,
            'system_reg': system_reg,
            'staff_cls': staff_cls,
            'staff_reg': staff_reg,
            'staffline_heatmaps': staffline_heatmaps,
            'staffline_offsets': staffline_offsets,
            'element_results': element_results
        }
        
        # Inference mode: decode predictions
        if not self.training:
            # Get feature sizes for anchor generation
            feature_sizes = {
                'p2': features['p2'].shape[2:],
                'p3': features['p3'].shape[2:],
                'p4': features['p4'].shape[2:],
                'p5': features['p5'].shape[2:],
            }
            
            # Image size
            image_size = images.shape[2:]
            
            # Decode predictions
            outputs.update(
                self._decode_predictions(outputs, feature_sizes, image_size)
            )
            
            # Relationship modeling
            if 'elements' in outputs and outputs['elements']:
                # Collect node features and positions
                node_features = []
                node_boxes = []
                node_classes = []
                
                for element in outputs['elements']:
                    # Get element features
                    feat_level = element['feat_level']
                    x_idx = min(int(element['x_idx']), features[feat_level].shape[3] - 1)
                    y_idx = min(int(element['y_idx']), features[feat_level].shape[2] - 1)
                    
                    # Extract features
                    feat = features[feat_level][0, :, y_idx, x_idx]
                    node_features.append(feat)
                    
                    # Get box
                    box = torch.tensor(element['bbox'], device=images.device)
                    node_boxes.append(box)
                    
                    # Get class
                    cls = element['class_id']
                    node_classes.append(cls)
                
                if node_features:
                    # Convert to tensors
                    node_features = torch.stack(node_features)
                    node_boxes = torch.stack(node_boxes)
                    node_classes = torch.tensor(node_classes, device=images.device)
                    
                    # Create potential edges
                    edge_indices = []
                    for i in range(len(node_features)):
                        for j in range(len(node_features)):
                            if i != j:
                                edge_indices.append((i, j))
                    
                    if edge_indices:
                        edge_indices = torch.tensor(edge_indices, device=images.device).t()
                        
                        # Predict relationships
                        updated_features, edge_scores = self.relationship_module(
                            node_features, node_boxes, edge_indices
                        )
                        
                        # Store relationship predictions
                        outputs['relationships'] = {
                            'node_features': updated_features,
                            'node_classes': node_classes,
                            'edge_indices': edge_indices,
                            'edge_scores': edge_scores
                        }
        
        # Training mode: compute losses
        if self.training and targets is not None:
            # Compute losses
            losses = self._compute_losses(outputs, targets)
            outputs.update(losses)
        
        return outputs
    
    def _decode_predictions(
        self,
        outputs: Dict,
        feature_sizes: Dict[str, Tuple[int, int]],
        image_size: Tuple[int, int]
    ) -> Dict:
        """
        Decode raw predictions to bounding boxes and classes
        
        Args:
            outputs: Raw prediction outputs
            feature_sizes: Sizes of feature maps
            image_size: Size of input image
            
        Returns:
            Dictionary of decoded predictions
        """
        decoded = {}
        device = next(iter(outputs.values())).device
        
        # Decode system predictions
        system_cls = outputs['system_cls']
        system_reg = outputs['system_reg']
        
        # Get system anchors
        system_anchors = self.system_detector.get_anchors(
            feature_sizes['p5'], image_size, device
        )
        
        # Decode system boxes
        systems = self._decode_boxes(
            system_cls, system_reg, system_anchors,
            self.config['inference']['system_score_thresh'],
            self.config['inference']['system_nms_thresh']
        )
        
        # Decode staff predictions
        staff_cls = outputs['staff_cls']
        staff_reg = outputs['staff_reg']
        
        # Get staff anchors
        staff_anchors = self.staff_detector.get_anchors(
            feature_sizes['p4'], image_size, device
        )
        
        # Decode staff boxes
        staves = self._decode_boxes(
            staff_cls, staff_reg, staff_anchors,
            self.config['inference']['staff_score_thresh'],
            self.config['inference']['staff_nms_thresh']
        )
        
        # Decode staffline predictions
        staffline_heatmaps = outputs['staffline_heatmaps']
        staffline_offsets = outputs['staffline_offsets']
        
        # Decode stafflines
        stafflines = self.staffline_detector.decode_stafflines(
            staffline_heatmaps,
            staffline_offsets,
            self.config['inference']['staffline_thresh']
        )
        
        # Decode element predictions
        elements = []
        
        for scale, (cls_scores, box_preds) in outputs['element_results'].items():
            # Get anchors for this scale
            scale_map = {
                'macro': 'p5',
                'mid': 'p4',
                'micro': 'p3'
            }
            feat_level = scale_map[scale]
            
            anchors = self.element_detector.get_anchors(
                {'scale': feature_sizes[feat_level]},
                image_size,
                device
            )[scale]
            
            # Decode elements
            scale_elements = self._decode_elements(
                cls_scores, box_preds, anchors,
                self.config['inference']['element_score_thresh'],
                self.config['inference']['element_nms_thresh'],
                self.config['element_detector']['class_groups'][scale],
                feat_level
            )
            
            elements.extend(scale_elements)
        
        # Sort elements by score
        elements.sort(key=lambda x: x['score'], reverse=True)
        
        decoded['systems'] = systems
        decoded['staves'] = staves
        decoded['stafflines'] = stafflines[0] if stafflines else {}
        decoded['elements'] = elements
        
        return decoded
    
    def _decode_boxes(
        self,
        cls_scores: torch.Tensor,
        box_preds: torch.Tensor,
        anchors: torch.Tensor,
        score_threshold: float,
        nms_threshold: float
    ) -> List[Dict]:
        """
        Decode box predictions
        
        Args:
            cls_scores: Classification scores [B, num_anchors, H, W]
            box_preds: Box predictions [B, num_anchors*4, H, W]
            anchors: Anchors [num_anchors*H*W, 4]
            score_threshold: Score threshold
            nms_threshold: NMS threshold
            
        Returns:
            List of decoded boxes
        """
        # Get scores and boxes
        batch_size, num_anchors, height, width = cls_scores.shape
        
        # Reshape cls_scores to [B, num_anchors*H*W]
        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(batch_size, -1)
        
        # Reshape box_preds to [B, num_anchors*H*W, 4]
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # Apply sigmoid to cls_scores
        scores = torch.sigmoid(cls_scores)
        
        # Convert deltas to boxes
        pred_boxes = self._delta2box(box_preds.reshape(-1, 4), anchors)
        pred_boxes = pred_boxes.reshape(batch_size, -1, 4)
        
        # Only process first batch for now
        scores = scores[0]
        boxes = pred_boxes[0]
        
        # Filter by score threshold
        keep = scores > score_threshold
        scores = scores[keep]
        boxes = boxes[keep]
        
        # Apply NMS
        keep = self._nms(boxes, scores, nms_threshold)
        scores = scores[keep]
        boxes = boxes[keep]
        
        # Create output dictionaries
        results = []
        for score, box in zip(scores.cpu().numpy(), boxes.cpu().numpy()):
            # Convert box from (x1, y1, x2, y2) to (x, y, w, h)
            x1, y1, x2, y2 = box
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            
            results.append({
                'bbox': [x, y, w, h],
                'score': float(score)
            })
        
        return results
    
    def _decode_elements(
        self,
        cls_scores: torch.Tensor,
        box_preds: torch.Tensor,
        anchors: torch.Tensor,
        score_threshold: float,
        nms_threshold: float,
        class_indices: List[int],
        feat_level: str
    ) -> List[Dict]:
        """
        Decode element predictions
        
        Args:
            cls_scores: Classification scores [B, num_anchors*(num_classes+1), H, W]
            box_preds: Box predictions [B, num_anchors*4, H, W]
            anchors: Anchors [num_anchors*H*W, 4]
            score_threshold: Score threshold
            nms_threshold: NMS threshold
            class_indices: Class indices for this scale
            feat_level: Feature level name
            
        Returns:
            List of decoded elements
        """
        # Get dimensions
        batch_size = cls_scores.shape[0]
        num_anchors = len(self.config['element_detector']['anchors_config'][feat_level.replace('p', 'scale')]['ratios'])
        height, width = cls_scores.shape[2], cls_scores.shape[3]
        num_classes = len(class_indices) + 1  # Add background class
        
        # Reshape cls_scores to [B, num_anchors*H*W, num_classes]
        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes)
        
        # Reshape box_preds to [B, num_anchors*H*W, 4]
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # Apply softmax to cls_scores
        scores = torch.softmax(cls_scores, dim=2)
        
        # Calculate anchor indices for feature locations
        anchor_indices = torch.arange(num_anchors * height * width, device=cls_scores.device)
        feature_y = (anchor_indices // (num_anchors * width)) % height
        feature_x = (anchor_indices // num_anchors) % width
        
        # Convert deltas to boxes
        pred_boxes = self._delta2box(box_preds.reshape(-1, 4), anchors)
        pred_boxes = pred_boxes.reshape(batch_size, -1, 4)
        
        # Only process first batch for now
        scores = scores[0]
        boxes = pred_boxes[0]
        
        # Process each class
        results = []
        
        for cls_idx in range(1, num_classes):  # Skip background class
            # Get class-specific scores
            cls_scores = scores[:, cls_idx]
            
            # Filter by score threshold
            keep = cls_scores > score_threshold
            if not keep.any():
                continue
                
            cls_scores = cls_scores[keep]
            cls_boxes = boxes[keep]
            cls_anchor_indices = anchor_indices[keep]
            cls_feature_y = feature_y[keep]
            cls_feature_x = feature_x[keep]
            
            # Apply NMS
            keep = self._nms(cls_boxes, cls_scores, nms_threshold)
            cls_scores = cls_scores[keep]
            cls_boxes = cls_boxes[keep]
            cls_anchor_indices = cls_anchor_indices[keep]
            cls_feature_y = cls_feature_y[keep]
            cls_feature_x = cls_feature_x[keep]
            
            # Create output dictionaries
            orig_cls_idx = class_indices[cls_idx - 1]  # Convert back to original class index
            
            for score, box, y_idx, x_idx in zip(
                cls_scores.cpu().numpy(),
                cls_boxes.cpu().numpy(),
                cls_feature_y.cpu().numpy(),
                cls_feature_x.cpu().numpy()
            ):
                # Convert box from (x1, y1, x2, y2) to (x, y, w, h)
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                results.append({
                    'bbox': [x, y, w, h],
                    'class_id': int(orig_cls_idx),
                    'score': float(score),
                    'feat_level': feat_level,
                    'y_idx': int(y_idx),
                    'x_idx': int(x_idx)
                })
        
        return results
    
    def _delta2box(
        self,
        deltas: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert box deltas to boxes
        
        Args:
            deltas: Box deltas [N, 4] in (dx, dy, dw, dh) format
            anchors: Anchors [N, 4] in (x1, y1, x2, y2) format
            
        Returns:
            boxes: Decoded boxes [N, 4] in (x1, y1, x2, y2) format
        """
        # Convert anchors to (cx, cy, w, h) format
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        
        # Apply deltas
        dx, dy, dw, dh = deltas.unbind(1)
        
        # Clamp dw and dh to avoid explosion
        dw = torch.clamp(dw, max=4.0)
        dh = torch.clamp(dh, max=4.0)
        
        # Get predicted box center and size
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        # Convert back to (x1, y1, x2, y2) format
        pred_x1 = pred_ctr_x - 0.5 * pred_w
        pred_y1 = pred_ctr_y - 0.5 * pred_h
        pred_x2 = pred_ctr_x + 0.5 * pred_w
        pred_y2 = pred_ctr_y + 0.5 * pred_h
        
        return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    
    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Non-maximum suppression
        
        Args:
            boxes: Boxes [N, 4] in (x1, y1, x2, y2) format
            scores: Box scores [N]
            threshold: IoU threshold
            
        Returns:
            keep: Indices of boxes to keep
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.int64, device=boxes.device)
            
        # Calculate areas
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort boxes by score
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            # Pick the box with highest score
            i = order[0].item()
            keep.append(i)
            
            # Exit if no boxes left
            if order.numel() == 1:
                break
                
            # Get remaining boxes
            order = order[1:]
            
            # Calculate IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order])
            yy1 = torch.max(y1[i], y1[order])
            xx2 = torch.min(x2[i], x2[order])
            yy2 = torch.min(y2[i], y2[order])
            
            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order] - inter)
            
            # Keep boxes with IoU below threshold
            idx = (iou <= threshold).nonzero().squeeze()
            if idx.numel() == 0:
                break
                
            order = order[idx]
        
        return torch.tensor(keep, device=boxes.device)
    
    def _compute_losses(
        self,
        outputs: Dict,
        targets: Dict
    ) -> Dict:
        """
        Compute losses for training
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # This would be implemented with your hierarchical loss functions
        # For now, return a placeholder
        return {'loss': torch.tensor(0.0, device=outputs['system_cls'].device)}