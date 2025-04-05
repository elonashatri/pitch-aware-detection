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
        
        # # Get the model config
        
        # model_config = config['model']
            # Determine if we have the model config directly or nested
        if 'backbone' in config:
            # The config is already the model section
            model_config = config
        elif 'model' in config and 'backbone' in config['model']:
            # The config has a nested model section
            model_config = config['model']
        else:
            raise ValueError("Could not find backbone configuration in the provided config")
        

        # Create backbone
        self.backbone = get_backbone(
        model_config['backbone']['name'],
        pretrained=model_config['backbone']['pretrained'],
        out_channels=model_config['backbone']['out_channels']
        )
        
        # System detector
        self.system_detector = SystemDetector(
            in_channels=model_config['backbone']['out_channels'],
            hidden_channels=model_config['system_detector']['hidden_channels'],
            num_anchors=len(model_config['system_detector']['prior_aspect_ratio']),
            prior_aspect_ratio=model_config['system_detector']['prior_aspect_ratio']
        )
    
        
        # Staff detector
        self.staff_detector = StaffDetector(
            in_channels=model_config['backbone']['out_channels'],
            hidden_channels=model_config['staff_detector']['hidden_channels'],
            num_anchors=len(model_config['staff_detector']['prior_aspect_ratio']),
            prior_aspect_ratio=model_config['staff_detector']['prior_aspect_ratio']
        )
        
        # Staffline detector
        self.staffline_detector = StafflineDetector(
            in_channels=model_config['backbone']['out_channels'],
            hidden_channels=model_config['staffline_detector']['hidden_channels'],
            groups=model_config['staffline_detector']['groups']
        )
        
        # Element detector
        self.element_detector = MusicElementDetector(
            in_channels_list=[
                model_config['backbone']['out_channels'],
                model_config['backbone']['out_channels'],
                model_config['backbone']['out_channels']
            ],
            hidden_channels=model_config['element_detector']['hidden_channels'],
            class_groups=model_config['element_detector']['class_groups'],
            anchors_config=model_config['element_detector']['anchors_config']
        )
        
        # Relationship module
        self.relationship_module = RelationshipModule(
            node_feat_dim=model_config['relationship']['node_feat_dim'],
            edge_feat_dim=model_config['relationship']['edge_feat_dim'],
            hidden_dim=model_config['relationship']['hidden_dim'],
            num_iterations=model_config['relationship']['num_iterations']
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
            # Relationship modeling
            if not self.training and 'elements' in outputs and outputs['elements']:
                # Collect node features and positions
                node_features = []
                node_boxes = []
                node_classes = []
                
                for element in outputs['elements']:
                    # Get feature level and box
                    feat_level = element.get('feat_level', 'p3')  # Default to p3 if missing
                    
                    # Check if we have x_idx and y_idx in the element
                    if 'x_idx' in element and 'y_idx' in element:
                        # Get element features from the feature map
                        x_idx = min(int(element['x_idx']), features[feat_level].shape[3] - 1)
                        y_idx = min(int(element['y_idx']), features[feat_level].shape[2] - 1)
                        
                        # Extract features
                        feat = features[feat_level][0, :, y_idx, x_idx]
                    else:
                        # No specific location, use center of bounding box to estimate
                        x, y, w, h = element['bbox']
                        img_h, img_w = images.shape[2:]
                        
                        # Calculate feature map position
                        feat_h, feat_w = features[feat_level].shape[2:]
                        feat_x = int(x * feat_w / img_w)
                        feat_y = int(y * feat_h / img_h)
                        
                        # Clamp to valid range
                        feat_x = max(0, min(feat_x, feat_w - 1))
                        feat_y = max(0, min(feat_y, feat_h - 1))
                        
                        # Extract features
                        feat = features[feat_level][0, :, feat_y, feat_x]
                    
                    # Add to node features
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
            total_loss, loss_dict = self._compute_losses(outputs, targets)
            # Add losses to outputs
            outputs['loss'] = total_loss
            outputs.update(loss_dict)  # This should update the dictionary correctly
            
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
        
        # Get inference thresholds with defaults
        inference_config = {}
        if 'inference' in self.config:
            inference_config = self.config['inference']
        
        # Set default thresholds
        system_score_thresh = inference_config.get('system_score_thresh', 0.5)
        system_nms_thresh = inference_config.get('system_nms_thresh', 0.5)
        staff_score_thresh = inference_config.get('staff_score_thresh', 0.5)
        staff_nms_thresh = inference_config.get('staff_nms_thresh', 0.5)
        staffline_thresh = inference_config.get('staffline_thresh', 0.5)
        element_score_thresh = inference_config.get('element_score_thresh', 0.3)
        element_nms_thresh = inference_config.get('element_nms_thresh', 0.5)

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
            system_score_thresh,
            system_nms_thresh
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
            staff_score_thresh,
            staff_nms_thresh
        )
        
        # Decode staffline predictions
        staffline_heatmaps = outputs['staffline_heatmaps']
        staffline_offsets = outputs['staffline_offsets']
        
        # Decode stafflines
        stafflines = self.staffline_detector.decode_stafflines(
            staffline_heatmaps,
            staffline_offsets,
            staffline_thresh
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
                element_score_thresh,
                element_nms_thresh,
                self.config['model']['element_detector']['class_groups'][scale],
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
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(batch_size, height * width * num_anchors, 4)
        
        # Apply sigmoid to cls_scores
        scores = torch.sigmoid(cls_scores)
        
        # Debugging output - print shape information
        print(f"box_preds shape: {box_preds.shape}, anchors shape: {anchors.shape}")
        
        # Make sure we're only using the correct number of predictions per batch
        # We're only dealing with the first batch for now
        batch_box_preds = box_preds[0, :anchors.shape[0], :]
        
        # Convert deltas to boxes for the first batch only
        pred_boxes = self._delta2box(batch_box_preds, anchors)
        
        # Only process first batch scores
        scores = scores[0, :anchors.shape[0]]
        
        # Filter by score threshold
        keep = scores > score_threshold
        if keep.sum() == 0:
            print(f"Warning: No predictions passed the score threshold of {score_threshold}")
            return []
            
        scores = scores[keep]
        boxes = pred_boxes[keep]
        
        # Apply NMS
        keep = self._nms(boxes, scores, nms_threshold)
        if keep.numel() == 0:
            return []
            
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
        """
        # Get dimensions
        batch_size = cls_scores.shape[0]
        height, width = cls_scores.shape[2], cls_scores.shape[3]
        
        # Print original tensor shapes for debugging
        print(f"Original shapes - cls_scores: {cls_scores.shape}, box_preds: {box_preds.shape}")
        
        # Calculate the total number of elements
        total_cls_elements = cls_scores.numel()
        total_box_elements = box_preds.numel()
        
        # Calculate number of elements per batch
        cls_elements_per_batch = total_cls_elements // batch_size
        box_elements_per_batch = total_box_elements // batch_size
        
        # Determine number of classes and anchors
        num_positions = height * width
        
        # For box predictions, should be divisible by 4 (x, y, w, h)
        num_boxes = box_elements_per_batch // (4 * num_positions)
        
        print(f"Calculated num_boxes: {num_boxes}")
        
        # Reshape with calculated dimensions
        try:
            # First reshape box predictions
            box_preds_reshaped = box_preds.view(batch_size, num_boxes * 4, height, width)
            box_preds_reshaped = box_preds_reshaped.permute(0, 2, 3, 1)
            box_preds_reshaped = box_preds_reshaped.reshape(batch_size, num_positions * num_boxes, 4)
            
            # Use num_boxes as number of anchors per position
            num_anchors = num_boxes
            
            # Calculate number of classes
            num_classes = cls_elements_per_batch // (num_positions * num_anchors)
            
            print(f"Using num_anchors: {num_anchors}, num_classes: {num_classes}")
            
            # Then reshape class scores
            cls_scores_reshaped = cls_scores.view(batch_size, num_anchors * num_classes, height, width)
            cls_scores_reshaped = cls_scores_reshaped.permute(0, 2, 3, 1)
            cls_scores_reshaped = cls_scores_reshaped.reshape(batch_size, num_positions * num_anchors, num_classes)
        except RuntimeError as e:
            print(f"Error reshaping tensors: {e}")
            # Simpler fallback: flatten everything
            print("Using fallback reshaping method")
            
            # Try to infer dimensions another way
            if box_preds.shape[1] % 4 == 0:
                num_anchors = box_preds.shape[1] // 4
                num_classes = cls_scores.shape[1] // num_anchors
                
                print(f"Fallback: num_anchors={num_anchors}, num_classes={num_classes}")
                
                # Flatten and reshape
                cls_scores_reshaped = cls_scores.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes)
                box_preds_reshaped = box_preds.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            else:
                print("Cannot determine proper reshaping, returning empty list")
                return []
        
        # Apply softmax or sigmoid to cls_scores
        if num_classes > 1:
            scores = torch.softmax(cls_scores_reshaped, dim=2)
        else:
            scores = torch.sigmoid(cls_scores_reshaped)
        
        # Make sure predictions match anchors
        if box_preds_reshaped.shape[1] != anchors.shape[0]:
            print(f"Warning: box_preds shape {box_preds_reshaped.shape} doesn't match anchors shape {anchors.shape}")
            # Use the number that exists in both
            num_elements = min(box_preds_reshaped.shape[1], anchors.shape[0])
            box_preds_reshaped = box_preds_reshaped[:, :num_elements, :]
            anchors = anchors[:num_elements]
        
        # Only process first batch for now
        batch_scores = scores[0]
        batch_boxes = box_preds_reshaped[0]
        
        # Process each class
        results = []
        
        # Determine valid class range
        max_cls = min(num_classes, len(class_indices) + 1)
        
        # Process each class (start from 1 to skip background if using softmax)
        start_cls = 1 if num_classes > 1 else 0
        for cls_idx in range(start_cls, max_cls):
            # Map to original class index
            if cls_idx > 0 and cls_idx - 1 < len(class_indices):
                orig_cls_idx = class_indices[cls_idx - 1]
            else:
                orig_cls_idx = cls_idx
            
            # Get class-specific scores
            cls_scores = batch_scores[:, cls_idx]
            
            # Filter by score threshold
            keep = cls_scores > score_threshold
            if not keep.any():
                continue
                
            cls_scores = cls_scores[keep]
            cls_boxes = batch_boxes[keep]
            kept_anchors = anchors[keep]
            
            # Convert deltas to boxes
            pred_boxes = self._delta2box(cls_boxes, kept_anchors)
            
            # Apply NMS
            keep_indices = self._nms(pred_boxes, cls_scores, nms_threshold)
            if keep_indices.numel() == 0:
                continue
                
            cls_scores = cls_scores[keep_indices]
            pred_boxes = pred_boxes[keep_indices]
            
            # Create output dictionaries
            for score, box in zip(cls_scores.cpu().numpy(), pred_boxes.cpu().numpy()):
                # Convert box from (x1, y1, x2, y2) to (x, y, w, h)
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                results.append({
                    'bbox': [x, y, w, h],
                    'class_id': int(orig_cls_idx),
                    'score': float(score),
                    'feat_level': feat_level
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
            anchors: Anchors [M, 4] in (x1, y1, x2, y2) format
            
        Returns:
            boxes: Decoded boxes [N, 4] in (x1, y1, x2, y2) format
        """
        # Check if deltas and anchors have the same number of boxes
        if deltas.size(0) != anchors.size(0):
            print(f"Warning: deltas shape {deltas.shape} doesn't match anchors shape {anchors.shape}")
            
            # Adjust the number of deltas to match anchors
            if deltas.size(0) > anchors.size(0):
                # Sample or repeat anchors to match deltas
                repeat_factor = deltas.size(0) // anchors.size(0)
                remainder = deltas.size(0) % anchors.size(0)
                
                # Repeat anchors
                anchors_repeated = anchors.repeat(repeat_factor, 1)
                
                # Add remaining anchors if needed
                if remainder > 0:
                    anchors_remainder = anchors[:remainder]
                    anchors = torch.cat([anchors_repeated, anchors_remainder], dim=0)
            else:
                # Sample deltas to match anchors
                deltas = deltas[:anchors.size(0)]
        
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
        # Handle empty inputs
        if boxes.shape[0] == 0 or scores.shape[0] == 0:
            return torch.zeros(0, dtype=torch.int64, device=boxes.device)
            
        # Calculate areas
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort boxes by score
        _, order = scores.sort(descending=True)
        
        # Double check if order is empty
        if order.numel() == 0:
            return torch.zeros(0, dtype=torch.int64, device=boxes.device)
        
        keep = []
        
        # Simple NMS implementation
        while order.shape[0] > 0:
            # Check if order is empty again to be extra safe
            if order.shape[0] == 0:
                break
                
            # Get the index of the highest scoring box
            i = order[0].item()
            keep.append(i)
            
            # If only one box left, we're done
            if order.shape[0] == 1:
                break
                
            # Calculate IoU between the highest scoring box and all other boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            # Calculate intersection area
            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)
            intersection = w * h
            
            # Calculate IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Keep boxes with IoU below threshold
            inds = torch.where(iou <= threshold)[0] + 1  # +1 because we skipped the first box
            
            if inds.shape[0] == 0:
                break
                
            order = order[inds]
        
        return torch.tensor(keep, device=boxes.device)
        
    def _compute_losses(
        self,
        outputs: Dict,
        targets: Dict
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute losses for training
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary of individual loss components
        """
        device = outputs['system_cls'].device
        
        # Create a dummy loss for now that's trainable
        dummy_loss = outputs['system_cls'].mean() * 0.0 + outputs['system_reg'].mean() * 0.0
        
        # You'll implement the actual loss functions here using your hierarchy_loss.py
        loss_dict = {
            'system_cls_loss': dummy_loss.clone(),
            'system_reg_loss': dummy_loss.clone(),
            'staff_cls_loss': dummy_loss.clone(),
            'staff_reg_loss': dummy_loss.clone(),
            'staffline_loss': dummy_loss.clone(),
            'element_loss': dummy_loss.clone()
        }
        
        # Total loss is the sum of all components
        total_loss = sum(loss_dict.values())
        
        return total_loss, loss_dict