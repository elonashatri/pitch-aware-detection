import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

def generate_anchors(
    feature_size: Tuple[int, int],
    image_size: Tuple[int, int],
    scales: List[int],
    ratios: List[float],
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate anchors for feature map
    
    Args:
        feature_size: (height, width) of feature map
        image_size: (height, width) of input image
        scales: List of anchor scales
        ratios: List of anchor aspect ratios
        device: Device to create tensors on
        
    Returns:
        anchors: Tensor of anchors [N, 4] in (x1, y1, x2, y2) format
    """
    feat_h, feat_w = feature_size
    img_h, img_w = image_size
    
    # Calculate stride
    stride_h = img_h / feat_h
    stride_w = img_w / feat_w
    
    # Create grid centers
    shift_x = torch.arange(0, feat_w, device=device) * stride_w + stride_w / 2
    shift_y = torch.arange(0, feat_h, device=device) * stride_h + stride_h / 2
    
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shifts = torch.stack([
        shift_x.reshape(-1), shift_y.reshape(-1),
        shift_x.reshape(-1), shift_y.reshape(-1)
    ], dim=1)
    
    # Create base anchors
    anchors = []
    for scale in scales:
        for ratio in ratios:
            # Calculate width and height
            w = scale * ratio ** 0.5
            h = scale / ratio ** 0.5
            
            # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
            x1 = -w / 2
            y1 = -h / 2
            x2 = w / 2
            y2 = h / 2
            
            anchors.append(torch.tensor([x1, y1, x2, y2], device=device))
    
    base_anchors = torch.stack(anchors, dim=0)
    
    # Broadcast anchors over positions
    num_anchors = len(scales) * len(ratios)
    num_positions = shifts.shape[0]
    
    # [A, 4] -> [1, A, 4]
    base_anchors = base_anchors.unsqueeze(0)
    
    # [P, 4] -> [P, 1, 4]
    shifts = shifts.unsqueeze(1)
    
    # [P, A, 4]
    all_anchors = base_anchors + shifts
    
    # [P*A, 4]
    all_anchors = all_anchors.reshape(-1, 4)
    
    return all_anchors


def generate_music_anchors(
    feature_sizes: Dict[str, Tuple[int, int]],
    image_size: Tuple[int, int],
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Generate specialized anchors for music notation elements
    
    Args:
        feature_sizes: Dictionary mapping scale names to feature map sizes
        image_size: (height, width) of input image
        device: Device to create tensors on
        
    Returns:
        anchors_dict: Dictionary mapping scale names to anchor tensors
    """
    anchors_dict = {}
    
    # Default configurations for different scales
    configs = {
        'system': {
            'scales': [64, 128, 256],
            'ratios': [8.0, 12.0, 16.0]  # Very wide for systems
        },
        'staff': {
            'scales': [32, 64, 128],
            'ratios': [5.0, 8.0, 10.0]  # Wide for staves
        },
        'macro': {
            'scales': [32, 64, 128],
            'ratios': [0.5, 1.0, 2.0, 4.0]  # For large symbols
        },
        'mid': {
            'scales': [16, 32, 64],
            'ratios': [0.5, 1.0, 2.0]  # For medium symbols
        },
        'micro': {
            'scales': [8, 16, 32],
            'ratios': [0.75, 1.0, 1.25]  # For small symbols
        },
        'staffline': {  # Extremely wide anchors for stafflines
            'scales': [4, 8, 16],
            'ratios': [100.0, 200.0, 300.0]
        }
    }
    
    # Generate anchors for each scale
    for scale, feature_size in feature_sizes.items():
        if scale in configs:
            config = configs[scale]
            anchors_dict[scale] = generate_anchors(
                feature_size, image_size, config['scales'], config['ratios'], device
            )
    
    return anchors_dict


def generate_staffline_anchors(
    feature_size: Tuple[int, int],
    image_size: Tuple[int, int],
    staff_height: int = 80,
    line_thickness: int = 4,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate specialized anchors for stafflines
    
    Args:
        feature_size: (height, width) of feature map
        image_size: (height, width) of input image
        staff_height: Average height of a staff (5 lines)
        line_thickness: Average thickness of stafflines
        device: Device to create tensors on
        
    Returns:
        anchors: Tensor of anchors [N, 4] in (x1, y1, x2, y2) format
    """
    feat_h, feat_w = feature_size
    img_h, img_w = image_size
    
    # Calculate stride
    stride_h = img_h / feat_h
    stride_w = img_w / feat_w
    
    # Create grid centers
    shift_x = torch.arange(0, feat_w, device=device) * stride_w + stride_w / 2
    shift_y = torch.arange(0, feat_h, device=device) * stride_h + stride_h / 2
    
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shifts = torch.stack([
        shift_x.reshape(-1), shift_y.reshape(-1),
        shift_x.reshape(-1), shift_y.reshape(-1)
    ], dim=1)
    
    # Create base anchors for stafflines
    # Use entire width, with appropriate thickness
    w = img_w
    h = line_thickness
    
    # Use multiple anchors for each position, with different heights
    anchors = []
    for thickness in [h * 0.5, h, h * 2]:
        x1 = -w / 2
        y1 = -thickness / 2
        x2 = w / 2
        y2 = thickness / 2
        
        anchors.append(torch.tensor([x1, y1, x2, y2], device=device))
    
    base_anchors = torch.stack(anchors, dim=0)
    
    # Broadcast anchors over positions
    num_anchors = len(anchors)
    num_positions = shifts.shape[0]
    
    # [A, 4] -> [1, A, 4]
    base_anchors = base_anchors.unsqueeze(0)
    
    # [P, 4] -> [P, 1, 4]
    shifts = shifts.unsqueeze(1)
    
    # [P, A, 4]
    all_anchors = base_anchors + shifts
    
    # [P*A, 4]
    all_anchors = all_anchors.reshape(-1, 4)
    
    return all_anchors


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between box pairs
    
    Args:
        boxes1: First set of boxes [N, 4] in (x1, y1, x2, y2) format
        boxes2: Second set of boxes [M, 4] in (x1, y1, x2, y2) format
        
    Returns:
        iou: IoU values [N, M]
    """
    # Area of boxes1
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    
    # Area of boxes2
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calculate union
    union = area1[:, None] + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union.clamp(min=1e-6)
    
    return iou


def assign_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign targets to anchors
    
    Args:
        anchors: Anchor boxes [N, 4] in (x1, y1, x2, y2) format
        gt_boxes: Ground truth boxes [M, 4] in (x1, y1, x2, y2) format
        gt_classes: Ground truth classes [M]
        pos_iou_thresh: IoU threshold for positive samples
        neg_iou_thresh: IoU threshold for negative samples
        
    Returns:
        assigned_boxes: Assigned box targets [N, 4]
        assigned_classes: Assigned class targets [N]
        assigned_weights: Sample weights [N]
    """
    # Calculate IoU between anchors and gt_boxes
    iou = box_iou(anchors, gt_boxes)  # [N, M]
    
    # Get best IoU and corresponding GT index for each anchor
    max_iou, max_idx = iou.max(dim=1)
    
    # Initialize targets
    num_anchors = anchors.shape[0]
    device = anchors.device
    
    assigned_boxes = torch.zeros((num_anchors, 4), device=device)
    assigned_classes = torch.zeros(num_anchors, dtype=torch.long, device=device)
    assigned_weights = torch.zeros(num_anchors, device=device)
    
    # Positive samples: anchors with IoU > pos_iou_thresh
    pos_mask = max_iou >= pos_iou_thresh
    
    # Also consider the anchor with highest IoU for each GT as positive
    for gt_idx in range(gt_boxes.shape[0]):
        anchor_idx = iou[:, gt_idx].argmax()
        pos_mask[anchor_idx] = True
    
    # Negative samples: anchors with IoU < neg_iou_thresh with all GTs
    neg_mask = max_iou < neg_iou_thresh
    
    # Ignore samples: anchors with IoU between neg_iou_thresh and pos_iou_thresh
    ignore_mask = ~(pos_mask | neg_mask)
    
    # Assign targets for positive samples
    if pos_mask.sum() > 0:
        pos_idx = pos_mask.nonzero().squeeze()
        gt_idx = max_idx[pos_mask]
        
        # Assign boxes using transformation from anchor to GT
        pos_anchors = anchors[pos_mask]
        pos_gt_boxes = gt_boxes[gt_idx]
        
        # Convert to center format
        pos_anchors_c = torch.zeros_like(pos_anchors)
        pos_anchors_c[:, 0] = (pos_anchors[:, 0] + pos_anchors[:, 2]) / 2  # cx
        pos_anchors_c[:, 1] = (pos_anchors[:, 1] + pos_anchors[:, 3]) / 2  # cy
        pos_anchors_c[:, 2] = pos_anchors[:, 2] - pos_anchors[:, 0]        # w
        pos_anchors_c[:, 3] = pos_anchors[:, 3] - pos_anchors[:, 1]        # h
        
        pos_gt_c = torch.zeros_like(pos_gt_boxes)
        pos_gt_c[:, 0] = (pos_gt_boxes[:, 0] + pos_gt_boxes[:, 2]) / 2  # cx
        pos_gt_c[:, 1] = (pos_gt_boxes[:, 1] + pos_gt_boxes[:, 3]) / 2  # cy
        pos_gt_c[:, 2] = pos_gt_boxes[:, 2] - pos_gt_boxes[:, 0]        # w
        pos_gt_c[:, 3] = pos_gt_boxes[:, 3] - pos_gt_boxes[:, 1]        # h
        
        # Calculate box deltas
        dx = (pos_gt_c[:, 0] - pos_anchors_c[:, 0]) / pos_anchors_c[:, 2]
        dy = (pos_gt_c[:, 1] - pos_anchors_c[:, 1]) / pos_anchors_c[:, 3]
        dw = torch.log(pos_gt_c[:, 2] / pos_anchors_c[:, 2])
        dh = torch.log(pos_gt_c[:, 3] / pos_anchors_c[:, 3])
        
        # Assign box targets
        assigned_boxes[pos_mask, 0] = dx
        assigned_boxes[pos_mask, 1] = dy
        assigned_boxes[pos_mask, 2] = dw
        assigned_boxes[pos_mask, 3] = dh
        
        # Assign class targets
        assigned_classes[pos_mask] = gt_classes[gt_idx]
        
        # Assign weights
        assigned_weights[pos_mask] = 1.0
    
    # For negative samples, set weight to 1.0 but keep target as 0 (background)
    assigned_weights[neg_mask] = 1.0
    
    return assigned_boxes, assigned_classes, assigned_weights