import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union

class DetectionHead(nn.Module):
    """
    Generic detection head for musical elements
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_anchors: int = 3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Detection head
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification head (num_classes + 1 for background)
        self.cls_head = nn.Conv2d(hidden_channels, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
        
        # Bounding box regression head
        self.reg_head = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=3, padding=1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize classification head with bias
        nn.init.constant_(self.cls_head.bias, -math.log(99))  # Initial p ≈ 0.01
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: Feature maps from backbone [B, C, H, W]
            
        Returns:
            cls_scores: Classification scores [B, num_anchors*(num_classes+1), H, W]
            bbox_preds: Bounding box predictions [B, num_anchors*4, H, W]
        """
        x = self.conv_layers(features)
        
        cls_scores = self.cls_head(x)
        bbox_preds = self.reg_head(x)
        
        return cls_scores, bbox_preds


class MusicElementDetector(nn.Module):
    """
    Multi-scale detector for different musical elements
    """
    def __init__(
        self,
        in_channels_list: List[int],
        hidden_channels: int = 256,
        class_groups: Dict[str, List[int]] = None,
        anchors_config: Dict = None
    ):
        """
        Initialize the multi-scale detector
        
        Args:
            in_channels_list: List of input channels for each scale
            hidden_channels: Number of channels in detection heads
            class_groups: Dictionary mapping scale name to list of class indices
            anchors_config: Dictionary with anchor configurations
        """
        super().__init__()
        
        # Default class groups if not provided
        if class_groups is None:
            class_groups = {
                'macro': list(range(0, 10)),    # Large elements (systems, staves, etc.)
                'mid': list(range(10, 40)),     # Medium elements (noteheads, clefs, etc.)
                'micro': list(range(40, 100))   # Small elements (dots, articulations, etc.)
            }
        
        self.class_groups = class_groups
        
        # Default anchor configs if not provided
        if anchors_config is None:
            anchors_config = {
                'macro': {
                    'scales': [32, 64, 128],
                    'ratios': [0.5, 1.0, 2.0, 4.0],
                    'num_anchors': 4
                },
                'mid': {
                    'scales': [16, 32, 64],
                    'ratios': [0.5, 1.0, 2.0],
                    'num_anchors': 3
                },
                'micro': {
                    'scales': [8, 16, 32],
                    'ratios': [0.75, 1.0, 1.25],
                    'num_anchors': 3
                }
            }
        
        self.anchors_config = anchors_config
        
        # Create detection heads for each scale
        self.macro_head = DetectionHead(
            in_channels_list[0],
            hidden_channels,
            len(class_groups['macro']),
            anchors_config['macro']['num_anchors']
        )
        
        self.mid_head = DetectionHead(
            in_channels_list[1],
            hidden_channels,
            len(class_groups['mid']),
            anchors_config['mid']['num_anchors']
        )
        
        self.micro_head = DetectionHead(
            in_channels_list[2],
            hidden_channels,
            len(class_groups['micro']),
            anchors_config['micro']['num_anchors']
        )
        
        # Track mapping from class indices to heads
        self.class_to_head = {}
        
        for cls_idx in class_groups['macro']:
            self.class_to_head[cls_idx] = ('macro', cls_idx - min(class_groups['macro']))
            
        for cls_idx in class_groups['mid']:
            self.class_to_head[cls_idx] = ('mid', cls_idx - min(class_groups['mid']))
            
        for cls_idx in class_groups['micro']:
            self.class_to_head[cls_idx] = ('micro', cls_idx - min(class_groups['micro']))
    
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            features_dict: Dictionary of feature maps keyed by scale name (e.g., p2, p3, p4, p5)
            
        Returns:
            Dictionary of (cls_scores, bbox_preds) tuples keyed by detector scale
        """
        # Use appropriate feature maps for each scale
        macro_features = features_dict['p5']  # Lowest resolution for large elements
        mid_features = features_dict['p4']    # Medium resolution
        micro_features = features_dict['p3']  # Higher resolution for small elements
        
        # Forward pass through each head
        macro_cls, macro_reg = self.macro_head(macro_features)
        mid_cls, mid_reg = self.mid_head(mid_features)
        micro_cls, micro_reg = self.micro_head(micro_features)
        
        return {
            'macro': (macro_cls, macro_reg),
            'mid': (mid_cls, mid_reg),
            'micro': (micro_cls, micro_reg)
        }
    
    def get_anchors(
        self,
        image_size: Tuple[int, int],
        feature_sizes: Dict[str, Tuple[int, int]],
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Generate anchors for all scales
        
        Args:
            image_size: (height, width) of input image
            feature_sizes: Dictionary mapping scale names to feature map sizes
            device: Device to create tensors on
            
        Returns:
            Dictionary mapping scale names to anchor tensors
        """
        anchors = {}
        
        # Generate anchors for each scale
        for scale, config in self.anchors_config.items():
            feature_size = feature_sizes[scale]
            anchors[scale] = self._generate_anchors(
                feature_size, image_size, config['scales'], config['ratios'], device
            )
        
        return anchors
    
    def _generate_anchors(
        self,
        feature_size: Tuple[int, int],
        image_size: Tuple[int, int],
        scales: List[int],
        ratios: List[float],
        device: str
    ) -> torch.Tensor:
        """
        Generate anchors for a specific scale
        
        Args:
            feature_size: (height, width) of feature map
            image_size: (height, width) of input image
            scales: List of anchor scales
            ratios: List of anchor aspect ratios
            device: Device for tensor creation
            
        Returns:
            Tensor of anchors, shape [N, 4] in (x1, y1, x2, y2) format
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