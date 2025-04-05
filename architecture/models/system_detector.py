import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


class SystemDetector(nn.Module):
    """
    Module for detecting music systems (groups of staves)
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        num_anchors: int = 3,
        prior_aspect_ratio: List[float] = [8.0, 12.0, 16.0]
    ):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.prior_aspect_ratio = prior_aspect_ratio
        
        # System detector head
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Outputs: classification and regression
        self.cls_head = nn.Conv2d(hidden_channels, num_anchors, kernel_size=3, padding=1)
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
        
        # Initialize classification head with bias towards negative class
        nn.init.constant_(self.cls_head.bias, -torch.log(torch.tensor(99.0)))
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the system detector
        
        Args:
            features: Feature maps from the backbone (B, C, H, W)
            
        Returns:
            cls_scores: Classification scores (B, num_anchors, H, W)
            bbox_preds: Bounding box predictions (B, num_anchors*4, H, W)
        """
        x = self.conv_layers(features)
        
        cls_scores = self.cls_head(x)
        bbox_preds = self.reg_head(x)
        
        return cls_scores, bbox_preds
    
    
    def get_anchors(
        self,
        feature_size: Tuple[int, int],
        image_size: Tuple[int, int],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate anchors for feature map of given size
        
        Args:
            feature_size: (height, width) of the feature map
            image_size: (height, width) of the input image
            device: device to create anchors on
            
        Returns:
            anchors: (H*W*num_anchors, 4) tensor of anchors in (x1, y1, x2, y2) format
        """
        feat_h, feat_w = feature_size
        img_h, img_w = image_size
        
        # Calculate stride between anchors
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
        
        # Create anchor sizes - wide anchors for systems
        # Base size is related to stride
        base_size = min(stride_h, stride_w) * 8  # Systems are typically large
        
        # Create anchors: wide rectangles for systems
        anchors = []
        for ratio in self.prior_aspect_ratio:
            anchor_w = base_size * ratio
            anchor_h = base_size
            
            # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
            x1 = -anchor_w / 2
            y1 = -anchor_h / 2
            x2 = anchor_w / 2
            y2 = anchor_h / 2
            
            anchors.append(torch.tensor([x1, y1, x2, y2], device=device))
        
        anchors = torch.stack(anchors, dim=0)
        
        # Broadcast anchors over all positions
        num_anchors = len(self.prior_aspect_ratio)
        num_positions = shifts.shape[0]
        
        # [A, 4] -> [1, A, 4]
        anchors = anchors.unsqueeze(0)
        
        # [P, 1, 4] -> [P, A, 4] -> [P*A, 4]
        shifts = shifts.unsqueeze(1)
        all_anchors = (anchors + shifts).reshape(-1, 4)
        
        # Print debug info
        print(f"Generated {all_anchors.shape[0]} anchors for feature map of size {feature_size}")
        
        return all_anchors

class StaffDetector(nn.Module):
    """
    Module for detecting staves within systems
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        num_anchors: int = 3,
        prior_aspect_ratio: List[float] = [5.0, 8.0, 10.0]
    ):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.prior_aspect_ratio = prior_aspect_ratio
        
        # Staff detector head
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Outputs: classification and regression
        self.cls_head = nn.Conv2d(hidden_channels, num_anchors, kernel_size=3, padding=1)
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
        nn.init.constant_(self.cls_head.bias, -torch.log(torch.tensor(99.0)))
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the staff detector
        
        Args:
            features: Feature maps from the backbone (B, C, H, W)
            
        Returns:
            cls_scores: Classification scores (B, num_anchors, H, W)
            bbox_preds: Bounding box predictions (B, num_anchors*4, H, W)
        """
        x = self.conv_layers(features)
        
        cls_scores = self.cls_head(x)
        bbox_preds = self.reg_head(x)
        
        return cls_scores, bbox_preds
    
    def get_anchors(self, feature_size: Tuple[int, int], image_size: Tuple[int, int], device: str = 'cpu') -> torch.Tensor:
        """
        Generate anchors for feature map of given size
        
        Args:
            feature_size: (height, width) of the feature map
            image_size: (height, width) of the input image
            device: device to create anchors on
            
        Returns:
            anchors: (H*W*num_anchors, 4) tensor of anchors in (x1, y1, x2, y2) format
        """
        feat_h, feat_w = feature_size
        img_h, img_w = image_size
        
        # Calculate stride between anchors
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
        
        # Create anchor sizes - wide anchors for staves, but not as wide as systems
        # Base size is related to stride
        base_size = min(stride_h, stride_w) * 4
        
        # Create anchors: wide rectangles for staves
        anchors = []
        for ratio in self.prior_aspect_ratio:
            anchor_w = base_size * ratio
            anchor_h = base_size
            
            # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
            x1 = -anchor_w / 2
            y1 = -anchor_h / 2
            x2 = anchor_w / 2
            y2 = anchor_h / 2
            
            anchors.append(torch.tensor([x1, y1, x2, y2], device=device))
        
        anchors = torch.stack(anchors, dim=0)
        
        # Broadcast anchors over all positions
        num_anchors = len(self.prior_aspect_ratio)
        num_positions = shifts.shape[0]
        
        all_anchors = anchors.view(1, num_anchors, 4) + shifts.view(num_positions, 1, 4)
        all_anchors = all_anchors.view(-1, 4)
        
        return all_anchors