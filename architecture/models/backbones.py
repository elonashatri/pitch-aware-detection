import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union

class ResNetBackbone(nn.Module):
    """
    ResNet backbone with FPN for multi-scale feature extraction
    """
    def __init__(
        self,
        name: str = 'resnet50',
        pretrained: bool = True,
        out_channels: int = 256
    ):
        super().__init__()
        
        # Select backbone
        if name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            backbone_channels = [64, 128, 256, 512]
        elif name == 'resnet34':
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            backbone_channels = [64, 128, 256, 512]
        elif name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            backbone_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        
        # Remove the classification head
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32
        
        # Feature Pyramid Network (FPN)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(backbone_channels[3], out_channels, kernel_size=1),
            nn.Conv2d(backbone_channels[2], out_channels, kernel_size=1),
            nn.Conv2d(backbone_channels[1], out_channels, kernel_size=1),
            nn.Conv2d(backbone_channels[0], out_channels, kernel_size=1),
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ])
        
        # Initialize weights for FPN
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        for m in self.fpn_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Bottom-up pathway
        c1 = self.layer0(x)       # 1/4
        c2 = self.layer1(c1)      # 1/4
        c3 = self.layer2(c2)      # 1/8
        c4 = self.layer3(c3)      # 1/16
        c5 = self.layer4(c4)      # 1/32
        
        # Top-down pathway and lateral connections
        p5 = self.lateral_convs[0](c5)
        p4 = self.lateral_convs[1](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[2](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[3](c2) + nn.functional.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        
        # Apply 3x3 convs to each feature map
        p5 = self.fpn_convs[0](p5)
        p4 = self.fpn_convs[1](p4)
        p3 = self.fpn_convs[2](p3)
        p2 = self.fpn_convs[3](p2)
        
        return {
            'p2': p2,  # 1/4 resolution
            'p3': p3,  # 1/8 resolution
            'p4': p4,  # 1/16 resolution
            'p5': p5   # 1/32 resolution
        }


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone with FPN for multi-scale feature extraction
    """
    def __init__(
        self,
        name: str = 'efficientnet_b0',
        pretrained: bool = True,
        out_channels: int = 256
    ):
        super().__init__()
        
        # Select backbone
        if name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            backbone_channels = [16, 24, 40, 112, 320]
        elif name == 'efficientnet_b2':
            backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            backbone_channels = [16, 24, 48, 120, 352]
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        
        # Extract feature layers
        self.features = backbone.features
        
        # Feature Pyramid Network
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(backbone_channels[4], out_channels, kernel_size=1),
            nn.Conv2d(backbone_channels[3], out_channels, kernel_size=1),
            nn.Conv2d(backbone_channels[2], out_channels, kernel_size=1),
            nn.Conv2d(backbone_channels[1], out_channels, kernel_size=1),
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ])
        
        # Initialize weights for FPN
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        for m in self.fpn_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features at different scales
        features = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in [1, 2, 3, 5, 8]:  # Indices of blocks with appropriate scales
                features.append(x)
        
        c1, c2, c3, c4, c5 = features
        
        # Top-down pathway and lateral connections
        p5 = self.lateral_convs[0](c5)
        p4 = self.lateral_convs[1](c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[2](c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[3](c2) + nn.functional.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        
        # Apply 3x3 convs to each feature map
        p5 = self.fpn_convs[0](p5)
        p4 = self.fpn_convs[1](p4)
        p3 = self.fpn_convs[2](p3)
        p2 = self.fpn_convs[3](p2)
        
        return {
            'p2': p2,  # 1/4 resolution
            'p3': p3,  # 1/8 resolution
            'p4': p4,  # 1/16 resolution
            'p5': p5   # 1/32 resolution
        }


def get_backbone(name: str, pretrained: bool = True, out_channels: int = 256) -> nn.Module:
    """
    Factory function to get a backbone network
    """
    if name.startswith('resnet'):
        return ResNetBackbone(name, pretrained, out_channels)
    elif name.startswith('efficientnet'):
        return EfficientNetBackbone(name, pretrained, out_channels)
    else:
        raise ValueError(f"Unsupported backbone: {name}")