import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class StafflineDetector(nn.Module):
    """
    Specialized detector for staff lines that accounts for their extreme aspect ratio
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        groups: int = 5  # 5 lines per staff
    ):
        super().__init__()
        
        self.groups = groups
        
        # Horizontal-focused convolutions for stafflines
        self.horizontal_conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=(1, 9), padding=(0, 4)
        )
        self.horizontal_conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=(1, 9), padding=(0, 4)
        )
        
        # Standard convolutions
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output layers - one per staffline in a standard staff (5 lines)
        self.staffline_predictor = nn.Conv2d(hidden_channels, groups, kernel_size=3, padding=1)
        
        # Regression head for fine adjustment of staffline height and position
        self.offset_predictor = nn.Conv2d(hidden_channels, groups * 2, kernel_size=3, padding=1)  # y-offset and height
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the staffline detector
        
        Args:
            features: Feature maps from the backbone [B, C, H, W]
            
        Returns:
            staffline_heatmaps: Heatmaps for each staffline [B, groups, H, W]
            staffline_offsets: Offset predictions for fine adjustment [B, groups*2, H, W]
        """
        # Extract horizontal features
        h_features = F.relu(self.horizontal_conv1(features))
        h_features = F.relu(self.horizontal_conv2(h_features))
        
        # Concatenate with original features
        combined_features = torch.cat([features, h_features], dim=1)
        
        # Process through standard convs
        x = self.conv_layers(combined_features)
        
        # Predict staffline heatmaps and offsets
        staffline_heatmaps = torch.sigmoid(self.staffline_predictor(x))
        staffline_offsets = self.offset_predictor(x)
        
        return staffline_heatmaps, staffline_offsets
    
    def decode_stafflines(
        self,
        heatmaps: torch.Tensor,
        offsets: torch.Tensor,
        threshold: float = 0.5,
        max_width_ratio: float = 0.25  # Max width as ratio of image width
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode staffline predictions from heatmaps and offsets
        
        Args:
            heatmaps: Staffline heatmaps [B, groups, H, W]
            offsets: Staffline offsets [B, groups*2, H, W]
            threshold: Detection threshold
            max_width_ratio: Maximum staffline width as ratio of image width
            
        Returns:
            stafflines: List of dicts containing staffline parameters
        """
        batch_size, _, height, width = heatmaps.shape
        device = heatmaps.device
        
        stafflines_batch = []
        
        for b in range(batch_size):
            batch_stafflines = []
            
            # Process each staffline (5 lines per staff)
            for g in range(self.groups):
                heatmap = heatmaps[b, g]
                y_offset = offsets[b, g*2]
                height_pred = offsets[b, g*2 + 1]
                
                # Find positions above threshold
                above_threshold = heatmap > threshold
                
                # If nothing detected, continue
                if not above_threshold.any():
                    continue
                
                # Group consecutive horizontal pixels
                staffline_segments = []
                for h in range(height):
                    row = above_threshold[h]
                    if not row.any():
                        continue
                    
                    # Find connected regions
                    changes = torch.cat([
                        torch.tensor([False], device=device),
                        row[1:] != row[:-1],
                        torch.tensor([False], device=device)
                    ])
                    change_indices = changes.nonzero().view(-1)
                    
                    # Process segments
                    for i in range(0, len(change_indices), 2):
                        if i + 1 >= len(change_indices):
                            break
                            
                        start, end = change_indices[i], change_indices[i + 1]
                        
                        # Skip if too short (noise)
                        segment_width = end - start
                        if segment_width < width * 0.05:  # Skip very short segments
                            continue
                            
                        # Skip if too wide (probably false positive)
                        if segment_width > width * max_width_ratio:
                            continue
                        
                        # Get average offset and height for this segment
                        seg_y_offset = y_offset[h, start:end].mean().item()
                        seg_height = torch.abs(height_pred[h, start:end].mean()).item()
                        
                        # Ensure minimum height
                        seg_height = max(seg_height, 1.0)
                        
                        # Add to segments
                        staffline_segments.append({
                            'x1': start.item(),
                            'y1': h + seg_y_offset - seg_height / 2,
                            'x2': end.item(),
                            'y2': h + seg_y_offset + seg_height / 2,
                            'confidence': heatmap[h, start:end].mean().item(),
                            'staffline_idx': g
                        })
                
                # Merge horizontally close segments (handle gaps)
                if staffline_segments:
                    merged_segments = []
                    staffline_segments.sort(key=lambda x: (x['y1'], x['x1']))
                    
                    current = staffline_segments[0]
                    for next_seg in staffline_segments[1:]:
                        # If same height (approximately) and close horizontally
                        if (abs(next_seg['y1'] - current['y1']) < 3 and 
                            next_seg['x1'] - current['x2'] < width * 0.05):
                            # Merge
                            current['x2'] = next_seg['x2']
                            current['confidence'] = (current['confidence'] + next_seg['confidence']) / 2
                        else:
                            merged_segments.append(current)
                            current = next_seg
                    
                    merged_segments.append(current)
                    batch_stafflines.extend(merged_segments)
            
            # Group stafflines into staves (5 lines per staff)
            staffs = []
            if len(batch_stafflines) >= 5:
                # Sort by y position
                batch_stafflines.sort(key=lambda x: x['y1'])
                
                # Group every 5 consecutive lines
                for i in range(0, len(batch_stafflines) - 4, 5):
                    staff_lines = batch_stafflines[i:i+5]
                    
                    # Check if these are likely to be from the same staff
                    y_positions = [line['y1'] for line in staff_lines]
                    y_diffs = [y_positions[j+1] - y_positions[j] for j in range(4)]
                    
                    # Calculate mean and std of spacings
                    mean_spacing = sum(y_diffs) / 4
                    std_spacing = (sum((d - mean_spacing) ** 2 for d in y_diffs) / 4) ** 0.5
                    
                    # If spacings are consistent (low std), consider as one staff
                    if std_spacing < mean_spacing * 0.2:  # 20% tolerance
                        staffs.append({
                            'stafflines': staff_lines,
                            'mean_spacing': mean_spacing,
                            'staff_idx': len(staffs)
                        })
            
            stafflines_batch.append({
                'stafflines': batch_stafflines,
                'staffs': staffs
            })
        
        return stafflines_batch