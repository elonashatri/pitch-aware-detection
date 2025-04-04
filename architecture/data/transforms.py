import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Union

def get_training_transforms(height: int = 1024, width: int = 1024) -> A.Compose:
    """
    Returns training augmentations suitable for music notation
    
    Args:
        height: Target height for resizing
        width: Target width for resizing
        
    Returns:
        Albumentations transform composition
    """
    return A.Compose(
        [
            # Resize with preserved aspect ratio
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            
            # Light augmentations (preserving readability of music notation)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
                A.CLAHE(clip_limit=2.0, p=0.8),
            ], p=0.5),
            
            # Subtle noise to simulate document imperfections
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5),
            ], p=0.3),
            
            # Paper-like deformations
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.5),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
            ], p=0.2),
            
            # Convert to tensor
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='coco',  # [x_min, y_min, width, height]
            label_fields=['class_labels']
        )
    )

def get_validation_transforms(height: int = 1024, width: int = 1024) -> A.Compose:
    """
    Returns validation transforms (no augmentation)
    
    Args:
        height: Target height for resizing
        width: Target width for resizing
        
    Returns:
        Albumentations transform composition
    """
    return A.Compose(
        [
            # Resize with preserved aspect ratio
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            
            # Convert to tensor
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='coco',  # [x_min, y_min, width, height]
            label_fields=['class_labels']
        )
    )

def get_test_transforms(height: int = 1024, width: int = 1024) -> A.Compose:
    """
    Returns test transforms (no augmentation, no resizing for exact evaluation)
    
    Args:
        height: Target height for padding
        width: Target width for padding
        
    Returns:
        Albumentations transform composition
    """
    return A.Compose(
        [
            # Only pad if needed, no resize to preserve exact proportions
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            
            # Convert to tensor
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='coco',  # [x_min, y_min, width, height]
            label_fields=['class_labels']
        )
    )