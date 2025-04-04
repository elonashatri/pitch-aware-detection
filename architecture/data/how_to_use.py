# Example usage
import torch
from torch.utils.data import DataLoader
from data import (
    OMRDataset, 
    get_training_transforms, 
    get_validation_transforms
)

# Paths
annotations_dir = '/homes/es314/omr-objdet-benchmark/data/annotations'
images_dir = '/homes/es314/omr-objdet-benchmark/data/images'

# Create datasets
train_dataset = OMRDataset(
    annotations_dir=annotations_dir,
    images_dir=images_dir,
    transform=get_training_transforms(height=1024, width=1024),
    split='train'
)

val_dataset = OMRDataset(
    annotations_dir=annotations_dir,
    images_dir=images_dir,
    transform=get_validation_transforms(height=1024, width=1024),
    split='val',
    class_map=train_dataset.class_map  # Use same class map as training
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=OMRDataset.collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=OMRDataset.collate_fn
)

# Check a batch
for batch in train_loader:
    print(f"Batch size: {batch['images'].shape}")
    print(f"Number of systems: {[len(sys) for sys in batch['systems']]}")
    print(f"Total elements: {sum(len(elems) for elems in batch['elements'])}")
    break