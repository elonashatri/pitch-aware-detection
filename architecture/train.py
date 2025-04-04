import os
import time
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from data import OMRDataset, get_training_transforms, get_validation_transforms
from models.omr_model import HierarchicalOMRModel
from losses.hierarchy_loss import HierarchicalDetectionLoss
from utils.visualization import visualize_hierarchical_detection
from utils.evaluation import evaluate_hierarchical_detection

def parse_args():
    parser = argparse.ArgumentParser(description='Train OMR model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(config, device):
    model = HierarchicalOMRModel(config['model'])
    model.to(device)
    return model

def get_optimizer(model, config):
    return optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

def get_scheduler(optimizer, config, n_iter_per_epoch):
    if config['training']['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'] * n_iter_per_epoch,
            eta_min=config['training']['learning_rate'] / 100
        )
    elif config['training']['lr_scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['epochs'] // 3,
            gamma=0.1
        )
    elif config['training']['lr_scheduler'] == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
    else:
        return None

def get_loss_function(config, device, class_names):
    return HierarchicalDetectionLoss(
        num_classes=len(class_names),
        class_names=class_names,
        lambda_det=config['training']['loss_weights']['element_cls'],
        lambda_staff=config['training']['loss_weights']['staffline'],
        lambda_rel=config['training']['loss_weights']['relationship']
    ).to(device)

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best
    if 'best_metric' not in metrics or metrics['val/combined_score'] > metrics['best_metric']:
        metrics['best_metric'] = metrics['val/combined_score']
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'], checkpoint.get('metrics', {})

def visualize_batch(images, targets, predictions, class_names, output_dir, epoch, batch_idx, split='train'):
    # Create visualization directory
    vis_dir = Path(output_dir) / 'visualizations' / split / f'epoch_{epoch}'
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to numpy for visualization
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Visualize each image in the batch
    for i, (image, target, pred) in enumerate(zip(images_np, targets, predictions)):
        # Convert to RGB
        image = (image * 255).astype(np.uint8)
        
        # Visualize
        vis = visualize_hierarchical_detection(
            image,
            pred,
            class_names,
            save_path=str(vis_dir / f'{batch_idx}_{i}.jpg')
        )

def train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch, config, output_dir, class_names):
    model.train()
    
    total_loss = 0
    loss_components = {}
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['images'].to(device)
        
        # Forward pass
        outputs = model(images, batch)
        
        # Compute loss
        loss, loss_dict = loss_fn(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Track loss components
        for k, v in loss_dict.items():
            if k in loss_components:
                loss_components[k] += v.item()
            else:
                loss_components[k] = v.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item()
        })
        
        # Visualize some predictions (every 100 batches)
        if batch_idx % 100 == 0:
            with torch.no_grad():
                predictions = model(images)
            
            visualize_batch(
                images,
                batch,
                predictions,
                class_names,
                output_dir,
                epoch,
                batch_idx,
                'train'
            )
    
    # Compute average loss
    avg_loss = total_loss / len(train_loader)
    avg_loss_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    
    metrics = {
        'train/loss': avg_loss,
        **{f'train/{k}': v for k, v in avg_loss_components.items()}
    }
    
    return metrics

def validate(model, val_loader, loss_fn, device, epoch, config, output_dir, class_names):
    model.eval()
    
    total_loss = 0
    loss_components = {}
    
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(device)
            
            # Forward pass
            outputs = model(images, batch)
            
            # Compute loss
            loss, loss_dict = loss_fn(outputs, batch)
            
            # Track loss
            total_loss += loss.item()
            
            # Track loss components
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v.item()
                else:
                    loss_components[k] = v.item()
            
            # Track predictions
            predictions = model(images)
            all_predictions.append(predictions)
            all_targets.append(batch)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item()
            })
            
            # Visualize some predictions (every 20 batches)
            if batch_idx % 20 == 0:
                visualize_batch(
                    images,
                    batch,
                    predictions,
                    class_names,
                    output_dir,
                    epoch,
                    batch_idx,
                    'val'
                )
    
    # Compute average loss
    avg_loss = total_loss / len(val_loader)
    avg_loss_components = {k: v / len(val_loader) for k, v in loss_components.items()}
    
    # Evaluate predictions
    metrics = evaluate_hierarchical_detection(
        all_predictions,
        all_targets,
        class_names,
        config['inference']['element_score_thresh']
    )
    
    # Add loss metrics
    metrics.update({
        'val/loss': avg_loss,
        **{f'val/{k}': v for k, v in avg_loss_components.items()},
        'val/combined_score': metrics['overall']['combined_score']
    })
    
    return metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config to output directory
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Load class map
    class_map_path = Path(config['data']['class_map_file'])
    if class_map_path.exists():
        with open(class_map_path, 'r') as f:
            class_map = json.load(f)
        class_names = [class_map[str(i)] for i in range(len(class_map))]
    else:
        class_names = None
    
    # Create datasets
    train_dataset = OMRDataset(
        annotations_dir=config['data']['annotations_dir'],
        images_dir=config['data']['images_dir'],
        transform=get_training_transforms(
            height=config['data']['resize_height'],
            width=config['data']['resize_width']
        ),
        split='train',
        split_ratio=config['data']['split_ratio'],
        class_map=class_map
    )
    
    val_dataset = OMRDataset(
        annotations_dir=config['data']['annotations_dir'],
        images_dir=config['data']['images_dir'],
        transform=get_validation_transforms(
            height=config['data']['resize_height'],
            width=config['data']['resize_width']
        ),
        split='val',
        split_ratio=config['data']['split_ratio'],
        class_map=train_dataset.class_map
    )
    
    # Save class map
    if not class_map_path.exists():
        with open(class_map_path, 'w') as f:
            json.dump(train_dataset.class_map, f)
        class_names = [train_dataset.class_map[str(i)] for i in range(len(train_dataset.class_map))]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=OMRDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=OMRDataset.collate_fn
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    # Create model
    model = get_model(config, device)
    
    # Create optimizer
    optimizer = get_optimizer(model, config)
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        config,
        len(train_loader)
    )
    
    # Create loss function
    loss_fn = get_loss_function(config, device, class_names)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    metrics = {}
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
    
    # Train loop
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f'Starting epoch {epoch}')
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            epoch,
            config,
            output_dir,
            class_names
        )
        
        # Update metrics
        metrics.update(train_metrics)
        
        # Evaluate every evaluation interval
        if epoch % config['training']['eval_interval'] == 0:
            val_metrics = validate(
                model,
                val_loader,
                loss_fn,
                device,
                epoch,
                config,
                output_dir,
                class_names
            )
            
            # Update metrics
            metrics.update(val_metrics)
        
        # Save checkpoint every save interval
        if epoch % config['training']['save_interval'] == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                output_dir
            )
        
        # Print metrics
        print(f'Epoch {epoch} metrics:')
        for k, v in metrics.items():
            if k.startswith('train/') or k.startswith('val/'):
                print(f'  {k}: {v}')

if __name__ == '__main__':
    main()