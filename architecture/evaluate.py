import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.omr_model import HierarchicalOMRModel
from utils.visualization import visualize_hierarchical_detection

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with OMR model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output-dir', type=str, default='inference_output', help='Output directory')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(config, device):
    model = HierarchicalOMRModel(config['model'])
    model.to(device)
    return model

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return checkpoint['epoch'], checkpoint.get('metrics', {})

def preprocess_image(image_path, config):
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize while preserving aspect ratio
    h, w = image.shape[:2]
    scale = min(
        config['data']['resize_height'] / h,
        config['data']['resize_width'] / w
    )
    
    new_h, new_w = int(h * scale), int(w * scale)
    
    image = cv2.resize(image, (new_w, new_h))
    
    # Pad to target size
    pad_h = config['data']['resize_height'] - new_h
    pad_w = config['data']['resize_width'] - new_w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    image = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    
    # Convert to tensor
    image = torch.tensor(image).float() / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)
    
    return image, (h, w), (top, left, bottom, right)

def process_image(image_path, model, config, device, output_dir, class_names, visualize=False):
    # Preprocess image
    image, orig_size, padding = preprocess_image(image_path, config)
    image = image.to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image)
    
    # Post-process predictions (convert to original image coordinates)
    processed_predictions = postprocess_predictions(
        predictions,
        orig_size,
        padding,
        config['data']['resize_height'],
        config['data']['resize_width']
    )
    
    # Save predictions
    output_path = Path(output_dir) / f"{Path(image_path).stem}_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(processed_predictions, f, indent=2, default=str)
    
    # Visualize predictions
    if visualize:
        # Load original image
        orig_image = cv2.imread(str(image_path))
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        # Visualize
        vis_path = Path(output_dir) / f"{Path(image_path).stem}_visualization.jpg"
        vis = visualize_hierarchical_detection(
            orig_image,
            processed_predictions,
            class_names,
            save_path=str(vis_path)
        )
    
    return processed_predictions

def postprocess_predictions(predictions, orig_size, padding, resize_h, resize_w):
    """
    Convert predictions back to original image coordinates
    """
    # Get padding and scale
    top, left, bottom, right = padding
    orig_h, orig_w = orig_size
    
    # Calculate scale
    scale_h = orig_h / (resize_h - top - bottom)
    scale_w = orig_w / (resize_w - left - right)
    
    # Process systems
    if 'systems' in predictions:
        for system in predictions['systems']:
            x, y, w, h = system['bbox']
            
            # Remove padding
            x -= left
            y -= top
            
            # Apply scale
            x *= scale_w
            y *= scale_h
            w *= scale_w
            h *= scale_h
            
            # Update bbox
            system['bbox'] = [x, y, w, h]
    
    # Process staves
    if 'staves' in predictions:
        for staff in predictions['staves']:
            x, y, w, h = staff['bbox']
            
            # Remove padding
            x -= left
            y -= top
            
            # Apply scale
            x *= scale_w
            y *= scale_h
            w *= scale_w
            h *= scale_h
            
            # Update bbox
            staff['bbox'] = [x, y, w, h]
    
    # Process stafflines
    if 'stafflines' in predictions and 'staffs' in predictions['stafflines']:
        for staff in predictions['stafflines']['staffs']:
            for line in staff['stafflines']:
                x1, y1 = line['x1'], line['y1']
                x2, y2 = line['x2'], line['y2']
                
                # Remove padding
                x1 -= left
                y1 -= top
                x2 -= left
                y2 -= top
                
                # Apply scale
                x1 *= scale_w
                y1 *= scale_h
                x2 *= scale_w
                y2 *= scale_h
                
                # Update coordinates
                line['x1'] = x1
                line['y1'] = y1
                line['x2'] = x2
                line['y2'] = y2
    
    # Process elements
    if 'elements' in predictions:
        for element in predictions['elements']:
            x, y, w, h = element['bbox']
            
            # Remove padding
            x -= left
            y -= top
            
            # Apply scale
            x *= scale_w
            y *= scale_h
            w *= scale_w
            h *= scale_h
            
            # Update bbox
            element['bbox'] = [x, y, w, h]
    
    return predictions

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
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
        print(f'Warning: Class map file {class_map_path} not found')
    
    # Create model
    model = get_model(config, device)
    
    # Load checkpoint
    epoch, checkpoint_metrics = load_checkpoint(args.checkpoint, model)
    print(f'Loaded checkpoint from epoch {epoch}')
    
    # Set model to evaluation mode
    model.eval()
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single image
        print(f'Processing image: {input_path}')
        process_image(
            input_path,
            model,
            config,
            device,
            output_dir,
            class_names,
            args.visualize
        )
    elif input_path.is_dir():
        # Process directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f'Processing {len(image_paths)} images in directory: {input_path}')
        
        for image_path in tqdm(image_paths):
            process_image(
                image_path,
                model,
                config,
                device,
                output_dir,
                class_names,
                args.visualize
            )
    else:
        print(f'Error: Input path {input_path} not found or not valid')

if __name__ == '__main__':
    main()