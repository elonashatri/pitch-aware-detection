import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OMRDataset(Dataset):
    """
    Dataset for Optical Music Recognition with hierarchical structure:
    System -> Staff -> Stafflines -> Musical Elements
    """
    def __init__(
        self,
        annotations_dir: str,
        images_dir: str,
        class_map: Dict[str, int] = None,
        transform: Optional[A.Compose] = None,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42
    ):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.transform = transform
        self.split = split
        
        # Get all XML files
        self.xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        
        # Create train/val/test splits if not already done
        if split:
            np.random.seed(seed)
            indices = np.random.permutation(len(self.xml_files))
            
            train_end = int(split_ratio[0] * len(indices))
            val_end = int((split_ratio[0] + split_ratio[1]) * len(indices))
            
            if split == 'train':
                self.xml_files = [self.xml_files[i] for i in indices[:train_end]]
            elif split == 'val':
                self.xml_files = [self.xml_files[i] for i in indices[train_end:val_end]]
            elif split == 'test':
                self.xml_files = [self.xml_files[i] for i in indices[val_end:]]
                
        # Class mapping
        self.class_map = class_map
        if self.class_map is None:
            # Create a default class map based on the dataset
            self.class_map = self._build_class_map()
        else:
            # If we're given a class map, ensure it's in the format {class_name: class_id}
            # If it's in the format {class_id: class_name}, invert it
            if all(isinstance(k, str) and isinstance(v, int) for k, v in class_map.items()):
                # It's already in the right format
                pass
            elif all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in class_map.keys()):
                # It's in the wrong format, invert it
                self.class_map = {v: int(k) if isinstance(k, str) else k for k, v in class_map.items()}
        
        # Initialize class counts dictionary
        self.class_counts = {cls_id: 0 for cls_id in self.class_map.values()}
                
    def _build_class_map(self) -> Dict[str, int]:
        """
        Build a class map from the dataset if not provided
        """
        classes = set()
        # Sample the first 100 files to find classes
        for xml_file in self.xml_files[:min(100, len(self.xml_files))]:
            tree = ET.parse(os.path.join(self.annotations_dir, xml_file))
            root = tree.getroot()
            
            for node in root.findall('.//Node'):
                class_name_elem = node.find('ClassName')
                if class_name_elem is not None:
                    classes.add(class_name_elem.text)
        
        return {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}
    
    def __len__(self) -> int:
        return len(self.xml_files)
    
    def __getitem__(self, idx: int) -> Dict:
        xml_file = self.xml_files[idx]
        img_file = xml_file.replace('.xml', '.png')  # Adjust extension if needed
        
        # Parse XML
        tree = ET.parse(os.path.join(self.annotations_dir, xml_file))
        root = tree.getroot()
        
        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Extract annotations by type
        systems = []
        staves = []
        stafflines = []
        elements = []
        
        # First, identify all stafflines to group them
        staffline_by_system = {}
        staffline_by_staff = {}
        
        for node in root.findall('.//Node'):
            class_name_elem = node.find('ClassName')
            if class_name_elem is None:
                continue
                
            class_name = class_name_elem.text
            class_id = self.class_map.get(class_name, -1)
            
            # Skip if not in our class map
            if class_id == -1:
                continue
                
            # Get bounding box
            top = float(node.find('Top').text)
            left = float(node.find('Left').text)
            width = float(node.find('Width').text)
            height = float(node.find('Height').text)
            
            # Convert to normalized coordinates (for YOLO format)
            x_center = (left + width / 2) / img_width
            y_center = (top + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            # Get element relationships and attributes
            system_id = None
            staff_id = None
            
            data_items = node.findall('.//DataItem')
            for item in data_items:
                key = item.get('key')
                if key == 'ordered_staff_id':
                    staff_id = int(item.text)
                elif key == 'spacing_run_id':
                    system_id = int(item.text)
            
            # Store absolute coordinates for processing
            bbox = [left, top, width, height]
            norm_bbox = [x_center, y_center, norm_width, norm_height]
            
            # Organize by element type
            if class_name == 'kStaffLine':
                stafflines.append({
                    'bbox': bbox,
                    'norm_bbox': norm_bbox,
                    'class_id': class_id,
                    'class_name': class_name,
                    'system_id': system_id,
                    'staff_id': staff_id
                })
                
                # Group stafflines by system and staff
                if system_id not in staffline_by_system:
                    staffline_by_system[system_id] = []
                staffline_by_system[system_id].append(len(stafflines) - 1)
                
                if staff_id not in staffline_by_staff:
                    staffline_by_staff[staff_id] = []
                staffline_by_staff[staff_id].append(len(stafflines) - 1)
            else:
                # Get relationship info from inlinks/outlinks
                inlinks = node.find('Inlinks')
                outlinks = node.find('Outlinks')
                
                inlink_ids = []
                outlink_ids = []
                
                if inlinks is not None and inlinks.text:
                    inlink_ids = [int(id_) for id_ in inlinks.text.split()]
                
                if outlinks is not None and outlinks.text:
                    outlink_ids = [int(id_) for id_ in outlinks.text.split()]
                
                elements.append({
                    'bbox': bbox,
                    'norm_bbox': norm_bbox,
                    'class_id': class_id,
                    'class_name': class_name,
                    'system_id': system_id,
                    'staff_id': staff_id,
                    'inlinks': inlink_ids,
                    'outlinks': outlink_ids
                })
                
                # Update class count
                self.class_counts[class_id] += 1
        
        # Group stafflines into staves (5 lines per staff)
        for staff_id, line_indices in staffline_by_staff.items():
            if len(line_indices) == 5:  # Complete staff
                staff_lines = [stafflines[i] for i in line_indices]
                
                # Sort stafflines by vertical position
                staff_lines.sort(key=lambda x: x['bbox'][1])  # Sort by top coordinate
                
                # Calculate staff bounding box
                staff_left = min(line['bbox'][0] for line in staff_lines)
                staff_top = min(line['bbox'][1] for line in staff_lines)
                staff_right = max(line['bbox'][0] + line['bbox'][2] for line in staff_lines)
                staff_bottom = max(line['bbox'][1] + line['bbox'][3] for line in staff_lines)
                
                staff_width = staff_right - staff_left
                staff_height = staff_bottom - staff_top
                
                staves.append({
                    'bbox': [staff_left, staff_top, staff_width, staff_height],
                    'norm_bbox': [
                        (staff_left + staff_width / 2) / img_width,
                        (staff_top + staff_height / 2) / img_height,
                        staff_width / img_width,
                        staff_height / img_height
                    ],
                    'staff_id': staff_id,
                    'system_id': staff_lines[0]['system_id'],
                    'staffline_indices': line_indices
                })
        
        # Group staves into systems
        system_staves = {}
        for staff in staves:
            if staff['system_id'] not in system_staves:
                system_staves[staff['system_id']] = []
            system_staves[staff['system_id']].append(staff)
        
        # Create system bounding boxes
        for system_id, sys_staves in system_staves.items():
            system_left = min(staff['bbox'][0] for staff in sys_staves)
            system_top = min(staff['bbox'][1] for staff in sys_staves)
            system_right = max(staff['bbox'][0] + staff['bbox'][2] for staff in sys_staves)
            system_bottom = max(staff['bbox'][1] + staff['bbox'][3] for staff in sys_staves)
            
            system_width = system_right - system_left
            system_height = system_bottom - system_top
            
            systems.append({
                'bbox': [system_left, system_top, system_width, system_height],
                'norm_bbox': [
                    (system_left + system_width / 2) / img_width,
                    (system_top + system_height / 2) / img_height,
                    system_width / img_width,
                    system_height / img_height
                ],
                'system_id': system_id,
                'staff_indices': [i for i, staff in enumerate(staves) if staff['system_id'] == system_id]
            })
        
        # Apply transformations
        if self.transform:
            # We need to transform bounding boxes along with the image
            bboxes = [elem['bbox'] for elem in elements]
            bbox_classes = [elem['class_id'] for elem in elements]
            
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=bbox_classes
            )
            
            image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            
            # Update element bboxes
            for i, (box, class_id) in enumerate(zip(transformed_bboxes, transformed['class_labels'])):
                left, top, width, height = box
                elements[i]['bbox'] = [left, top, width, height]
                elements[i]['norm_bbox'] = [
                    (left + width / 2) / img_width,
                    (top + height / 2) / img_height,
                    width / img_width,
                    height / img_height
                ]
        
        # Convert to tensor if not done in transform
        # if self.transform is None or not any(isinstance(t, ToTensorV2) for t in self.transform.transforms):
        #     image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        # Convert to tensor if not done in transform
        if self.transform is None or not any(isinstance(t, ToTensorV2) for t in self.transform.transforms):
            # Explicitly normalize and convert to float
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return {
            'image': image,
            'systems': systems,
            'staves': staves,
            'stafflines': stafflines,
            'elements': elements,
            'file_name': xml_file,
            'image_id': idx,
            'image_size': (img_height, img_width)
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights inversely proportional to frequency
        """
        counts = np.array([self.class_counts[i] for i in range(len(self.class_map))])
        counts = np.maximum(counts, 1)  # Avoid division by zero
        weights = 1.0 / counts
        weights = weights / weights.sum()  # Normalize
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for variable sized data
        """
        images = torch.stack([item['image'] for item in batch])
        
        return {
            'images': images,
            'systems': [item['systems'] for item in batch],
            'staves': [item['staves'] for item in batch],
            'stafflines': [item['stafflines'] for item in batch],
            'elements': [item['elements'] for item in batch],
            'file_names': [item['file_name'] for item in batch],
            'image_ids': [item['image_id'] for item in batch],
            'image_sizes': [item['image_size'] for item in batch]
        }