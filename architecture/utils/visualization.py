import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Union
import random

def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    score_threshold: float = 0.5,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize detection results on an image
    
    Args:
        image: Input image [H, W, 3] (RGB)
        boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
        scores: Detection scores [N]
        classes: Predicted classes [N]
        class_names: List of class names
        color_map: Dictionary mapping class indices to colors
        score_threshold: Threshold for showing detections
        line_thickness: Thickness of bounding box lines
        font_scale: Font scale for text
        save_path: Path to save visualization
        
    Returns:
        vis_img: Visualization image
    """
    # Make a copy of the image
    vis_img = image.copy()
    
    # Convert to BGR for OpenCV
    if vis_img.ndim == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # Default color map
    if color_map is None:
        color_map = {}
    
    # Filter by score threshold
    if scores is not None:
        keep = scores >= score_threshold
        boxes = boxes[keep]
        
        if classes is not None:
            classes = classes[keep]
        
        if scores is not None:
            scores = scores[keep]
    
    # Ensure boxes are integers
    boxes = boxes.astype(np.int32)
    
    # Draw boxes
    for i, box in enumerate(boxes):
        # Get class and color
        if classes is not None:
            cls_id = classes[i]
            if class_names is not None and cls_id < len(class_names):
                cls_name = class_names[cls_id]
            else:
                cls_name = f"Class {cls_id}"
        else:
            cls_id = 0
            cls_name = "Object"
        
        # Get color for this class
        if cls_id in color_map:
            color = color_map[cls_id]
        else:
            # Generate random color
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            color_map[cls_id] = color
        
        # Draw box
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        if scores is not None:
            label = f"{cls_name}: {scores[i]:.2f}"
        else:
            label = cls_name
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Draw label background
        cv2.rectangle(
            vis_img,
            (x1, y1 - label_h - baseline),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            vis_img,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1
        )
    
    # Save if path is provided
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    # Convert back to RGB
    if vis_img.ndim == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return vis_img


def visualize_stafflines(
    image: np.ndarray,
    stafflines: List[Dict],
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize staffline detection results
    
    Args:
        image: Input image [H, W, 3] (RGB)
        stafflines: List of staffline dictionaries with data
        color_map: Dictionary mapping staff indices to colors
        line_thickness: Thickness of staffline
        save_path: Path to save visualization
        
    Returns:
        vis_img: Visualization image
    """
    # Make a copy of the image
    vis_img = image.copy()
    
    # Convert to BGR for OpenCV
    if vis_img.ndim == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # Default color map
    if color_map is None:
        color_map = {}
    
    # Draw staffs
    for staff_idx, staff in enumerate(stafflines['staffs']):
        # Get color for this staff
        if staff_idx in color_map:
            color = color_map[staff_idx]
        else:
            # Generate random color
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            color_map[staff_idx] = color
        
        # Draw stafflines
        for line in staff['stafflines']:
            x1, y1 = int(line['x1']), int(line['y1'])
            x2, y2 = int(line['x2']), int(line['y2'])
            
            # Draw line
            cv2.line(vis_img, (x1, int((y1+y2)/2)), (x2, int((y1+y2)/2)), color, line_thickness)
            
            # Draw staffline index
            line_idx = line['staffline_idx']
            cv2.putText(
                vis_img,
                f"L{line_idx+1}",
                (x1, int((y1+y2)/2) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    # Save if path is provided
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    # Convert back to RGB
    if vis_img.ndim == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return vis_img


def visualize_relationships(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    edge_indices: np.ndarray,
    edge_scores: np.ndarray,
    class_names: Optional[List[str]] = None,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    score_threshold: float = 0.5,
    line_thickness: int = 1,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize relationship graph
    
    Args:
        image: Input image [H, W, 3] (RGB)
        boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
        classes: Node classes [N]
        edge_indices: Edge indices [2, E]
        edge_scores: Edge scores [E]
        class_names: List of class names
        color_map: Dictionary mapping class indices to colors
        score_threshold: Threshold for showing relationships
        line_thickness: Thickness of lines
        save_path: Path to save visualization
        
    Returns:
        vis_img: Visualization image
    """
    # Make a copy of the image
    vis_img = image.copy()
    
    # Convert to BGR for OpenCV
    if vis_img.ndim == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # Default color map
    if color_map is None:
        color_map = {}
    
    # Filter edges by score threshold
    keep = edge_scores >= score_threshold
    edge_indices = edge_indices[:, keep]
    edge_scores = edge_scores[keep]
    
    # Draw boxes
    for i, box in enumerate(boxes):
        # Get class and color
        cls_id = classes[i]
        if class_names is not None and cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = f"Class {cls_id}"
        
        # Get color for this class
        if cls_id in color_map:
            color = color_map[cls_id]
        else:
            # Generate random color
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            color_map[cls_id] = color
        
        # Draw box
        x1, y1, x2, y2 = box.astype(np.int32)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        cv2.putText(
            vis_img,
            cls_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    # Draw edges
    for i in range(edge_indices.shape[1]):
        src_idx, dst_idx = edge_indices[:, i]
        score = edge_scores[i]
        
        # Get box centers
        src_box = boxes[src_idx]
        dst_box = boxes[dst_idx]
        
        src_center = (
            int((src_box[0] + src_box[2]) / 2),
            int((src_box[1] + src_box[3]) / 2)
        )
        
        dst_center = (
            int((dst_box[0] + dst_box[2]) / 2),
            int((dst_box[1] + dst_box[3]) / 2)
        )
        
        # Edge color based on score (green for high, red for low)
        r = int(255 * (1 - score))
        g = int(255 * score)
        b = 0
        
        # Draw line
        cv2.line(vis_img, src_center, dst_center, (b, g, r), line_thickness)
    
    # Save if path is provided
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    # Convert back to RGB
    if vis_img.ndim == 3 and vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    return vis_img


def visualize_hierarchical_detection(
    image: np.ndarray,
    predictions: Dict,
    class_names: List[str],
    score_threshold: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize hierarchical detection results
    
    Args:
        image: Input image [H, W, 3] (RGB)
        predictions: Dictionary of detection results
        class_names: List of class names
        score_threshold: Threshold for showing detections
        save_path: Path to save visualization
        
    Returns:
        vis_img: Visualization image
    """
    # Make a copy of the image
    vis_img = image.copy()
    
    # Colors for different hierarchical levels
    level_colors = {
        'system': (255, 0, 0),    # Red for systems
        'staff': (0, 255, 0),     # Green for staves
        'staffline': (0, 0, 255), # Blue for stafflines
        'element': (255, 255, 0)  # Yellow for elements
    }
    
    # Draw systems
    if 'systems' in predictions:
        for system in predictions['systems']:
            # Get box
            box = system['bbox']
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            
            # Draw system box
            cv2.rectangle(
                vis_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                level_colors['system'],
                2
            )
            
            # Draw system ID
            cv2.putText(
                vis_img,
                f"System {system['system_id']}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                level_colors['system'],
                2
            )
    
    # Draw staves
    if 'staves' in predictions:
        for staff in predictions['staves']:
            # Get box
            box = staff['bbox']
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            
            # Draw staff box
            cv2.rectangle(
                vis_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                level_colors['staff'],
                2
            )
            
            # Draw staff ID
            cv2.putText(
                vis_img,
                f"Staff {staff['staff_id']}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                level_colors['staff'],
                1
            )
    
    # Draw stafflines
    if 'stafflines' in predictions:
        # Use visualize_stafflines for stafflines
        stafflines_vis = visualize_stafflines(
            vis_img,
            predictions['stafflines'],
            line_thickness=1
        )
        vis_img = stafflines_vis
    
    # Draw musical elements
    if 'elements' in predictions:
        # Prepare boxes, scores, and classes
        boxes = []
        scores = []
        classes = []
        
        for element in predictions['elements']:
            if 'score' in element and element['score'] < score_threshold:
                continue
                
            boxes.append(element['bbox'])
            
            if 'score' in element:
                scores.append(element['score'])
            else:
                scores.append(1.0)
                
            classes.append(element['class_id'])
        
        if boxes:
            # Convert to numpy arrays
            boxes = np.array(boxes)
            scores = np.array(scores)
            classes = np.array(classes)
            
            # Draw elements
            elements_vis = visualize_detection(
                vis_img,
                boxes,
                scores,
                classes,
                class_names,
                score_threshold=score_threshold
            )
            vis_img = elements_vis
    
    # Draw relationships
    if 'relationships' in predictions and 'elements' in predictions:
        # Extract edge information
        edge_indices = predictions['relationships']['edge_indices']
        edge_scores = predictions['relationships']['edge_scores']
        
        # Draw relationships
        if len(edge_indices) > 0 and len(edge_scores) > 0:
            # Prepare boxes and classes
            boxes = []
            classes = []
            
            for element in predictions['elements']:
                boxes.append(element['bbox'])
                classes.append(element['class_id'])
            
            boxes = np.array(boxes)
            classes = np.array(classes)
            
            relations_vis = visualize_relationships(
                vis_img,
                boxes,
                classes,
                edge_indices,
                edge_scores,
                class_names,
                score_threshold=score_threshold
            )
            vis_img = relations_vis
    
    # Save if path is provided
    if save_path:
        if vis_img.ndim == 3 and vis_img.shape[2] == 3:
            # Convert to BGR for OpenCV if in RGB
            save_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        else:
            save_img = vis_img
            
        cv2.imwrite(save_path, save_img)
    
    return vis_img