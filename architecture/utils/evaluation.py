import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def compute_iou(
    boxes1: np.ndarray,
    boxes2: np.ndarray
) -> np.ndarray:
    """
    Compute IoU between box pairs
    
    Args:
        boxes1: First set of boxes [N, 4] in (x1, y1, x2, y2) or (x, y, w, h) format
        boxes2: Second set of boxes [M, 4] in same format as boxes1
        
    Returns:
        iou: IoU values [N, M]
    """
    # Convert from (x, y, w, h) to (x1, y1, x2, y2) if needed
    if boxes1.shape[1] == 4 and (boxes1[:, 2:] > 0).all() and (boxes2[:, 2:] > 0).all():
        if (boxes1[:, 2] < boxes1[:, 0]).any() or (boxes1[:, 3] < boxes1[:, 1]).any():
            # Already in (x1, y1, x2, y2) format
            pass
        else:
            # Convert from (x, y, w, h) to (x1, y1, x2, y2)
            boxes1_new = boxes1.copy()
            boxes1_new[:, 2] = boxes1[:, 0] + boxes1[:, 2]
            boxes1_new[:, 3] = boxes1[:, 1] + boxes1[:, 3]
            boxes1 = boxes1_new
            
            boxes2_new = boxes2.copy()
            boxes2_new[:, 2] = boxes2[:, 0] + boxes2[:, 2]
            boxes2_new[:, 3] = boxes2[:, 1] + boxes2[:, 3]
            boxes2 = boxes2_new
    
    # Area of boxes1
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    
    # Area of boxes2
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calculate intersection
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    
    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calculate union
    union = area1[:, None] + area2[None, :] - intersection
    
    # Calculate IoU
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou


def evaluate_detection(
    pred_boxes: List[np.ndarray],
    pred_classes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_classes: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Evaluate detection performance
    
    Args:
        pred_boxes: List of predicted boxes arrays, one per image
        pred_classes: List of predicted classes arrays, one per image
        pred_scores: List of predicted scores arrays, one per image
        gt_boxes: List of ground truth boxes arrays, one per image
        gt_classes: List of ground truth classes arrays, one per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Initialize metrics
    class_metrics = {
        'precision': np.zeros(num_classes),
        'recall': np.zeros(num_classes),
        'ap': np.zeros(num_classes),
        'f1': np.zeros(num_classes),
        'count': np.zeros(num_classes, dtype=np.int32)
    }
    
    # Process each image
    for pred_box, pred_class, pred_score, gt_box, gt_class in zip(
        pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes
    ):
        # Skip if no predictions or ground truths
        if len(pred_box) == 0 or len(gt_box) == 0:
            continue
        
        # Compute IoU between predictions and ground truths
        iou = compute_iou(pred_box, gt_box)
        
        # For each prediction, find the best matching ground truth
        for i, (box, cls, score) in enumerate(zip(pred_box, pred_class, pred_score)):
            # Consider only ground truths of the same class
            same_class_mask = gt_class == cls
            if not same_class_mask.any():
                continue
                
            # Get IoU with ground truths of the same class
            iou_cls = iou[i, same_class_mask]
            if len(iou_cls) == 0:
                continue
                
            # Find the best matching ground truth
            best_iou = iou_cls.max()
            if best_iou >= iou_threshold:
                best_gt_idx = same_class_mask.nonzero()[0][iou_cls.argmax()]
                
                # Mark this ground truth as matched
                gt_class[best_gt_idx] = -1
                
                # Update metrics for this class
                class_metrics['count'][cls] += 1
    
    # Compute precision and recall for each class
    for cls in range(num_classes):
        # Count predictions and ground truths for this class
        pred_count = sum(np.sum(pred_cls == cls) for pred_cls in pred_classes)
        gt_count = sum(np.sum(gt_cls == cls) for gt_cls in gt_classes)
        
        # Skip if no predictions or ground truths
        if pred_count == 0 or gt_count == 0:
            continue
            
        # Compute precision and recall
        class_metrics['precision'][cls] = class_metrics['count'][cls] / max(pred_count, 1)
        class_metrics['recall'][cls] = class_metrics['count'][cls] / max(gt_count, 1)
        
        # Compute F1 score
        if class_metrics['precision'][cls] + class_metrics['recall'][cls] > 0:
            class_metrics['f1'][cls] = (
                2 * class_metrics['precision'][cls] * class_metrics['recall'][cls] /
                (class_metrics['precision'][cls] + class_metrics['recall'][cls])
            )
    
    # Compute mAP
    # (A simplified version for now - in practice, we'd use PR curves)
    class_metrics['ap'] = class_metrics['precision'] * class_metrics['recall']
    
    # Compute overall metrics
    metrics = {
        'mAP': np.mean(class_metrics['ap'][class_metrics['count'] > 0]),
        'mF1': np.mean(class_metrics['f1'][class_metrics['count'] > 0]),
        'mPrecision': np.mean(class_metrics['precision'][class_metrics['count'] > 0]),
        'mRecall': np.mean(class_metrics['recall'][class_metrics['count'] > 0]),
        'by_class': {
            'precision': {cls: class_metrics['precision'][cls] for cls in range(num_classes) if class_metrics['count'][cls] > 0},
            'recall': {cls: class_metrics['recall'][cls] for cls in range(num_classes) if class_metrics['count'][cls] > 0},
            'f1': {cls: class_metrics['f1'][cls] for cls in range(num_classes) if class_metrics['count'][cls] > 0},
            'ap': {cls: class_metrics['ap'][cls] for cls in range(num_classes) if class_metrics['count'][cls] > 0},
            'count': {cls: int(class_metrics['count'][cls]) for cls in range(num_classes) if class_metrics['count'][cls] > 0}
        }
    }
    
    return metrics


def evaluate_staffline_detection(
    pred_stafflines: List[Dict],
    gt_stafflines: List[Dict],
    completeness_threshold: float = 0.8,
    spacing_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Evaluate staffline detection performance
    
    Args:
        pred_stafflines: List of predicted stafflines dictionaries
        gt_stafflines: List of ground truth stafflines dictionaries
        completeness_threshold: Threshold for staff completeness
        spacing_threshold: Threshold for staff spacing consistency
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Initialize metrics
    metrics = {
        'staff_detection_precision': 0.0,
        'staff_detection_recall': 0.0,
        'staff_detection_f1': 0.0,
        'staffline_detection_precision': 0.0,
        'staffline_detection_recall': 0.0,
        'staffline_detection_f1': 0.0,
        'staff_completeness': 0.0,
        'staff_spacing_consistency': 0.0
    }
    
    # Counters for staves
    total_pred_staves = 0
    total_gt_staves = 0
    matched_staves = 0
    
    # Counters for stafflines
    total_pred_stafflines = 0
    total_gt_stafflines = 0
    matched_stafflines = 0
    
    # Process each image
    for pred, gt in zip(pred_stafflines, gt_stafflines):
        pred_staves = pred['staffs']
        gt_staves = gt['staffs']
        
        # Update counters
        total_pred_staves += len(pred_staves)
        total_gt_staves += len(gt_staves)
        
        # Match staves
        for pred_staff in pred_staves:
            pred_staff_box = np.array(pred_staff['bbox']).reshape(1, 4)
            
            # Convert gt staff boxes
            gt_staff_boxes = np.array([s['bbox'] for s in gt_staves]).reshape(-1, 4)
            
            if len(gt_staff_boxes) > 0:
                # Compute IoU
                iou = compute_iou(pred_staff_box, gt_staff_boxes)[0]
                
                # Find best match
                best_idx = iou.argmax()
                best_iou = iou[best_idx]
                
                if best_iou >= 0.5:  # IoU threshold for staff matching
                    matched_staves += 1
                    
                    # Check staff completeness
                    pred_staff_lines = pred_staff['stafflines']
                    gt_staff_lines = gt_staves[best_idx]['stafflines']
                    
                    # Check if staff has 5 lines
                    if len(pred_staff_lines) == 5 and len(gt_staff_lines) == 5:
                        metrics['staff_completeness'] += 1
                    elif len(pred_staff_lines) / 5 >= completeness_threshold:
                        metrics['staff_completeness'] += len(pred_staff_lines) / 5
                    
                    # Check staffline spacing consistency
                    if len(pred_staff_lines) >= 2:
                        # Sort stafflines by y position
                        pred_staff_lines = sorted(pred_staff_lines, key=lambda x: x['y1'])
                        
                        # Calculate spacings
                        spacings = []
                        for i in range(1, len(pred_staff_lines)):
                            spacing = pred_staff_lines[i]['y1'] - pred_staff_lines[i-1]['y1']
                            spacings.append(spacing)
                        
                        # Calculate consistency
                        mean_spacing = np.mean(spacings)
                        std_spacing = np.std(spacings)
                        rel_std = std_spacing / mean_spacing
                        
                        if rel_std <= spacing_threshold:
                            metrics['staff_spacing_consistency'] += 1
        
        # Count stafflines
        pred_stafflines_list = [line for staff in pred_staves for line in staff['stafflines']]
        gt_stafflines_list = [line for staff in gt_staves for line in staff['stafflines']]
        
        total_pred_stafflines += len(pred_stafflines_list)
        total_gt_stafflines += len(gt_stafflines_list)
        
        # Match stafflines
        for pred_line in pred_stafflines_list:
            pred_line_box = np.array([
                pred_line['x1'],
                pred_line['y1'],
                pred_line['x2'] - pred_line['x1'],
                pred_line['y2'] - pred_line['y1']
            ]).reshape(1, 4)
            
            # Convert gt staffline boxes
            gt_line_boxes = np.array([
                [gl['x1'], gl['y1'], gl['x2'] - gl['x1'], gl['y2'] - gl['y1']]
                for gl in gt_stafflines_list
            ]).reshape(-1, 4)
            
            if len(gt_line_boxes) > 0:
                # Compute IoU
                iou = compute_iou(pred_line_box, gt_line_boxes)[0]
                
                # Find best match
                best_idx = iou.argmax()
                best_iou = iou[best_idx]
                
                if best_iou >= 0.5:  # IoU threshold for staffline matching
                    matched_stafflines += 1
    
    # Compute staff detection metrics
    if total_pred_staves > 0:
        metrics['staff_detection_precision'] = matched_staves / total_pred_staves
    
    if total_gt_staves > 0:
        metrics['staff_detection_recall'] = matched_staves / total_gt_staves
    
    if metrics['staff_detection_precision'] + metrics['staff_detection_recall'] > 0:
        metrics['staff_detection_f1'] = (
            2 * metrics['staff_detection_precision'] * metrics['staff_detection_recall'] /
            (metrics['staff_detection_precision'] + metrics['staff_detection_recall'])
        )
    
    # Compute staffline detection metrics
    if total_pred_stafflines > 0:
        metrics['staffline_detection_precision'] = matched_stafflines / total_pred_stafflines
    
    if total_gt_stafflines > 0:
        metrics['staffline_detection_recall'] = matched_stafflines / total_gt_stafflines
    
    if metrics['staffline_detection_precision'] + metrics['staffline_detection_recall'] > 0:
        metrics['staffline_detection_f1'] = (
            2 * metrics['staffline_detection_precision'] * metrics['staffline_detection_recall'] /
            (metrics['staffline_detection_precision'] + metrics['staffline_detection_recall'])
        )
    
    # Normalize completeness and consistency
    if matched_staves > 0:
        metrics['staff_completeness'] /= matched_staves
        metrics['staff_spacing_consistency'] /= matched_staves
    
    return metrics


def evaluate_relationship_prediction(
    pred_edges: List[Tuple[int, int]],
    pred_scores: np.ndarray,
    gt_edges: List[Tuple[int, int]],
    score_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate relationship prediction performance
    
    Args:
        pred_edges: List of predicted edge tuples (src, dst)
        pred_scores: Predicted edge scores
        gt_edges: List of ground truth edge tuples (src, dst)
        score_threshold: Threshold for positive predictions
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Threshold predictions
    positive_preds = pred_scores >= score_threshold
    pred_edges_binary = np.array(pred_edges)[positive_preds]
    
    # Convert to sets for easy comparison
    pred_edges_set = {(src, dst) for src, dst in pred_edges_binary}
    gt_edges_set = {(src, dst) for src, dst in gt_edges}
    
    # Count true positives, false positives, false negatives
    tp = len(pred_edges_set.intersection(gt_edges_set))
    fp = len(pred_edges_set - gt_edges_set)
    fn = len(gt_edges_set - pred_edges_set)
    
    # Compute metrics
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # Compute AUROC and AP for edge prediction
    # (Simplified versions)
    auroc = 0.0
    ap = 0.0
    
    if len(pred_scores) > 0:
        # Create binary labels for edges
        gt_labels = np.zeros(len(pred_edges), dtype=np.int32)
        for i, (src, dst) in enumerate(pred_edges):
            if (src, dst) in gt_edges_set:
                gt_labels[i] = 1
        
        # Sort by score
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_scores = pred_scores[sorted_indices]
        sorted_labels = gt_labels[sorted_indices]
        
        # Compute precision and recall at each threshold
        precision_at_thresh = []
        recall_at_thresh = []
        
        for i in range(len(sorted_scores)):
            pred_labels = sorted_scores >= sorted_scores[i]
            prec, rec, _, _ = precision_recall_fscore_support(
                sorted_labels, pred_labels, average='binary'
            )
            precision_at_thresh.append(prec)
            recall_at_thresh.append(rec)
        
        # Compute AP
        ap = np.sum(np.diff(np.append(0, recall_at_thresh)) * np.array(precision_at_thresh))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap,
        'auroc': auroc
    }


def evaluate_hierarchical_detection(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate hierarchical music notation detection performance
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        class_names: List of class names
        iou_threshold: IoU threshold for matching
        score_threshold: Score threshold for positive predictions
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Initialize metrics dictionary
    metrics = {
        'detection': {},
        'staffline': {},
        'relationship': {},
        'overall': {}
    }
    
    # Extract detection data
    pred_boxes_list = []
    pred_classes_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_classes_list = []
    
    for pred, gt in zip(predictions, ground_truths):
        if 'elements' in pred and 'elements' in gt:
            # Extract predictions
            pred_boxes = []
            pred_classes = []
            pred_scores = []
            
            for elem in pred['elements']:
                if 'score' in elem and elem['score'] < score_threshold:
                    continue
                    
                pred_boxes.append(elem['bbox'])
                pred_classes.append(elem['class_id'])
                
                if 'score' in elem:
                    pred_scores.append(elem['score'])
                else:
                    pred_scores.append(1.0)
            
            # Extract ground truths
            gt_boxes = [elem['bbox'] for elem in gt['elements']]
            gt_classes = [elem['class_id'] for elem in gt['elements']]
            
            # Convert to numpy arrays
            pred_boxes = np.array(pred_boxes)
            pred_classes = np.array(pred_classes)
            pred_scores = np.array(pred_scores)
            gt_boxes = np.array(gt_boxes)
            gt_classes = np.array(gt_classes)
            
            # Append to lists
            pred_boxes_list.append(pred_boxes)
            pred_classes_list.append(pred_classes)
            pred_scores_list.append(pred_scores)
            gt_boxes_list.append(gt_boxes)
            gt_classes_list.append(gt_classes)
    
    # Evaluate detection
    if pred_boxes_list:
        detection_metrics = evaluate_detection(
            pred_boxes_list,
            pred_classes_list,
            pred_scores_list,
            gt_boxes_list,
            gt_classes_list,
            len(class_names),
            iou_threshold
        )
        metrics['detection'] = detection_metrics
    
    # Evaluate staffline detection
    staffline_preds = [pred['stafflines'] for pred in predictions if 'stafflines' in pred]
    staffline_gts = [gt['stafflines'] for gt in ground_truths if 'stafflines' in gt]
    
    if staffline_preds and staffline_gts:
        staffline_metrics = evaluate_staffline_detection(
            staffline_preds,
            staffline_gts
        )
        metrics['staffline'] = staffline_metrics
    
    # Evaluate relationship prediction
    relationship_metrics = {}
    
    for pred, gt in zip(predictions, ground_truths):
        if 'relationships' in pred and 'relationships' in gt:
            # Extract edges and scores
            pred_edges = list(map(tuple, pred['relationships']['edge_indices'].T))
            pred_scores = pred['relationships']['edge_scores']
            gt_edges = list(map(tuple, gt['relationships']['edge_indices'].T))
            
            # Evaluate relationships
            rel_metrics = evaluate_relationship_prediction(
                pred_edges,
                pred_scores,
                gt_edges,
                score_threshold
            )
            
            # Accumulate metrics
            for k, v in rel_metrics.items():
                if k in relationship_metrics:
                    relationship_metrics[k].append(v)
                else:
                    relationship_metrics[k] = [v]
    
    # Average relationship metrics
    if relationship_metrics:
        metrics['relationship'] = {
            k: np.mean(v) for k, v in relationship_metrics.items()
        }
    
    # Compute overall metrics
    metrics['overall'] = {
        'mAP': metrics['detection'].get('mAP', 0.0),
        'staff_f1': metrics['staffline'].get('staff_detection_f1', 0.0),
        'relationship_f1': metrics['relationship'].get('f1', 0.0)
    }
    
    # Compute combined score
    metrics['overall']['combined_score'] = (
        0.5 * metrics['overall']['mAP'] +
        0.3 * metrics['overall']['staff_f1'] +
        0.2 * metrics['overall']['relationship_f1']
    )
    
    return metrics