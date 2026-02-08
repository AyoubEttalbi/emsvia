import cv2
import numpy as np
from typing import List, Dict, Any
import logging
from .tiling import apply_nms

logger = logging.getLogger(__name__)

def fuse_detections(all_detections: List[Dict[str, Any]], iou_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    Fuse detections from multiple models and assign confidence scores based on agreement.
    
    Logic:
    1. Collect all detections from all models.
    2. Apply NMS across all detections.
    3. Assign new confidence scores:
       - Detected by multiple models: higher confidence.
       - Detected by a single model: original confidence or weighted lower.
       
    Args:
        all_detections: Combined detections from multiple models
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Fused and filtered detections.
    """
    if not all_detections:
        return []

    # Perform NMS to merge overlapping boxes from different models
    # apply_nms already sorts by confidence
    fused = apply_nms(all_detections, iou_threshold=iou_threshold)
    
    # Optional: Bonus confidence for multi-model agreement
    # (Checking which original detections were 'merged' into the fused ones)
    for f_det in fused:
        f_box = f_det['box']
        match_count = 0
        
        # Check against all original detections to see how many models saw this face
        for o_det in all_detections:
            o_box = o_det['box']
            iou = calculate_iou(f_box, o_box)
            if iou > 0.5: # Agreement threshold
                match_count += 1
        
        # Adjust confidence based on agreement (simplified model)
        # 1 model: 0.5, 2 models: 0.8, 3+ models: 1.0
        # However, we'll stick to a simple weighted approach for now
        if match_count >= 2:
            f_det['confidence'] = min(1.0, f_det['confidence'] * 1.1)
            f_det['agreement_score'] = match_count
        else:
            f_det['agreement_score'] = 1
            
    return fused

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two boxes [x, y, w, h]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
