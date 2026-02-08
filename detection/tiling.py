import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

def get_tiles(image: np.ndarray, tile_size: Tuple[int, int] = (1080, 1080), overlap: float = 0.2) -> List[Dict[str, Any]]:
    """
    Divide a frame into overlapping tiles for better detection of small faces.
    
    Args:
        image: Original BGR image (usually 1920x1080)
        tile_size: Size of each square tile (width, height)
        overlap: Percentage of overlap between tiles (0.0 to 1.0)
        
    Returns:
        List of dictionaries: {'tile': np.ndarray, 'origin': (x, y)}
    """
    if image is None:
        return []
        
    h, w = image.shape[:2]
    tw, th = tile_size
    
    # Calculate step size based on overlap
    step_x = int(tw * (1 - overlap))
    step_y = int(th * (1 - overlap))
    
    tiles = []
    
    # Iterate through the image and extract tiles
    # For a standard 1080p frame (1920x1080) with 1080x1080 tiles:
    # We usually want 2 tiles horizontally (left, right) with overlap
    for y in range(0, max(1, h - th + step_y), step_y):
        for x in range(0, max(1, w - tw + step_x), step_x):
            # Ensure we don't go out of bounds
            x_end = min(x + tw, w)
            y_end = min(y + th, h)
            
            # Start coordinate might need adjustment if we are at the very end
            x_start = max(0, x_end - tw)
            y_start = max(0, y_end - th)
            
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append({
                'tile': tile,
                'origin': (x_start, y_start)
            })
            
            # If we reached the edge, stop for this row/column
            if x_end == w: break
        if y_end == h: break
            
    return tiles

def map_to_original(box: List[int], origin: Tuple[int, int]) -> List[int]:
    """
    Map bounding box coordinates from a tile back to the original image reference.
    
    Args:
        box: [x, y, w, h] in tile coordinates
        origin: (x_offset, y_offset) of the tile in the original image
        
    Returns:
        [x, y, w, h] in original image coordinates
    """
    tx, ty = origin
    x, y, w, h = box
    return [x + tx, y + ty, w, h]

def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    Perform Non-Maximum Suppression (NMS) to remove duplicate detections.
    
    Args:
        detections: List of detection dicts with 'box' and 'confidence'
        iou_threshold: Intersection over Union threshold
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
        
    # Extract boxes and scores
    boxes = np.array([d['box'] for d in detections])
    scores = np.array([d['confidence'] for d in detections])
    
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return [detections[i] for i in keep]
