import cv2
from config.settings import (
    CLAHE_CLIP_LIMIT, 
    CLAHE_TILE_GRID_SIZE, 
    DARK_THRESHOLD, 
    GAMMA_CORRECTION
)
from .clahe import apply_clahe
from .exposure import normalize_exposure

def preprocess_frame(image):
    """
    Orchestrates the full preprocessing pipeline for a camera frame.
    
    Pipeline Steps:
    1. Exposure Normalization (Gamma Correction)
    2. Local Contrast Enhancement (CLAHE)
    
    Args:
        image: Raw camera frame (BGR)
        
    Returns:
        Preprocessed frame ready for detection/recognition
    """
    if image is None:
        return None
        
    # 1. Normalize exposure if dark
    processed = normalize_exposure(
        image, 
        threshold=DARK_THRESHOLD, 
        gamma=GAMMA_CORRECTION
    )
    
    # 2. Apply CLAHE for local contrast boosting
    processed = apply_clahe(
        processed, 
        clip_limit=CLAHE_CLIP_LIMIT, 
        tile_grid_size=CLAHE_TILE_GRID_SIZE
    )
    
    return processed
