import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    
    This technique boosts local contrast and helps recover faces from shadows.
    We convert to LAB color space and apply CLAHE only to the Luminance (L) channel
    to avoid color distortion.
    
    Args:
        image: BGR image (OpenCV format)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced BGR image
    """
    if image is None:
        return None
        
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to the L channel
    l_enhanced = clahe.apply(l)
    
    # Merge channels back
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    
    # Convert back to BGR
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result
