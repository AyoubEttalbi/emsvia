import cv2
import numpy as np

def is_dark(image, threshold=50):
    """
    Check if an image is too dark based on mean pixel intensity.
    
    Args:
        image: BGR image
        threshold: Mean intensity threshold (0-255)
        
    Returns:
        True if the image is considered dark
    """
    if image is None:
        return False
        
    # Convert to grayscale to calculate mean intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    
    return mean_intensity < threshold

def normalize_exposure(image, threshold=50, gamma=2.2):
    """
    Apply gamma correction to normalize exposure if the image is too dark.
    
    Args:
        image: BGR image
        threshold: Intensity threshold below which to apply correction
        gamma: Gamma value (usually > 1.0 to brighten)
        
    Returns:
        Exposure-normalized BGR image
    """
    if image is None:
        return None
        
    if not is_dark(image, threshold):
        return image
        
    # Apply gamma correction
    # Formula: adjusted = ((img/255) ^ (1/gamma)) * 255
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)
