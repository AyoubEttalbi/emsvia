"""
Utility functions for Face Recognition Attendance System
"""
import logging
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: Path, level: str = "INFO"):
    """
    Set up logging configuration with file and console handlers
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized - Level: {level}")


def save_unknown_face(face_image: np.ndarray, directory: Path) -> str:
    """
    Save unknown face image with timestamp
    
    Args:
        face_image: Face image array
        directory: Directory to save unknown faces
        
    Returns:
        Path to saved image
    """
    directory.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"unknown_{timestamp}.jpg"
    filepath = directory / filename
    
    cv2.imwrite(str(filepath), face_image)
    
    return str(filepath)


def validate_image_quality(image: np.ndarray, min_size: int = 160) -> tuple[bool, str]:
    """
    Validate image quality for face recognition
    
    Args:
        image: Image array
        min_size: Minimum image size
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if image is empty
    if image is None or image.size == 0:
        return False, "Empty image"
    
    # Check image size
    height, width = image.shape[:2]
    if height < min_size or width < min_size:
        return False, f"Image too small ({width}x{height}), minimum {min_size}x{min_size}"
    
    # Check for blur using Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < 100:  # Threshold for blur detection
        return False, f"Image too blurry (score: {blur_score:.2f})"
    
    # Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < 40:
        return False, f"Image too dark (brightness: {mean_brightness:.2f})"
    if mean_brightness > 220:
        return False, f"Image too bright (brightness: {mean_brightness:.2f})"
    
    return True, "OK"


def create_directories():
    """Create all required data directories"""
    from config.settings import (
        DATA_DIR, STUDENT_IMAGES_DIR, EMBEDDINGS_DIR, 
        UNKNOWN_FACES_DIR, DATABASE_DIR, LOGS_DIR
    )
    
    for directory in [DATA_DIR, STUDENT_IMAGES_DIR, EMBEDDINGS_DIR, 
                      UNKNOWN_FACES_DIR, DATABASE_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def calculate_fps(timestamps: list, window_size: int = 30) -> float:
    """
    Calculate FPS from timestamps
    
    Args:
        timestamps: List of frame timestamps
        window_size: Number of recent frames to consider
        
    Returns:
        FPS value
    """
    if len(timestamps) < 2:
        return 0.0
    
    recent_timestamps = timestamps[-window_size:]
    time_diff = recent_timestamps[-1] - recent_timestamps[0]
    
    if time_diff == 0:
        return 0.0
    
    return len(recent_timestamps) / time_diff
