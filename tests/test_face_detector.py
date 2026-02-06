import pytest
import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.face_detector import FaceDetector

@pytest.fixture
def detector():
    """Fixture to initialize FaceDetector."""
    return FaceDetector(min_confidence=0.5, min_face_size=(20, 20))

@pytest.fixture
def sample_image():
    """Create a sample synthetic image with a 'face-like' structure."""
    # Create valid 3-channel image (100x100)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Draw a rectangle mimicking a face (simple visual for debugging, meaningful for basic ops)
    cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)
    return image

def test_initialization():
    """Test successful initialization."""
    det = FaceDetector()
    assert det.detector is not None

def test_extract_face_valid(detector, sample_image):
    """Test face extraction with valid parameters."""
    box = [30, 30, 40, 40] # x, y, w, h
    extracted = detector.extract_face(sample_image, box, target_size=(50, 50))
    
    assert extracted is not None
    assert extracted.shape == (50, 50, 3)

def test_extract_face_out_of_bounds(detector, sample_image):
    """Test face extraction handling boundaries (padding should handle this)."""
    box = [0, 0, 50, 50] # Top-left corner
    extracted = detector.extract_face(sample_image, box, target_size=(50, 50))
    
    assert extracted is not None
    assert extracted.shape == (50, 50, 3)

def test_quality_check_dark(detector):
    """Test quality check for dark image."""
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8) # Black image
    result = detector.check_image_quality(dark_img)
    
    assert result["passed"] is False
    assert result["is_too_dark"] is True

def test_quality_check_blur(detector):
    """Test quality check for flat color image (extremely blurry/no detail)."""
    flat_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = detector.check_image_quality(flat_img)
    
    assert result["passed"] is False
    assert result["is_blurry"] is True

def test_detect_face_empty(detector):
    """Test detection on empty image."""
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    detections = detector.detect_faces(empty_img)
    assert detections == []
