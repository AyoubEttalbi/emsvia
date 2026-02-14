import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.face_recognizer import FaceRecognizer

@pytest.fixture
def recognizer():
    return FaceRecognizer(model_name="Facenet512")

def test_initialization(recognizer):
    """Test successful initialization."""
    assert recognizer.model_name == "Facenet512"

@patch("models.face_recognizer.DeepFace")
def test_generate_embedding_success(mock_deepface, recognizer):
    """Test successful embedding generation."""
    # Mock return value of DeepFace.represent
    mock_deepface.represent.return_value = [{"embedding": [0.1] * 512}]
    
    # Create fake image
    fake_img = np.zeros((160, 160, 3), dtype=np.uint8)
    
    embedding = recognizer.generate_embedding(fake_img)
    
    assert embedding is not None
    assert embedding.shape == (512,)
    assert embedding[0] == 0.1

def test_generate_embedding_empty_image(recognizer):
    """Test handling of empty image."""
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    embedding = recognizer.generate_embedding(empty_img)
    assert embedding is None

def test_calculate_distance_cosine(recognizer):
    """Test cosine distance calculation."""
    # Zero distance for identical vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    dist = recognizer.calculate_distance(v1, v2, metric="cosine")
    assert dist == 0.0

    # Max distance (2.0) for opposite vectors
    v3 = np.array([-1, 0, 0])
    dist_opposite = recognizer.calculate_distance(v1, v3, metric="cosine")
    assert np.isclose(dist_opposite, 2.0)

def test_find_best_match(recognizer):
    """Test finding best match in database."""
    target = np.array([1.0, 0.0])
    
    # DB with one close match and one far match
    db = {
        1: [np.array([1.0, 0.1])], # Very close
        2: [np.array([0.0, 1.0])]  # Far (orthogonal)
    }
    
    # Result should be ID 1
    result = recognizer.find_best_match(target, db)
    
    assert result["match_found"] is True
    assert result["student_id"] == 1
    assert result["distance"] < recognizer.THRESHOLD

def test_no_match_found(recognizer):
    """Test case where no match satisfies threshold."""
    target = np.array([1.0, 0.0])
    
    # DB with only far matches
    db = {
        2: [np.array([0.0, 1.0])] # Perpendicular (dist ~1.0 > 0.30)
    }
    
    result = recognizer.find_best_match(target, db)
    
    assert result["match_found"] is False
    assert result["student_id"] is None
