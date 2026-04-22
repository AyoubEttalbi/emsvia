
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Mock environment for models
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Use CPU for quick test

from models.face_recognizer import FaceRecognizer

def test_normalization():
    print("Testing embedding normalization...")
    fr = FaceRecognizer(model_names=["ArcFace"])
    
    # Create a dummy image
    img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    print("Generating embedding...")
    emb = fr.generate_embedding(img)
    
    if emb is None:
        print("❌ Failed to generate embedding (check DeepFace setup)")
        return
    
    norm = np.linalg.norm(emb)
    print(f"Embedding Norm: {norm}")
    
    # Allow small floating point variance
    if np.isclose(norm, 1.0, atol=1e-5):
        print("✅ SUCCESS: Embedding is correctly normalized to 1.0")
    else:
        print(f"❌ FAILURE: Embedding norm is {norm}, expected 1.0")

if __name__ == "__main__":
    test_normalization()
