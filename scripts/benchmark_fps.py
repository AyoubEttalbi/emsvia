import cv2
import time
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.face_detector import FaceDetector
from config.settings import FACE_DETECTION_MODEL

def benchmark():
    print(f"Benchmarking detector: {FACE_DETECTION_MODEL}")
    detector = FaceDetector()
    
    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white rectangle to simulate a "face" spot (though Haar won't find it, it exercises the loop)
    cv2.rectangle(frame, (200, 200), (400, 400), (255, 255, 255), -1)
    
    start_time = time.time()
    iterations = 50
    
    for i in range(iterations):
        _ = detector.detect_faces(frame)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} frames...")
            
    end_time = time.time()
    total_time = end_time - start_time
    fps = iterations / total_time
    
    print(f"\nBenchmark Result:")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {fps:.2f}")

if __name__ == "__main__":
    benchmark()
