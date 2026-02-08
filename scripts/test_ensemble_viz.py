"""
Visualization script for Ensemble Detection.
Shows detections from different models with different colors.
"""
import cv2
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, USE_ENSEMBLE, ENSEMBLE_DETECTORS
from models.face_detector import FaceDetector

def main():
    print(f"Starting Ensemble Detection Visualization...")
    print(f"Models: {ENSEMBLE_DETECTORS}")
    print(f"Ensemble Enabled: {USE_ENSEMBLE}")
    print("Press 'q' to quit.")
    
    # Initialize detector
    detector = FaceDetector()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Colors for different models (BGR)
    MODEL_COLORS = {
        'retinaface': (0, 255, 0),    # Green
        'mtcnn': (255, 0, 0),         # Blue
        'opencv': (0, 0, 255),        # Red
        'yolov8': (255, 255, 0),      # Cyan
        'fused': (0, 255, 255)        # Yellow
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            
            start_time = time.time()
            # Detect faces
            detections = detector.detect_faces(frame)
            end_time = time.time()
            
            process_time = (end_time - start_time) * 1000 # ms
            fps = 1.0 / (end_time - start_time)

            # Draw detections
            for det in detections:
                x, y, w, h = det['box']
                conf = det['confidence']
                agreement = det.get('agreement_score', 1)
                
                # Choose color based on agreement or model
                color = MODEL_COLORS.get('fused' if agreement > 1 else 'retinaface', (0, 255, 0))
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Label
                label = f"{conf:.2f} (Agr: {agreement})"
                cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Info overlay
            cv2.putText(display_frame, f"Faces: {len(detections)} | Time: {process_time:.0f}ms | FPS: {fps:.1f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show list of detectors
            y0, dy = 60, 25
            for i, model in enumerate(ENSEMBLE_DETECTORS):
                cv2.putText(display_frame, f"- {model}", (10, y0 + i*dy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Ensemble Detection Test", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Visualization test closed.")

if __name__ == "__main__":
    main()
