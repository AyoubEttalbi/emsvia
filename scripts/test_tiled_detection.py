"""
Visualization script to test tiled multi-scale detection.
Displays detections on the full frame with tiling feedback.
"""
import cv2
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, USE_TILING
from models.face_detector import FaceDetector

def main():
    print(f"Starting Tiled Detection Visualization (Tiling: {USE_TILING})...")
    print("This will test RetinaFace + Tiling. Press 'q' to quit.")
    
    # Initialize detector
    detector = FaceDetector()
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            # Detect faces using the new tiled pipeline
            detections = detector.detect_faces(frame)
            end_time = time.time()
            
            fps = 1.0 / (end_time - start_time)

            # Draw detections
            for det in detections:
                x, y, w, h = det['box']
                conf = det['confidence']
                
                # Green box for detections
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Confidence label
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Info overlay
            cv2.putText(frame, f"Faces: {len(detections)} | FPS: {fps:.1f} | Tiling: {USE_TILING}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Tiled Detection Test", frame)

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
