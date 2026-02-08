"""
Visualization script for Super-Resolution.
Detects a face, shows original crop vs enhanced crop.
"""
import cv2
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, USE_SUPER_RESOLUTION
from models.face_detector import FaceDetector

def main():
    print(f"Starting Super-Resolution Visualization...")
    print(f"SR Enabled: {USE_SUPER_RESOLUTION}")
    print("This demo will show the Original vs Enhanced face side-by-side.")
    print("Press 'q' to quit.")
    
    # Initialize detector (which now has integrated SR)
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
            
            display_frame = frame.copy()
            
            # Detect faces
            detections = detector.detect_faces(frame, use_tiling=False) 
            
            if detections:
                # Just take the largest face for demo
                det = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                x, y, w, h = det['box']
                
                # Draw box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract ORIGINAL face (no enhancement)
                # We access the raw extraction logic
                face_crop_orig = frame[y:y+h, x:x+w]
                
                # Apply Enhancement explicitly to visualize it
                # (The detector does it automatically during extract_face, but we want to see the difference)
                face_crop_enhanced = detector.enhancer.enhance_face(face_crop_orig)
                
                # Resize original to match enhanced size for display
                h_e, w_e = face_crop_enhanced.shape[:2]
                face_crop_orig_resized = cv2.resize(face_crop_orig, (w_e, h_e), interpolation=cv2.INTER_NEAREST)
                
                # Stack them horizontally: Left (Original), Right (Enhanced)
                comparison = np.hstack((face_crop_orig_resized, face_crop_enhanced))
                
                # Show comparison in a separate window
                cv2.imshow("Original (Resized) vs Enhanced (SR)", comparison)
                
            cv2.imshow("Main Feed", display_frame)

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
