"""
Visualization script to test preprocessing effects on live camera feed.
Displays "Before" and "After" frames side-by-side.
"""
import cv2
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
from preprocessing.pipeline import preprocess_frame

def main():
    print("Starting Preprocessing Visualization Test...")
    print("Close the window or press 'q' to quit.")
    
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
                print("Failed to capture frame")
                break

            # Process frame
            processed_frame = preprocess_frame(frame)

            # Combine side-by-side
            # Resize if necessary to fit screen (optional, but good for large resolutions)
            display_h, display_w = 480, 640
            frame_res = cv2.resize(frame, (display_w, display_h))
            processed_res = cv2.resize(processed_frame, (display_w, display_h))
            
            combined = np.hstack((frame_res, processed_res))

            # Add labels
            cv2.putText(combined, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined, "PREPROCESSED", (display_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show stats
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            cv2.putText(combined, f"Mean Intensity: {mean_val:.1f}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Preprocessing Comparison (Before vs After)", combined)

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
