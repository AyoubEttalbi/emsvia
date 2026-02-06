import cv2
import time
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from core.camera_handler import CameraHandler

def test_camera_handler():
    print("Initializing CameraHandler...")
    try:
        handler = CameraHandler(source=0)
        handler.start()
        print("Camera thread started.")
        
        start_time = time.time()
        while time.time() - start_time < 5: # Test for 5 seconds
            success, frame = handler.read()
            if success and frame is not None:
                # Preprocess
                display_frame = handler.preprocess_frame(frame)
                
                # Show FPS
                fps = handler.get_fps()
                cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("CameraHandler Test", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        handler.stop()
        cv2.destroyAllWindows()
        print("CameraHandler test completed.")
        
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_camera_handler()
