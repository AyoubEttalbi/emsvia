"""
Camera Test Script - Verify camera hardware and face detection
"""
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

def list_available_cameras(max_cameras=5):
    """List all available camera devices"""
    print("Searching for available cameras...")
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"  Camera {i}: Available")
            cap.release()
    
    return available_cameras

def test_camera(camera_index=0):
    """
    Test camera and display live feed
    
    Args:
        camera_index: Camera device index
    """
    print(f"\n{'='*60}")
    print(f"Camera Test - Index: {camera_index}")
    print(f"{'='*60}\n")
    
    # Open camera
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        print(f"❌ ERROR: Cannot open camera {camera_index}")
        return False
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Get actual properties
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(camera.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Camera opened successfully!")
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    print(f"\nControls:")
    print(f"  - Press 'q' to quit")
    print(f"  - Press 's' to save screenshot")
    print(f"\nDisplaying live feed...\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add info overlay
            cv2.putText(
                frame, 
                f"Frame: {frame_count} | Press 'q' to quit | 's' to screenshot", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )
            
            # Display frame
            cv2.imshow(f'Camera Test - Index {camera_index}', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✅ Test completed successfully!")
                break
            elif key == ord('s'):
                screenshot_path = Path(__file__).parent.parent / "logs" / f"camera_test_{camera_index}.jpg"
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(screenshot_path), frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    print(f"\nTotal frames captured: {frame_count}")
    return True

def test_face_detection_basic():
    """Test basic face detection with Haar Cascade (lightweight)"""
    print(f"\n{'='*60}")
    print("Basic Face Detection Test")
    print(f"{'='*60}\n")
    
    # Load Haar Cascade (comes with OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ ERROR: Could not load Haar Cascade classifier")
        return False
    
    camera = cv2.VideoCapture(CAMERA_INDEX)
    
    if not camera.isOpened():
        print(f"❌ ERROR: Cannot open camera {CAMERA_INDEX}")
        return False
    
    print("✅ Face detection initialized")
    print("   Using Haar Cascade (OpenCV built-in)")
    print("\nLook at the camera to test face detection")
    print("Press 'q' to quit\n")
    
    faces_detected_count = 0
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Face Detected!",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            if len(faces) > 0:
                faces_detected_count += 1
            
            # Display info
            info_text = f"Faces: {len(faces)} | Total detections: {faces_detected_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Face Detection Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    if faces_detected_count > 0:
        print(f"\n✅ Face detection working! Detected faces in {faces_detected_count} frames")
        return True
    else:
        print("\n⚠️  No faces detected. Try adjusting lighting or camera position.")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM - CAMERA TEST")
    print("="*60)
    
    # List available cameras
    cameras = list_available_cameras()
    
    if not cameras:
        print("\n❌ ERROR: No cameras found!")
        print("Please ensure:")
        print("  1. Camera is connected")
        print("  2. Camera permissions are granted")
        print("  3. Camera is not being used by another application")
        sys.exit(1)
    
    print(f"\nFound {len(cameras)} camera(s)")
    
    # Test default camera
    print(f"\nTesting camera {CAMERA_INDEX}...")
    success = test_camera(CAMERA_INDEX)
    
    if not success:
        sys.exit(1)
    
    # Ask if user wants to test face detection
    print(f"\n{'='*60}")
    response = input("\nTest face detection? (y/n): ").lower()
    
    if response == 'y':
        test_face_detection_basic()
    
    print(f"\n{'='*60}")
    print("Camera test complete!")
    print("="*60 + "\n")
