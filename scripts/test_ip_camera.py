import cv2
import time

def test_ip_camera(url):
    """
    Tests if WSL can reach a camera via a network URL (IP Camera, OBS RTSP, etc.)
    """
    print(f"--- WSL IP Camera Test ---")
    print(f"Attempting to connect to: {url}")
    
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Result: FAILED. Could not connect to the stream.")
        print("Check if:")
        print("1. The IP/URL is correct.")
        print("2. Your Windows firewall is allowing WSL to talk to this port.")
        return

    print("Result: SUCCESS! Connection established.")
    
    start_time = time.time()
    frames = 0
    while time.time() - start_time < 5: # Test for 5 seconds
        ret, frame = cap.read()
        if not ret:
            print("Dropped frame...")
            continue
        
        frames += 1
        cv2.imshow("IP Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {frames} frames in 5 seconds.")

if __name__ == "__main__":
    # Example for DroidCam or IP Webcam: "http://192.168.1.50:4747/video"
    # Example for OBS RTSP: "rtsp://192.168.1.10:554/live"
    url = input("Enter your IP Camera URL (e.g., http://XXX.XXX.X.XX:8080/video): ").strip()
    if url:
        test_ip_camera(url)
    else:
        print("No URL provided.")
