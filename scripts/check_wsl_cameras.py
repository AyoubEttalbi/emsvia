import cv2
import os
import glob

def check_cameras():
    print("--- WSL Camera Detection Test ---")
    
    # 1. Check for device nodes in Linux filesystem
    print("\nChecking /dev/video* devices...")
    video_devices = glob.glob("/dev/video*")
    if not video_devices:
        print("Result: No /dev/video devices found. WSL does not see any hardware cameras.")
        print("Tip: If using WSL2, you MUST use 'usbipd-win' to attach your USB camera to WSL.")
    else:
        for dev in sorted(video_devices):
            print(f"Found: {dev}")

    # 2. Attempt to open with OpenCV
    print("\nTesting OpenCV VideoCapture indices (0-5)...")
    found_any = False
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Index {i}: SUCCESS (Camera opened and frame captured!)")
                found_any = True
            else:
                print(f"Index {i}: PARTIAL (Opened, but failed to read frame)")
            cap.release()
        else:
            print(f"Index {i}: FAILED")
    
    if not found_any:
        print("\nConclusion: No working cameras detected via OpenCV.")
        print("If you are using virtual cameras (Camo, Nvidia Broadcast), they do NOT show up in WSL by default.")
    else:
        print("\nConclusion: At least one camera is working!")

if __name__ == "__main__":
    check_cameras()
