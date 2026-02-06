import cv2
import tensorflow as tf
from deepface import DeepFace
import sys

print("-" * 50)
print("Face Recognition System Setup Verification")
print("-" * 50)

# OpenCV Check
print(f"OpenCV Version: {cv2.__version__}")

# TensorFlow Check
print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpu_devices) > 0}")
for i, device in enumerate(gpu_devices):
    print(f"  GPU {i}: {device.name}")

# DeepFace Check
try:
    # Just checking if it imports and can access models attribute or similar
    print("DeepFace: Successfully imported")
except Exception as e:
    print(f"DeepFace Check Failed: {e}")

# Camera Test (Check if index 0 is accessible)
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Camera: Success! Captured a test frame.")
    else:
        print("Camera: Warning! Cap opened but failed to read frame.")
    cap.release()
else:
    print("Camera: Error! Could not open camera index 0. (Expected in some WSL setups without USB pass-through)")

print("-" * 50)
print("Verification Complete")
print("-" * 50)
