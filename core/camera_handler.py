import cv2
import threading
import time
import logging
from typing import Optional, Tuple, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

def detect_available_cameras(max_index: int = 5) -> List[Dict]:
    """
    Detect all available cameras and return their info.
    
    Returns:
        List of dicts with 'index', 'name', and 'working' keys
    """
    cameras = []
    
    for i in range(max_index + 1):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    backend_name = cap.getBackendName()
                    cameras.append({
                        'index': i,
                        'name': f"{backend_name} (Index {i})",
                        'working': True
                    })
                    logger.info(f"Found camera at index {i}: {backend_name}")
                cap.release()
        except Exception as e:
            logger.debug(f"Camera {i} not available: {e}")
    
    return cameras

def select_camera(cameras: List[Dict]) -> int:
    """
    Let user select a camera from available options.
    
    Args:
        cameras: List of available camera dicts
        
    Returns:
        Selected camera index
    """
    if not cameras:
        print("No cameras found!")
        return 0
    
    if len(cameras) == 1:
        print(f"Using only available camera: {cameras[0]['name']}")
        return cameras[0]['index']
    
    print("\n" + "=" * 50)
    print("📷 Available Cameras:")
    print("=" * 50)
    for cam in cameras:
        print(f"  [{cam['index']}] {cam['name']}")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\nSelect camera index (or press Enter for default 0): ").strip()
            if choice == "":
                return 0
            idx = int(choice)
            if any(c['index'] == idx for c in cameras):
                return idx
            print(f"Invalid index. Choose from: {[c['index'] for c in cameras]}")
        except ValueError:
            print("Please enter a valid number.")

class CameraHandler:
    """
    Handles camera operations in a separate thread for low-latency frame access.
    """
    
    def __init__(self, source: int = 0, resolution: Tuple[int, int] = (640, 480), auto_select: bool = False):
        """
        Initialize the Camera Handler.
        
        Args:
            source: Camera index or video path.
            resolution: Desired capture resolution (width, height).
            auto_select: If True and source is int, scan for cameras.
        """
        self.source = source
        self.resolution = resolution
        
        # Auto-detect and select camera if needed
        if auto_select and isinstance(source, int):
            cameras = detect_available_cameras()
            if cameras:
                self.source = select_camera(cameras)
        
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera source: {source}")
            raise RuntimeError(f"Could not open camera {source}")
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        
        # FPS calculation
        self.prev_time = 0
        self.fps = 0
        
    def start(self):
        """Start the camera capture thread."""
        t = threading.Thread(target=self._update, args=(), daemon=True)
        t.start()
        
        # Wait for first frame (up to 3 seconds)
        for _ in range(30):
            with self.lock:
                if self.ret and self.frame is not None:
                    logger.info(f"Camera {self.source} started successfully.")
                    return self
            time.sleep(0.1)
        
        logger.warning(f"Camera {self.source} started but first frame is slow.")
        return self

    def _update(self):
        """Internal loop to continuously grab frames."""
        while not self.stopped:
            ret, frame = self.cap.read()
            
            with self.lock:
                self.ret = ret
                self.frame = frame
                
            if not ret:
                time.sleep(0.1)
                continue
                
            # Calculate FPS
            curr_time = time.time()
            if self.prev_time > 0:
                time_diff = curr_time - self.prev_time
                if time_diff > 0:
                    self.fps = 1.0 / time_diff
            self.prev_time = curr_time
            
            # Tiny sleep to prevent 100% CPU usage
            time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame.
        
        Returns:
            Tuple of (success, frame).
        """
        with self.lock:
            return self.ret, self.frame

    def get_fps(self) -> float:
        """Return the current capture FPS."""
        return self.fps

    def stop(self):
        """Stop the camera thread and release resources."""
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()
        logger.info(f"Camera {self.source} stopped and released.")

    def preprocess_frame(self, frame: np.ndarray, flip_horizontal: bool = True) -> np.ndarray:
        """
        Basic preprocessing for the frame.
        """
        if frame is None:
            return None
        
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
            
        return frame
