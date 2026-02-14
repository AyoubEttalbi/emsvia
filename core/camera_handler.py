import cv2
import threading
import time
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class CameraHandler:
    """
    Handles camera operations in a separate thread for low-latency frame access.
    """
    
    def __init__(self, source: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize the Camera Handler.
        
        Args:
            source: Camera index or video path.
            resolution: Desired capture resolution (width, height).
        """
        self.source = source
        self.resolution = resolution
        self.cap = cv2.VideoCapture(source)
        
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
        logger.info(f"Camera thread started for source {self.source}")
        return self

    def _update(self):
        """Internal loop to continuously grab frames."""
        while not self.stopped:
            ret, frame = self.cap.read()
            
            with self.lock:
                self.ret = ret
                self.frame = frame
                
            # Calculate FPS
            curr_time = time.time()
            if self.prev_time > 0:
                time_diff = curr_time - self.prev_time
                if time_diff > 0:
                    self.fps = 1.0 / time_diff
            self.prev_time = curr_time
            
            # Tiny sleep to prevent 100% CPU usage if capture is faster than needed
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
            
        # Apply Low Light Enhancement (CLAHE)
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge and convert back to BGR
        limg = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
        return frame
