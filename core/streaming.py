import os
import cv2
import json
import time
import numpy as np
from pathlib import Path

class FrameBridge:
    """
    Ultra-low latency frame sharing between Engine and API using /dev/shm.
    This works only on Linux systems.
    """
    def __init__(self, bridge_name: str = "emsvia_stream"):
        self.shm_path = Path(f"/dev/shm/{bridge_name}.jpg")
        self.meta_path = Path(f"/dev/shm/{bridge_name}.json")
        
    def push(self, frame: np.ndarray, meta: dict = None):
        """
        Encode and write frame to shared memory.
        """
        try:
            # Encode to JPEG (Medium quality for speed/size balance)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            # Atomic write via temporary file (standard practice even in /dev/shm)
            temp_path = self.shm_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(buffer)
            temp_path.replace(self.shm_path)
            
            # Write metadata (Heartbeat + Stats)
            if meta is not None:
                meta['timestamp'] = time.time()
                with open(self.meta_path, "w") as f:
                    json.dump(meta, f)
        except Exception as e:
            # Silently fail in production if needed, or log locally
            pass

    def get(self):
        """
        Read raw bytes of the latest frame.
        Returns: (bytes, meta)
        """
        if not self.shm_path.exists():
            return None, None
            
        # Check if heartbeat is fresh (within 2 seconds)
        meta = self.get_meta()
        if not meta or (time.time() - meta.get('timestamp', 0) > 2.0):
            return None, None
            
        try:
            with open(self.shm_path, "rb") as f:
                return f.read(), meta
        except:
            return None, None

    def get_meta(self):
        """Read engine metadata/heartbeat."""
        if not self.meta_path.exists():
            return None
        try:
            with open(self.meta_path, "r") as f:
                return json.load(f)
        except:
            return None

    def clear(self):
        """Cleanup shared files."""
        if self.shm_path.exists(): self.shm_path.unlink()
        if self.meta_path.exists(): self.meta_path.unlink()
