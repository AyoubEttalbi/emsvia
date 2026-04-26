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
    def __init__(
        self,
        bridge_name: str = "emsvia_stream",
        target_fps: int = 15,
        jpeg_quality: int = 65,
        stream_width: int | None = None,
        stream_height: int | None = None,
    ):
        # Double-buffer to avoid partial reads and expensive atomic renames each frame.
        # Writer alternates between A/B then flips meta to the newest buffer.
        self.shm_a_path = Path(f"/dev/shm/{bridge_name}.a.jpg")
        self.shm_b_path = Path(f"/dev/shm/{bridge_name}.b.jpg")
        self.meta_path = Path(f"/dev/shm/{bridge_name}.json")
        self._flip = False
        self.target_fps = max(1, int(target_fps)) if target_fps else 15
        self.jpeg_quality = int(jpeg_quality)
        self._min_push_interval_s = 1.0 / float(self.target_fps)
        self._last_push_ts = 0.0
        self.stream_width = int(stream_width) if stream_width else None
        self.stream_height = int(stream_height) if stream_height else None
        
    def push(self, frame: np.ndarray, meta: dict = None):
        """
        Encode and write frame to shared memory.
        """
        try:
            # Throttle publishing to keep browser stream smooth even if engine runs 40+ FPS.
            now = time.time()
            if (now - self._last_push_ts) < self._min_push_interval_s:
                return
            self._last_push_ts = now

            # Downscale ONLY for streaming to reduce JPEG encode/decode cost.
            if frame is not None and (self.stream_width or self.stream_height):
                h, w = frame.shape[:2]
                tw = self.stream_width
                th = self.stream_height
                if tw and th:
                    new_w, new_h = tw, th
                elif tw:
                    new_w = tw
                    new_h = int((tw / float(w)) * h)
                else:
                    new_h = th
                    new_w = int((th / float(h)) * w)
                if new_w > 0 and new_h > 0 and (new_w != w or new_h != h):
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode to JPEG (slightly lower quality to reduce bandwidth/CPU spikes in browser streaming)
            t0 = time.time()
            ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                return

            target = self.shm_a_path if self._flip else self.shm_b_path
            self._flip = not self._flip

            # Write directly to target buffer (reader uses meta to pick the newest buffer)
            with open(target, "wb") as f:
                f.write(buffer)
            
            # Write metadata (Heartbeat + Stats)
            if meta is not None:
                meta['timestamp'] = time.time()
                meta['buffer'] = 'a' if target == self.shm_a_path else 'b'
                meta['jpeg_bytes'] = int(len(buffer))
                meta['jpeg_encode_ms'] = round((time.time() - t0) * 1000.0, 2)
                meta['stream_fps_target'] = self.target_fps
                if self.stream_width or self.stream_height:
                    meta['stream_size'] = [frame.shape[1], frame.shape[0]]
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
        # Check if heartbeat is fresh (within 2 seconds)
        meta = self.get_meta()
        if not meta or (time.time() - meta.get('timestamp', 0) > 2.0):
            return None, None

        buf = meta.get("buffer")
        shm_path = self.shm_a_path if buf == "a" else self.shm_b_path
        if not shm_path.exists():
            # Fallback: if buffer missing, try the other.
            alt = self.shm_b_path if shm_path == self.shm_a_path else self.shm_a_path
            if not alt.exists():
                return None, None
            shm_path = alt
            
        try:
            with open(shm_path, "rb") as f:
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
        if self.shm_a_path.exists(): self.shm_a_path.unlink()
        if self.shm_b_path.exists(): self.shm_b_path.unlink()
        if self.meta_path.exists(): self.meta_path.unlink()
