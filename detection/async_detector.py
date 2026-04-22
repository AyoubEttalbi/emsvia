import threading
import time
import queue
import cv2
import numpy as np

class AsyncDetector:
    """
    Wraps any detector (ParallelDetector or FaceDetector) in a background thread.
    This allows the main video loop to run at full camera FPS (30+) while
    detection runs at its own pace (e.g., 3-10 FPS) without blocking.
    """
    def __init__(self, detector):
        self.detector = detector
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.last_result = [] # Start with empty detections
        self.lock = threading.Lock()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)

    def detect_async(self, frame):
        """
        Non-blocking submit. 
        1. If the background thread is idle (queue empty), submit new frame.
        2. If result is ready, grab it.
        Returns: The *latest available* detections.
        """
        # 1. Try to submit new frame if slot open
        if not self.frame_queue.full():
            try:
                # We must copy the frame because it might be modified/overwritten in main loop
                self.frame_queue.put_nowait(frame.copy()) 
            except queue.Full:
                pass 
        
        # 2. Check for new results
        try:
            # If new result ready, update our 'last_result'
            new_result = self.result_queue.get_nowait()
            with self.lock:
                self.last_result = new_result
        except queue.Empty:
            pass
            
        with self.lock:
            return self.last_result

    def _process_loop(self):
        while not self.stopped:
            try:
                # Wait for a frame
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # --- Heavy Lifting (300ms+) ---
            try:
                if hasattr(self.detector, 'detect_ensemble'):
                    detections = self.detector.detect_ensemble(frame, use_tiling=False)
                else:
                    detections = self.detector.detect_faces(frame)
            except Exception as e:
                print(f"Async Detection Error: {e}")
                detections = []
            
            # --- Submit Result ---
            # Clear old result if exists (we only care about freshest)
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.result_queue.put(detections)
