import threading
import queue
import logging
import time
import json
import numpy as np

logger = logging.getLogger(__name__)

# #region agent log
def _debug_log(location, message, data, hypothesis_id=None, run_id="debug"):
    try:
        log_entry = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id
        }
        if hypothesis_id:
            log_entry["hypothesisId"] = hypothesis_id
        with open("/home/ayoub/projects/Ai-Ml/emsvia/.cursor/debug.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
# #endregion

class AsyncRecognizer:
    """
    Runs face recognition (crop + quality check + embedding + matching) in a 
    background thread so the main display loop never blocks.
    
    Usage:
        async_rec = AsyncRecognizer(detector, recognizer, em_manager)
        async_rec.start()
        
        # In main loop:
        async_rec.submit(track_id, frame, box)   # non-blocking
        result = async_rec.get_result(track_id)   # returns None or match_result
    """
    
    def __init__(self, detector, recognizer, em_manager):
        self.detector = detector
        self.recognizer = recognizer
        self.em_manager = em_manager
        
        self.task_queue = queue.Queue(maxsize=4)
        self.results = {}  # track_id -> match_result
        self.results_lock = threading.Lock()
        self.busy = False  # Is the worker currently processing?
        self.busy_lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
    
    def start(self):
        self.thread.start()
        return self
    
    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2.0)
    
    def submit(self, track_id, frame, box):
        """
        Submit a recognition job. Non-blocking.
        If the worker is busy or queue is full, the job is silently dropped.
        """
        with self.busy_lock:
            if self.busy:
                return  # Worker is busy, skip this frame
        
        try:
            self.task_queue.put_nowait((track_id, frame.copy(), box))
        except queue.Full:
            pass  # Drop if queue is full
    
    def get_result(self, track_id):
        """
        Get the latest recognition result for a track. Non-blocking.
        Returns None if no result is available yet.
        """
        with self.results_lock:
            return self.results.pop(track_id, None)
    
    def _worker(self):
        """Background worker that processes recognition tasks."""
        while not self.stopped:
            try:
                track_id, frame, box = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            with self.busy_lock:
                self.busy = True
            
            try:
                # #region agent log
                _debug_log("async_recognizer.py:76", "recognition_start", {
                    "track_id": track_id,
                    "frame_shape": frame.shape if frame is not None else None,
                    "box": box
                }, hypothesis_id="H1")
                # #endregion
                
                # 1. Extract face crop
                face_crop = self.detector.extract_face(frame, box)
                if face_crop is None:
                    # #region agent log
                    _debug_log("async_recognizer.py:78", "face_extraction_failed", {
                        "track_id": track_id
                    }, hypothesis_id="H1")
                    # #endregion
                    continue

                # 1b. Apply preprocessing (CRITICAL: Sync with enrollment pipeline)
                from preprocessing.pipeline import preprocess_frame
                face_crop = preprocess_frame(face_crop)
                
                # 2. Quality check
                quality = self.detector.check_image_quality(face_crop)
                if not quality.get('passed', False):
                    # #region agent log
                    _debug_log("async_recognizer.py:83", "quality_check_failed", {
                        "track_id": track_id,
                        "quality": quality
                    }, hypothesis_id="H4")
                    # #endregion
                    continue
                
                # 3. Generate embeddings (THIS is the 500ms+ operation)
                embeddings = self.recognizer.generate_embeddings(face_crop)
                if not embeddings:
                    # #region agent log
                    _debug_log("async_recognizer.py:88", "embedding_generation_failed", {
                        "track_id": track_id
                    }, hypothesis_id="H1")
                    # #endregion
                    continue
                
                # 4. Find best match
                all_known = self.em_manager.get_all_embeddings()
                match_result = self.recognizer.find_best_match(embeddings, all_known)
                
                # #region agent log
                _debug_log("async_recognizer.py:94", "recognition_complete", {
                    "track_id": track_id,
                    "match_found": match_result.get("match_found", False),
                    "student_id": match_result.get("student_id"),
                    "confidence": float(match_result.get("confidence", 0.0)),
                    "vote_ratio": float(match_result.get("vote_ratio", 0.0))
                }, hypothesis_id="H2")
                # #endregion
                
                # 5. Store result
                with self.results_lock:
                    self.results[track_id] = match_result
                    
            except Exception as e:
                logger.error(f"AsyncRecognizer error for track {track_id}: {e}")
                # #region agent log
                _debug_log("async_recognizer.py:100", "recognition_error", {
                    "track_id": track_id,
                    "error": str(e)
                }, hypothesis_id="H1")
                # #endregion
            finally:
                with self.busy_lock:
                    self.busy = False
