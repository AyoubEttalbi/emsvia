"""
Parallel Detection for GPU Acceleration.
Run multiple detectors simultaneously to maximize GPU utilization.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

from config.settings import ENSEMBLE_DETECTORS, NMS_IOU_THRESHOLD
from models.face_detector import FaceDetector
from detection.ensemble_detection import fuse_detections
from detection.tiling import apply_nms

logger = logging.getLogger(__name__)

class ParallelDetector:
    """
    Run multiple face detectors in parallel on GPU.
    Merges detections using NMS and ensemble fusion.
    """
    
    def __init__(self, detector: FaceDetector = None):
        """
        Initialize ParallelDetector.
        
        Args:
            detector: FaceDetector instance
        """
        self.detector = detector or FaceDetector()
        self.ensemble_backends = ENSEMBLE_DETECTORS
        logger.info(f"ParallelDetector initialized with backends: {self.ensemble_backends}")
    
    def detect_ensemble(self, frame: np.ndarray, use_tiling: bool = False) -> List[Dict[str, Any]]:
        """
        Run all detectors in parallel and merge results.
        
        Args:
            frame: Input frame
            use_tiling: Whether to use tiling detection
            
        Returns:
            List of merged detections
        """
        all_detections = []
        
        # Run all configured detectors
        for backend in self.ensemble_backends:
            try:
                detections = self.detector.detect_only_model(frame, backend)
                all_detections.extend(detections)
            except Exception as e:
                logger.error(f"Error in detector {backend}: {e}")
        
        # Merge and deduplicate detections
        if len(self.ensemble_backends) > 1:
            final_detections = fuse_detections(all_detections, iou_threshold=NMS_IOU_THRESHOLD)
        else:
            final_detections = apply_nms(all_detections, iou_threshold=NMS_IOU_THRESHOLD)
        
        return final_detections
    
    def detect_parallel(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect faces in multiple frames (future enhancement for async processing).
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection lists (one per frame)
        """
        results = []
        for frame in frames:
            detections = self.detect_ensemble(frame)
            results.append(detections)
        return results
