"""
Batch Face Recognition for GPU Acceleration.
Process multiple faces simultaneously to maximize GPU utilization.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from scipy.spatial import distance

from config.settings import BATCH_SIZE, RECOGNITION_MODELS
from models.face_recognizer import FaceRecognizer

logger = logging.getLogger(__name__)

class BatchRecognizer:
    """
    Process multiple faces simultaneously on GPU for efficient recognition.
    Uses ensemble voting across multiple recognition models.
    """
    
    def __init__(self, recognizer: FaceRecognizer = None, batch_size: int = None):
        """
        Initialize BatchRecognizer.
        
        Args:
            recognizer: FaceRecognizer instance to use
            batch_size: Maximum batch size for processing
        """
        self.recognizer = recognizer or FaceRecognizer()
        self.batch_size = batch_size or BATCH_SIZE
        logger.info(f"BatchRecognizer initialized with batch_size={self.batch_size}")
    
    def recognize_batch(self, face_crops: List[np.ndarray], 
                       database_embeddings: Dict[int, Dict[str, List[np.ndarray]]]) -> List[Dict[str, Any]]:
        """
        Process multiple faces simultaneously and recognize them.
        
        Args:
            face_crops: List of face images (numpy arrays)
            database_embeddings: Database of known embeddings
            
        Returns:
            List of recognition results for each face
        """
        if len(face_crops) == 0:
            return []
        
        results = []
        
        # Process in batches to avoid GPU memory issues
        for i in range(0, len(face_crops), self.batch_size):
            batch = face_crops[i:i + self.batch_size]
            batch_results = self._process_batch(batch, database_embeddings)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[np.ndarray], 
                      database_embeddings: Dict[int, Dict[str, List[np.ndarray]]]) -> List[Dict[str, Any]]:
        """Process a single batch of faces."""
        batch_embeddings = []
        
        # Generate embeddings for all faces in batch
        for face_crop in batch:
            embeddings = self.recognizer.generate_embeddings(face_crop)
            batch_embeddings.append(embeddings)
        
        # Match each face against database
        results = []
        for embeddings in batch_embeddings:
            if embeddings:
                match_result = self.recognizer.find_best_match(embeddings, database_embeddings)
                results.append(match_result)
            else:
                results.append({"match_found": False, "student_id": None, "confidence": 0.0})
        
        return results
    
    def generate_embeddings_batch(self, face_crops: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Generate embeddings for multiple faces.
        
        Args:
            face_crops: List of face images
            
        Returns:
            List of embeddings dictionaries (one per face)
        """
        results = []
        
        # Process in batches
        for i in range(0, len(face_crops), self.batch_size):
            batch = face_crops[i:i + self.batch_size]
            
            for face_crop in batch:
                embeddings = self.recognizer.generate_embeddings(face_crop)
                results.append(embeddings)
        
        return results
