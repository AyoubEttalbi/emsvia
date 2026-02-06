import numpy as np
from deepface import DeepFace
from typing import List, Optional, Dict, Union
import logging
from scipy.spatial import distance

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Handles face embedding generation and matching using DeepFace.
    """
    
    # Thresholds for Facenet512 (Cosine Distance)
    # 0.30 is generally recommended for Facenet512 cosine similarity
    THRESHOLD = 0.30 
    
    def __init__(self, model_name: str = "Facenet512"):
        """
        Initialize the Face Recognizer.
        
        Args:
            model_name: DeepFace model to use (default: Facenet512)
        """
        self.model_name = model_name
        logger.info(f"FaceRecognizer initialized with model: {model_name}")

    def generate_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a face embedding vector for a given aligned face image.
        
        Args:
            image: Aligned face image (BGR)
            
        Returns:
            Numpy array representing the embedding (512-dim for Facenet512) or None if failure.
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to generate_embedding")
            return None
            
        try:
            # enforce_detection=False because we assume we are passing an aligned crop
            embedding_objs = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip" # We already detected and aligned
            )
            
            if not embedding_objs:
                return None
                
            # DeepFace returns a list of dicts. We expect one since we disabled detection.
            embedding = embedding_objs[0]["embedding"]
            return np.array(embedding)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine") -> float:
        """
        Calculate distance between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Distance metric ('cosine', 'euclidean')
            
        Returns:
            Distance value (float). Lower is closer.
        """
        if metric == "cosine":
            return distance.cosine(embedding1, embedding2)
        elif metric == "euclidean":
            return distance.euclidean(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_best_match(self, source_embedding: np.ndarray, 
                        database_embeddings: Dict[int, List[np.ndarray]]) -> Dict[str, Union[int, float, bool]]:
        """
        Find the closest match for a source embedding in a dictionary of database embeddings.
        
        Args:
            source_embedding: The embedding to search for.
            database_embeddings: Dict mapping student_id -> list of embedding vectors.
            
        Returns:
            Dictionary with match results:
            {
                'match_found': bool,
                'student_id': int or None,
                'min_distance': float
            }
        """
        min_dist = float("inf")
        best_match_id = None
        
        for student_id, vectors in database_embeddings.items():
            for db_vector in vectors:
                dist = self.calculate_distance(source_embedding, db_vector, metric="cosine")
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = student_id
        
        # Verify against threshold
        if min_dist <= self.THRESHOLD:
            return {
                "match_found": True,
                "student_id": best_match_id,
                "distance": min_dist
            }
        else:
            return {
                "match_found": False,
                "student_id": None,
                "distance": min_dist
            }
