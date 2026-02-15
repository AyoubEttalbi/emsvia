import numpy as np
from deepface import DeepFace
from typing import List, Optional, Dict, Union, Any
import logging
from scipy.spatial import distance
from config.settings import (
    FACE_RECOGNITION_MODEL, 
    RECOGNITION_THRESHOLD,
    USE_RECOGNITION_ENSEMBLE,
    RECOGNITION_MODELS,
    ENSEMBLE_VOTING_THRESHOLD
)

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Handles face embedding generation and matching using DeepFace.
    Supports single model or Ensemble Recognition.
    """
    
    # Default thresholds for models (Cosine Distance)
    THRESHOLDS = {
        "Facenet512": 0.30,
        "ArcFace": 0.68,
        "VGG-Face": 0.40,
        "Facenet": 0.40,
        "OpenFace": 0.10,
        "DeepFace": 0.23,
        "DeepID": 0.01
    }
    
    def __init__(self, model_names: List[str] = None):
        """
        Initialize the Face Recognizer.
        
        Args:
            model_names: List of DeepFace models to use.
        """
        if model_names:
            self.model_names = model_names
        else:
            from config.settings import get_active_recognizers
            self.model_names = get_active_recognizers()
            
        logger.info(f"FaceRecognizer initialized with active models: {self.model_names}")

    def generate_embeddings(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate face embedding vectors for all models.
        """
        results = {}
        if image is None or image.size == 0:
            return results
            
        from models.gpu_manager import GPUModelManager
        gpu_mgr = GPUModelManager()
        
        models_to_run = self.model_names
            
        for model in models_to_run:
            try:
                embedding_objs = DeepFace.represent(
                    img_path=image,
                    model_name=model,
                    enforce_detection=False,
                    detector_backend="skip"
                )
                if embedding_objs:
                    results[model] = np.array(embedding_objs[0]["embedding"])
            except Exception as e:
                logger.error(f"Error generating embedding for {model}: {e}")
        return results

    def generate_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Single embedding generation (backward compatibility).
        """
        embeddings = self.generate_embeddings(image)
        # Return first one or None
        if not embeddings: return None
        return next(iter(embeddings.values()))

    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine") -> float:
        """Calculate distance between two embeddings."""
        if metric == "cosine":
            return distance.cosine(embedding1, embedding2)
        elif metric == "euclidean":
            return distance.euclidean(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_best_match(self, source_embeddings: Dict[str, np.ndarray], 
                        database_embeddings: Dict[int, Dict[str, List[np.ndarray]]]) -> Dict[str, Any]:
        """
        Find the closest match using Ensemble Voting.
        """
        votes = {} # student_id -> score
        model_results = {} # model_name -> best_match_for_that_model

        for model_name, source_vector in source_embeddings.items():
            min_dist = float("inf")
            best_id_for_model = None
            
            threshold = self.THRESHOLDS.get(model_name, 0.40)
            
            for student_id, model_dict in database_embeddings.items():
                if not isinstance(model_dict, dict) or model_name not in model_dict:
                    continue
                    
                for db_vector in model_dict[model_name]:
                    dist = self.calculate_distance(source_vector, db_vector, metric="cosine")
                    if dist < min_dist:
                        min_dist = dist
                        best_id_for_model = student_id
            
            if best_id_for_model is not None and min_dist <= threshold:
                model_results[model_name] = {"id": best_id_for_model, "dist": min_dist}
                votes[best_id_for_model] = votes.get(best_id_for_model, 0) + 1
            else:
                model_results[model_name] = {"id": None, "dist": min_dist}

        if not votes:
            return {"match_found": False, "student_id": None, "confidence": 0.0}

        # Find ID with most votes
        winner_id = max(votes, key=votes.get)
        num_models_run = len(source_embeddings)
        vote_ratio = votes[winner_id] / num_models_run if num_models_run > 0 else 0
        
        # Adaptive Baseline: Ensemble vote
        match_found = vote_ratio >= ENSEMBLE_VOTING_THRESHOLD
        
        # Override: If any model is SUPER confident, trust it!
        # This helps if 2 models fail but 1 is 100% sure.
        for model_name, res in model_results.items():
            threshold = self.THRESHOLDS.get(model_name, 0.40)
            if res["id"] is not None and res["dist"] <= (threshold * 0.7):
                match_found = True
                winner_id = res["id"]
                logger.info(f"Strong match override for {model_name}: {winner_id} (dist: {res['dist']:.3f})")
                break

        return {
            "match_found": match_found,
            "student_id": winner_id if match_found else None,
            "vote_ratio": vote_ratio,
            "model_results": model_results
        }
