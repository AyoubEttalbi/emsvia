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
        elif USE_RECOGNITION_ENSEMBLE:
            self.model_names = RECOGNITION_MODELS
        else:
            self.model_names = [FACE_RECOGNITION_MODEL]
            
        logger.info(f"FaceRecognizer initialized with models: {self.model_names}")

    def generate_embeddings(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate face embedding vectors for all models.
        """
        results = {}
        if image is None or image.size == 0:
            return results
            
        for model in self.model_names:
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
            
            # Use the configurable threshold from settings instead of hardcoded values
            threshold = RECOGNITION_THRESHOLD
            
            # Consensus Matching Logic
            # Instead of taking the single best match, we count how many embeddings for this person
            # are within the threshold. This filters out "bad apples".
            
            potential_matches = {} # student_id -> count of passing vectors
            
            for student_id, model_dict in database_embeddings.items():
                if model_name not in model_dict:
                    continue
                
                passing_vectors = 0
                total_vectors = len(model_dict[model_name])
                
                for db_vector in model_dict[model_name]:
                    dist = self.calculate_distance(source_vector, db_vector, metric="cosine")
                    
                    if dist <= threshold:
                        passing_vectors += 1
                        # Track minimum distance for tie-breaking
                        if dist < min_dist:
                            min_dist = dist
                            best_id_for_model = student_id
                
                # CONSENSUS RULE:
                # Require > 20% of total vectors to match to confirm identity.
                # This filters out "bad apple" embeddings that match everyone.
                
                match_ratio = passing_vectors / total_vectors if total_vectors > 0 else 0
                
                # Minimum requirement: 2 matches OR 20% of total
                required_matches = max(2, int(total_vectors * 0.20))
                
                if passing_vectors >= required_matches:
                    potential_matches[student_id] = passing_vectors
                    # logger.info(f"Candidate {student_id} confirmed: {passing_vectors}")
                else:
                    if passing_vectors > 0:
                        pass # logger.debug(f"Rejecting {student_id}: {passing_vectors}/{total_vectors} (< 20%)")

            # After checking all students, if we have a best_id, ensure it passed consensus
            if best_id_for_model in potential_matches:
                model_results[model_name] = {"id": best_id_for_model, "dist": min_dist}
                votes[best_id_for_model] = votes.get(best_id_for_model, 0) + 1
            else:
                model_results[model_name] = {"id": None, "dist": min_dist}

        if not votes:
            return {"match_found": False, "student_id": None, "confidence": 0.0}

        # Find ID with most votes
        winner_id = max(votes, key=votes.get)
        vote_ratio = votes[winner_id] / len(self.model_names)
        
        match_found = vote_ratio >= ENSEMBLE_VOTING_THRESHOLD

        return {
            "match_found": match_found,
            "student_id": winner_id if match_found else None,
            "vote_ratio": vote_ratio,
            "model_results": model_results
        }
