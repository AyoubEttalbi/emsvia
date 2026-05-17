import numpy as np
from deepface import DeepFace
from typing import List, Optional, Dict, Union, Any
import logging
from scipy.spatial import distance
import json
import time
from config.settings import (
    FACE_RECOGNITION_MODEL,
    RECOGNITION_THRESHOLD,
    USE_RECOGNITION_ENSEMBLE,
    RECOGNITION_MODELS,
    ENSEMBLE_VOTING_THRESHOLD,
    ARCFACE_THRESHOLD,
    FACENET512_THRESHOLD,
    VGGFACE_THRESHOLD,
    RECOGNITION_CONFIDENCE_FLOOR,
    MIN_VECTOR_PASS_RATIO,
    MIN_CONFIDENCE_GAP,
)

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

class FaceRecognizer:
    """
    Handles face embedding generation and matching using DeepFace.
    Supports single model or Ensemble Recognition.
    """
    
    # Default thresholds for models (Cosine Distance)
    # These are overridden at runtime by per-model env vars from settings.
    # Kept here as fallbacks and for models not exposed via env.
    THRESHOLDS = {
        "Facenet512": FACENET512_THRESHOLD,
        "ArcFace": ARCFACE_THRESHOLD,
        "VGG-Face": VGGFACE_THRESHOLD,
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
        # #region agent log
        _debug_log("face_recognizer.py:50", "generate_embeddings_start", {
            "image_shape": image.shape if image is not None else None,
            "image_size": image.size if image is not None else 0,
            "image_dtype": str(image.dtype) if image is not None else None,
            "image_mean": float(np.mean(image)) if image is not None else None,
            "image_std": float(np.std(image)) if image is not None else None,
            "image_min": float(np.min(image)) if image is not None else None,
            "image_max": float(np.max(image)) if image is not None else None,
            "models": self.model_names
        }, hypothesis_id="H1")
        # #endregion
        
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
                    embedding = np.array(embedding_objs[0]["embedding"])
                    
                    # CRITICAL FIX: Explicit L2 Normalization
                    # Ensures consistent scale for cosine distance even if pixel stats vary
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    results[model] = embedding
                    
                    # #region agent log
                    _debug_log("face_recognizer.py:72", "embedding_generated", {
                        "model": model,
                        "embedding_shape": embedding.shape,
                        "embedding_norm": float(np.linalg.norm(embedding)),
                        "embedding_mean": float(np.mean(embedding)),
                        "embedding_std": float(np.std(embedding)),
                        "embedding_min": float(np.min(embedding)),
                        "embedding_max": float(np.max(embedding))
                    }, hypothesis_id="H3")
                    # #endregion
            except Exception as e:
                logger.error(f"Error generating embedding for {model}: {e}")
                # #region agent log
                _debug_log("face_recognizer.py:74", "embedding_error", {
                    "model": model,
                    "error": str(e)
                }, hypothesis_id="H1")
                # #endregion
        
        # #region agent log
        _debug_log("face_recognizer.py:75", "generate_embeddings_end", {
            "results_count": len(results),
            "models_success": list(results.keys())
        }, hypothesis_id="H1")
        # #endregion
        
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
        Find the closest match using Consensus Voting.
        
        Consensus Rule: Evaluate all stored embeddings for a student and require that >20% 
        of the templates tie matches within the threshold. Matches with the highest pass
        vectors will be selected.
        """
        votes = {}  # student_id -> count
        model_results = {}  # model_name -> {id, dist, second_best_dist}
        all_best_distances = []  # Track all best distances for rejection
        
        # For each model, find best match taking into account consensus
        for model_name, source_vector in source_embeddings.items():
            threshold = self.THRESHOLDS.get(model_name, 0.40)
            
            # #region agent log
            _debug_log("face_recognizer.py:110", "model_matching_start", {
                "model": model_name,
                "threshold": threshold,
                "source_embedding_norm": float(np.linalg.norm(source_vector)),
                "database_students_count": len(database_embeddings)
            }, hypothesis_id="H2")
            # #endregion
            
            student_metrics = {} # student_id -> {"passing_vectors": int, "match_ratio": float, "min_dist": float}
            all_student_min_dists = {} # student_id -> closest distance (used for second-best gap check)

            for student_id, model_dict in database_embeddings.items():
                if not isinstance(model_dict, dict) or model_name not in model_dict:
                    continue
                
                passing_vectors = 0
                total_vectors = len(model_dict[model_name])
                student_min_dist = float("inf")
                
                # Find minimum distance across ALL embeddings for this student
                for db_vector in model_dict[model_name]:
                    dist = self.calculate_distance(source_vector, db_vector, metric="cosine")
                    
                    if dist <= threshold:
                        passing_vectors += 1
                        
                    if dist < student_min_dist:
                        student_min_dist = dist
                            
                # CONSENSUS RULE:
                # Require >= MIN_VECTOR_PASS_RATIO of total vectors to match to confirm identity.
                # This filters out "bad apple" embeddings that match everyone.
                
                match_ratio = passing_vectors / total_vectors if total_vectors > 0 else 0
                
                # Minimum requirement: 2 matches OR MIN_VECTOR_PASS_RATIO of total
                required_matches = max(2, int(total_vectors * MIN_VECTOR_PASS_RATIO))
                
                if passing_vectors >= required_matches:
                    student_metrics[student_id] = {
                        "passing_vectors": passing_vectors,
                        "match_ratio": match_ratio,
                        "min_dist": student_min_dist
                    }
                    # #region agent log
                    _debug_log("face_recognizer.py:146", "candidate_consensus_passed", {
                        "model": model_name,
                        "student_id": student_id,
                        "passing_vectors": passing_vectors,
                        "match_ratio": match_ratio,
                        "total_vectors": total_vectors,
                        "min_dist": float(student_min_dist) if student_min_dist < float("inf") else None
                    }, hypothesis_id="H2")
                    # #endregion
                else:
                    if passing_vectors > 0:
                        # #region agent log
                        _debug_log("face_recognizer.py:155", "candidate_consensus_failed", {
                            "model": model_name,
                            "student_id": student_id,
                            "passing_vectors": passing_vectors,
                            "total_vectors": total_vectors,
                            "required_matches": required_matches
                        }, hypothesis_id="H2")
                        # #endregion
                        pass

                if student_min_dist < float("inf"):
                    all_best_distances.append(student_min_dist)
                    all_student_min_dists[student_id] = student_min_dist

            # Decide winner based on strongest consensus ratio, tie-break by distance
            if student_metrics:
                sorted_metrics = sorted(
                    student_metrics.items(),
                    key=lambda x: (-x[1]["match_ratio"], x[1]["min_dist"])
                )
                best_id_for_model = sorted_metrics[0][0]
                best_min_dist = sorted_metrics[0][1]["min_dist"]

                # Second-best gap check: find the closest non-winner across ALL students
                # (including those that failed consensus — they may still be dangerously close)
                second_best_dist = float("inf")
                for sid, dist in sorted(all_student_min_dists.items(), key=lambda x: x[1]):
                    if sid != best_id_for_model:
                        second_best_dist = dist
                        break

                gap_ratio = (second_best_dist - best_min_dist) / threshold if second_best_dist < float("inf") else float("inf")

                if second_best_dist < float("inf") and gap_ratio < MIN_CONFIDENCE_GAP:
                    # Ambiguous: winner and runner-up are too close in embedding space
                    model_results[model_name] = {"id": None, "dist": best_min_dist, "second_best_dist": second_best_dist}
                    # #region agent log
                    _debug_log("face_recognizer.py:177", "match_rejected_ambiguous", {
                        "model": model_name,
                        "best_id": best_id_for_model,
                        "best_dist": float(best_min_dist),
                        "second_best_dist": float(second_best_dist),
                        "gap_ratio": float(gap_ratio),
                        "min_confidence_gap": MIN_CONFIDENCE_GAP
                    }, hypothesis_id="H2")
                    # #endregion
                else:
                    model_results[model_name] = {"id": best_id_for_model, "dist": best_min_dist, "second_best_dist": second_best_dist}
                    votes[best_id_for_model] = votes.get(best_id_for_model, 0) + 1
                    # #region agent log
                    _debug_log("face_recognizer.py:177", "match_accepted", {
                        "model": model_name,
                        "best_id": best_id_for_model,
                        "best_dist": float(best_min_dist),
                        "second_best_dist": float(second_best_dist) if second_best_dist < float("inf") else None,
                        "gap_ratio": float(gap_ratio) if gap_ratio < float("inf") else None
                    }, hypothesis_id="H2")
                    # #endregion
            else:
                model_results[model_name] = {"id": None, "dist": float("inf"), "second_best_dist": float("inf")}
                # #region agent log
                _debug_log("face_recognizer.py:144", "rejected_threshold", {
                    "model": model_name,
                    "reason": "failed_consensus_rule_all_students"
                }, hypothesis_id="H2")
                # #endregion
                     
        # If no models found a valid match, reject
        if not votes:
            best_overall_dist = min(all_best_distances) if all_best_distances else float("inf")
            logger.debug(f"No valid matches. Best distance: {best_overall_dist:.3f}")
            # #region agent log
            _debug_log("face_recognizer.py:181", "no_votes_final", {
                "best_overall_dist": float(best_overall_dist) if best_overall_dist < float("inf") else None,
                "model_results": {k: {"id": v.get("id"), "dist": float(v.get("dist")) if v.get("dist") < float("inf") else None} for k, v in model_results.items()}
            }, hypothesis_id="H2")
            # #endregion
            return {
                "match_found": False,
                "student_id": None,
                "confidence": 0.0,
                "best_distance": best_overall_dist,
                "model_results": model_results
            }
        
        # Find winner (student with most votes)
        winner_id = max(votes, key=votes.get)
        num_models = len(source_embeddings)
        vote_ratio = votes[winner_id] / num_models
        
        # Calculate average distance for winner
        winner_distances = [r["dist"] for r in model_results.values() if r.get("id") == winner_id]
        avg_distance = np.mean(winner_distances) if winner_distances else float("inf")
        
        # Final check: Require majority agreement (50% of models)
        match_found = vote_ratio >= ENSEMBLE_VOTING_THRESHOLD
        
        # Calculate confidence (1.0 = perfect match, 0.0 = poor match)
        if match_found:
            # Confidence based on distance quality
            avg_threshold = np.mean([self.THRESHOLDS.get(m, 0.40) for m in source_embeddings.keys()])
            # Rescale: distances well below threshold should score high confidence.
            # Without this, distance=0.275 / threshold=0.30 → confidence=0.083 (too harsh).
            confidence = max(0.0, min(1.0, 1.0 - (avg_distance / avg_threshold)) * 2.0)
        else:
            confidence = 0.0
            logger.info(f"❌ REJECTED: Vote ratio {vote_ratio:.2f} < threshold {ENSEMBLE_VOTING_THRESHOLD:.2f}")

        # Confidence floor: reject even winning matches below threshold
        if match_found and confidence < RECOGNITION_CONFIDENCE_FLOOR:
            match_found = False
            confidence = 0.0
            logger.info(f"❌ REJECTED: Confidence {confidence:.3f} (dist={avg_distance:.3f}) below floor {RECOGNITION_CONFIDENCE_FLOOR:.2f}")
        
        # #region agent log
        _debug_log("face_recognizer.py:213", "final_result", {
            "match_found": match_found,
            "winner_id": winner_id if match_found else None,
            "vote_ratio": float(vote_ratio),
            "votes": dict(votes),
            "avg_distance": float(avg_distance) if avg_distance < float("inf") else None,
            "confidence": float(confidence),
            "ensemble_threshold": ENSEMBLE_VOTING_THRESHOLD,
            "rejection_reason": "vote_ratio_too_low" if not match_found else None
        }, hypothesis_id="H2")
        # #endregion
        
        return {
            "match_found": match_found,
            "student_id": winner_id if match_found else None,
            "vote_ratio": vote_ratio,
            "best_distance": avg_distance,
            "avg_distance": avg_distance,
            "confidence": confidence,
            "model_results": model_results
        }
