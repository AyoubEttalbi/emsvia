import pickle
import logging
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from database.crud import AttendanceDB

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """
    Manages loading, saving, and caching face embeddings.
    Supports both database and local pickle file for faster startup.
    """
    
    def __init__(self, db_manager: AttendanceDB, cache_file: str = "data/embeddings/cache.pkl"):
        """
        Initialize the Embeddings Manager.
        
        Args:
            db_manager: Database manager instance
            cache_file: Path to the local pickle cache file
        """
        self.db = db_manager
        self.cache_file = Path(cache_file)
        self.embedding_cache: Dict[int, Dict[str, List[np.ndarray]]] = {}
        self.cache_loaded = False
        
        # Ensure directory exists
        if not self.cache_file.parent.exists():
            self.cache_file.parent.mkdir(parents=True)

    def load_embeddings(self, session: Session, force_refresh: bool = False):
        """
        Load embeddings into memory.
        Tries to load from pickle first, then falls back to DB if missing or forced.
        
        Args:
            session: Database session
            force_refresh: Force loading from DB and updating cache
        """
        if not force_refresh and self.cache_file.exists():
            try:
                self._load_from_pickle()
                logger.info("Embeddings loaded from cache.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Falling back to database.")
        
        self._load_from_db(session)
        self._save_to_pickle()
        logger.info("Embeddings loaded from database and cached.")

    def _load_from_pickle(self):
        """Load embeddings from pickle file."""
        with open(self.cache_file, "rb") as f:
            self.embedding_cache = pickle.load(f)
        self.cache_loaded = True

    def _save_to_pickle(self):
        """Save current cache to pickle file."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")

    def _load_from_db(self, session: Session):
        """Load all embeddings from the database."""
        self.embedding_cache = self.db.get_all_embeddings_for_recognition(session)
        self.cache_loaded = True

    def get_all_embeddings(self) -> Dict[int, Dict[str, List[np.ndarray]]]:
        """Return the dictionary of all embeddings (student_id -> {model_name: [vectors]})."""
        return self.embedding_cache

    def add_embedding(self, session: Session, student_id: int, vector: np.ndarray, model_name: str = "Facenet512"):
        """
        Add a new embedding to both DB and Cache.
        """
        # Add to DB
        success = self.db.add_face_embedding(session, student_id, vector, model_name=model_name)
        if success:
            # Update Cache
            if student_id not in self.embedding_cache:
                self.embedding_cache[student_id] = {}
            if model_name not in self.embedding_cache[student_id]:
                self.embedding_cache[student_id][model_name] = []
                
            self.embedding_cache[student_id][model_name].append(vector)
            
            # Persist cache update
            self._save_to_pickle()
        return success
