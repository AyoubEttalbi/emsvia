import cv2
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session
from database.crud import AttendanceDB
import numpy as np

logger = logging.getLogger(__name__)

class UnknownFaceHandler:
    """
    Handles logging and image saving for unrecognized faces.
    """
    
    def __init__(self, db_manager: AttendanceDB, 
                 storage_dir: str = "data/unknown_faces",
                 cooldown_seconds: int = 60):
        """
        Initialize the Unknown Face Handler.
        
        Args:
            db_manager: AttendanceDB instance.
            storage_dir: Directory to save unknown face images.
            cooldown_seconds: Minimum time between logging same 'unknown' event.
        """
        self.db = db_manager
        self.storage_dir = Path(storage_dir)
        self.cooldown_seconds = cooldown_seconds
        
        # Ensure storage directory exists
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)
            
        self.last_log_time = 0
        
    def handle_unknown_face(self, session: Session, face_image: np.ndarray, confidence: Optional[float] = None) -> bool:
        """
        Processes an unknown face: saves image and logs to DB.
        """
        now = time.time()
        
        # Cooldown check to prevent flooding with images of the same unknown person
        if now - self.last_log_time < self.cooldown_seconds:
            return False
            
        try:
            # 1. Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}.jpg"
            save_path = self.storage_dir / filename
            
            # 2. Save image
            cv2.imwrite(str(save_path), face_image)
            
            # 3. Log to DB
            # We store the relative path for portability
            relative_path = f"data/unknown_faces/{filename}"
            self.db.log_unknown_face(session, relative_path, confidence)
            
            self.last_log_time = now
            logger.info(f"Unknown face logged: {relative_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling unknown face: {e}")
            return False

    def get_stats(self, session: Session) -> int:
        """Get total number of unreviewed unknown faces."""
        return len(self.db.get_unreviewed_faces(session))
