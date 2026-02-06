import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from sqlalchemy.orm import Session
from database.crud import AttendanceDB

logger = logging.getLogger(__name__)

class AttendanceManager:
    """
    Manages high-level attendance logic, including cooldowns and verification.
    """
    
    def __init__(self, db_manager: AttendanceDB, 
                 cooldown_minutes: int = 30,
                 min_consecutive_frames: int = 3):
        """
        Initialize the Attendance Manager.
        
        Args:
            db_manager: Instance of AttendanceDB.
            cooldown_minutes: Time to wait before re-marking same person.
            min_consecutive_frames: Frames a person must be seen before marking.
        """
        self.db = db_manager
        self.cooldown_minutes = cooldown_minutes
        self.min_consecutive_frames = min_consecutive_frames
        
        # Tracking state
        self.consecutive_counts: Dict[int, int] = {} # student_id -> count
        self.last_marked: Dict[int, float] = {}      # student_id -> timestamp (unix)
        
    def process_recognition(self, session: Session, student_id: int) -> bool:
        """
        Process a successful recognition event.
        Returns True if attendance was actually marked in the DB.
        """
        # 1. Update consecutive counts
        self.consecutive_counts[student_id] = self.consecutive_counts.get(student_id, 0) + 1
        
        # 2. Check if we reached the required frame threshold
        if self.consecutive_counts[student_id] >= self.min_consecutive_frames:
            
            # 3. Cooldown check
            now = time.time()
            last_time = self.last_marked.get(student_id, 0)
            
            if (now - last_time) / 60.0 >= self.cooldown_minutes:
                # 4. Mark in Database
                success = self.db.mark_attendance(session, student_id, status="Present")
                
                if success:
                    self.last_marked[student_id] = now
                    logger.info(f"Attendance marked for student {student_id}")
                    # Reset count after marking
                    self.consecutive_counts[student_id] = 0
                    return True
                
            else:
                # Still in cooldown, just reset count to avoid "spamming" the logic
                self.consecutive_counts[student_id] = 0
                
        return False

    def reset_counts_except(self, active_student_ids: Set[int]):
        """
        Reset consecutive counts for students not currently in frame.
        This prevents 'ghost' counts from accumulating over time.
        """
        students_to_clear = set(self.consecutive_counts.keys()) - active_student_ids
        for sid in students_to_clear:
            self.consecutive_counts[sid] = 0
            
    def get_session_status(self) -> Dict[str, int]:
        """Return statistics for the current running session."""
        return {
            "marked_this_session": len(self.last_marked),
            "tracking_active": len([c for c in self.consecutive_counts.values() if c > 0])
        }
