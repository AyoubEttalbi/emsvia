import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from sqlalchemy.orm import Session
from database.crud import AttendanceDB
from config.settings import ATTENDANCE_MIN_CONSISTENCY, ATTENDANCE_COOLDOWN

logger = logging.getLogger(__name__)

class AttendanceManager:
    """
    Manages high-level attendance logic, including cooldowns and verification.
    Now supports track-based evidence accumulation.
    """
    
    def __init__(self, db_manager: AttendanceDB, 
                 cooldown_seconds: int = ATTENDANCE_COOLDOWN,
                 min_consistency: int = ATTENDANCE_MIN_CONSISTENCY):
        self.db = db_manager
        self.cooldown_seconds = cooldown_seconds
        self.min_consistency = min_consistency
        
        # Tracking state
        self.track_evidence: Dict[int, Dict[int, int]] = {} # track_id -> {student_id: count}
        self.last_marked: Dict[int, float] = {}             # student_id -> timestamp (unix)
        
    def process_recognition(self, session: Session, student_id: int, 
                             track_id: int, confidence: float = 0.0) -> bool:
        """
        Process a recognition event using temporal evidence.
        """
        if track_id not in self.track_evidence:
            self.track_evidence[track_id] = {}
        
        # Accumulate evidence for this track
        self.track_evidence[track_id][student_id] = self.track_evidence[track_id].get(student_id, 0) + 1
        
        # Check if we have enough consistent evidence for THIS student on THIS track
        if self.track_evidence[track_id][student_id] >= self.min_consistency:
            
            # Cooldown check for the student
            now = time.time()
            last_time = self.last_marked.get(student_id, 0)
            
            if (now - last_time) >= self.cooldown_seconds:
                # Mark in Database
                success = self.db.mark_attendance(session, student_id, confidence=confidence)
                
                if success:
                    self.last_marked[student_id] = now
                    logger.info(f"Attendance marked for student {student_id} (Track: {track_id})")
                    # Clear evidence for THIS track to prevent re-marking
                    del self.track_evidence[track_id]
                    return True
            else:
                # In cooldown, don't clear track evidence but don't mark
                # (The student might stay in frame, we just wait for cooldown)
                pass
                
        return False

    def clean_stale_tracks(self, active_track_ids: Set[int]):
        """Remove evidence for tracks that are no longer visible."""
        stale_tracks = set(self.track_evidence.keys()) - active_track_ids
        for tid in stale_tracks:
            del self.track_evidence[tid]

    def reset_counts_except(self, active_student_ids: Set[int]):
        """Compatibility method for main.py."""
        pass # Now handled by track evidence and cooldowns

    def get_session_status(self) -> Dict[str, int]:
        return {
            "marked_this_session": len(self.last_marked),
            "tracking_active": len(self.track_evidence)
        }
