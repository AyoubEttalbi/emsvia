from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from api.dependencies import get_db, get_db_manager
from api import schemas

router = APIRouter()

@router.get("/recent", response_model=List[schemas.AttendanceResponse])
def get_recent_attendance(limit: int = 10, db: Session = Depends(get_db)):
    """Retrieve the most recent attendance logs."""
    from database.models import AttendanceRecord
    return db.query(AttendanceRecord).order_by(AttendanceRecord.timestamp.desc()).limit(limit).all()

@router.get("/records", response_model=List[schemas.AttendanceResponse])

def get_attendance_records(
    student_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Query attendance history with filters."""
    db_manager = get_db_manager()
    return db_manager.get_attendance_history(db, student_id, start_date, end_date)

@router.post("/mark", response_model=schemas.AttendanceResponse, status_code=status.HTTP_201_CREATED)
def mark_attendance_manual(
    attendance_in: schemas.AttendanceCreate, 
    db: Session = Depends(get_db)
):
    """Manually mark attendance (Override)."""
    db_manager = get_db_manager()
    from core.attendance_manager import AttendanceManager
    
    # We use a temporary AttendanceManager instance for the manual marking
    # since it already wraps the logic correctly
    manager = AttendanceManager(db_manager)
    
    success = manager.manual_mark_attendance(
        db, 
        student_id=attendance_in.student_id,
        status=attendance_in.status,
        confidence=attendance_in.confidence_score,
        camera_id=attendance_in.camera_id
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to mark attendance")
    
    # Return the latest record we just created
    return db_manager.get_latest_attendance(db, attendance_in.student_id)

@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_attendance(record_id: int, db: Session = Depends(get_db)):
    """Delete an attendance record."""
    db_manager = get_db_manager()
    success = db_manager.delete_attendance_record(db, record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return None

@router.patch("/{record_id}", response_model=bool)
def update_attendance(
    record_id: int, 
    updates: schemas.AttendanceUpdate, 
    db: Session = Depends(get_db)
):
    """Update an attendance record (correction)."""
    db_manager = get_db_manager()
    success = db_manager.update_attendance_record(db, record_id, updates.model_dump(exclude_unset=True))
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return True
