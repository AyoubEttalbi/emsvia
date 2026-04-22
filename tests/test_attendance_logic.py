import pytest
import time
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Student, AttendanceRecord
from database.crud import AttendanceDB
from core.attendance_manager import AttendanceManager

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture
def db_manager():
    manager = AttendanceDB(TEST_DATABASE_URL)
    Base.metadata.create_all(manager.engine)
    return manager

@pytest.fixture
def session(db_manager):
    session = db_manager.get_session()
    yield session
    session.close()

@pytest.fixture
def attendance_manager(db_manager):
    # Use small consistency and cooldown for faster testing
    return AttendanceManager(db_manager, cooldown_seconds=1, min_consistency=2)

def test_evidence_accumulation(attendance_manager, session):
    """Test that attendance is marked only after reaching min_consistency."""
    student = attendance_manager.db.add_student(session, "STU001", "Test", "User")
    student_id = student.id
    track_id = 101
    
    # 1st recognition - should not mark
    marked = attendance_manager.process_recognition(session, student_id, track_id)
    assert marked is False
    assert len(attendance_manager.db.get_attendance_history(session, student_id=student_id)) == 0
    
    # 2nd recognition - should mark (consistency=2)
    marked = attendance_manager.process_recognition(session, student_id, track_id)
    assert marked is True
    assert len(attendance_manager.db.get_attendance_history(session, student_id=student_id)) == 1

def test_cooldown_logic(attendance_manager, session):
    """Test that cooldown prevents double marking within the window."""
    student = attendance_manager.db.add_student(session, "STU002", "Test", "User")
    student_id = student.id
    track_id_1 = 101
    track_id_2 = 102
    
    # Mark once on Track 1
    attendance_manager.process_recognition(session, student_id, track_id_1)
    marked = attendance_manager.process_recognition(session, student_id, track_id_1)
    assert marked is True
    assert len(attendance_manager.db.get_attendance_history(session, student_id=student_id)) == 1
    
    # Try marking again immediately on a new track (Track 2) - should be in cooldown
    # Recognition 1 on Track 2
    attendance_manager.process_recognition(session, student_id, track_id_2)
    # Recognition 2 on Track 2 - threshold reached but cooldown should block
    marked = attendance_manager.process_recognition(session, student_id, track_id_2)
    assert marked is False
    assert len(attendance_manager.db.get_attendance_history(session, student_id=student_id)) == 1
    
    # Wait for cooldown (using 1.5s for 1s cooldown)
    time.sleep(1.2)
    
    # Recognition 3 on Track 2 - now consistency is 3 (>=2) and cooldown is over
    marked = attendance_manager.process_recognition(session, student_id, track_id_2)
    assert marked is True
    assert len(attendance_manager.db.get_attendance_history(session, student_id=student_id)) == 2

def test_manual_override(attendance_manager, session):
    """Test manual attendance marking."""
    student = attendance_manager.db.add_student(session, "STU003", "Test", "User")
    student_id = student.id
    success = attendance_manager.manual_mark_attendance(session, student_id, status="late")
    assert success is True
    
    history = attendance_manager.db.get_attendance_history(session, student_id=student_id)
    assert len(history) == 1
    assert history[0].status == "late"
    assert history[0].camera_id == "manual_override"

def test_departure_logging(attendance_manager, session):
    """Test logging student departure."""
    student = attendance_manager.db.add_student(session, "STU004", "Test", "User")
    student_id = student.id
    attendance_manager.log_departure(session, student_id)
    
    latest = attendance_manager.db.get_latest_attendance(session, student_id)
    assert latest is not None
    assert latest.status == "departed"

def test_record_correction(db_manager, session):
    """Test updating and deleting records."""
    student = db_manager.add_student(session, "STU005", "Test", "User")
    student_id = student.id
    
    # Create a record
    record = db_manager.mark_attendance(session, student_id, 0.9)
    record_id = record.id
    
    # Update it
    success = db_manager.update_attendance_record(session, record_id, {"status": "excused"})
    assert success is True
    
    updated = session.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first()
    assert updated.status == "excused"
    
    # Delete it
    success = db_manager.delete_attendance_record(session, record_id)
    assert success is True
    assert session.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first() is None
