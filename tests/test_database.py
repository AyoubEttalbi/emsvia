import pytest
import os
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Add project root to sys.path if not already there
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from database.models import Base, Student, FaceEmbedding, AttendanceRecord, UnknownFace
from database.crud import AttendanceDB

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture
def db_manager():
    """Fixture to create a fresh in-memory database manager."""
    manager = AttendanceDB(TEST_DATABASE_URL)
    Base.metadata.create_all(manager.engine)
    return manager

@pytest.fixture
def session(db_manager):
    """Fixture to provide a database session."""
    session = db_manager.get_session()
    yield session
    session.close()

def test_add_and_get_student(db_manager, session):
    """Test creating and retrieving a student."""
    student_id = "TEST001"
    first_name = "Test"
    last_name = "User"
    email = "test@example.com"
    
    # Add student
    student = db_manager.add_student(session, student_id, first_name, last_name, email)
    assert student is not None
    assert student.student_id == student_id
    
    # Get student
    retrieved = db_manager.get_student_by_id(session, student_id)
    assert retrieved is not None
    assert retrieved.first_name == first_name
    assert retrieved.last_name == last_name

def test_add_face_embedding(db_manager, session):
    """Test adding a face embedding for a student."""
    # Add student first
    student = db_manager.add_student(session, "STU123", "Face", "Test")
    
    # Create random embedding
    embedding_vector = np.random.rand(512)
    
    # Add embedding
    success = db_manager.add_face_embedding(session, student.id, embedding_vector)
    assert success is True
    
    # Retrieve embeddings
    embeddings = db_manager.get_student_embeddings(session, student.id)
    assert len(embeddings) == 1
    np.testing.assert_array_almost_equal(embeddings[0]["vector"], embedding_vector)

def test_mark_attendance(db_manager, session):
    """Test marking attendance records."""
    student = db_manager.add_student(session, "ATT001", "Attendance", "User")
    
    record = db_manager.mark_attendance(session, student.id, 0.95, camera_id="test_cam")
    assert record is not None
    assert record.student_id == student.id
    assert record.confidence_score == 0.95
    
    # Check history
    history = db_manager.get_attendance_history(session, student_id=student.id)
    assert len(history) == 1
    assert history[0].camera_id == "test_cam"

def test_unknown_face_logging(db_manager, session):
    """Test logging and reviewing unknown faces."""
    img_path = "path/to/unknown.jpg"
    unknown = db_manager.log_unknown_face(session, img_path, confidence=0.4)
    assert unknown is not None
    assert unknown.image_path == img_path
    
    # Check unreviewed
    unreviewed = db_manager.get_unreviewed_faces(session)
    assert len(unreviewed) == 1
    
    # Mark reviewed
    success = db_manager.mark_unknown_face_reviewed(session, unknown.id)
    assert success is True
    
    unreviewed_after = db_manager.get_unreviewed_faces(session)
    assert len(unreviewed_after) == 0
