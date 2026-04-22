import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import models to ensure they are registered with Base metadata
from database.models import Base, Student, AttendanceRecord, UnknownFace
from api.main import app
from api.dependencies import get_db

# Setup Test DB (using a physical file for better cross-thread/process reliability in tests)
TEST_DB_FILE = "test_api.db"
TEST_DATABASE_URL = f"sqlite:///./{TEST_DB_FILE}"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    # Remove existing test db if any
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_create_and_get_student():
    # Create
    student_data = {
        "student_id": "API_TEST_001",
        "first_name": "API",
        "last_name": "Tester",
        "email": "api@test.com"
    }
    response = client.post("/api/students/", json=student_data)
    assert response.status_code == 201
    
    # Get
    response = client.get(f"/api/students/API_TEST_001")
    assert response.status_code == 200
    assert response.json()["first_name"] == "API"

def test_attendance_manual_mark():
    # Get student internal ID (create new for this test)
    student_data = {
        "student_id": "API_TEST_002",
        "first_name": "Mark",
        "last_name": "Test"
    }
    s_resp = client.post("/api/students/", json=student_data)
    internal_id = s_resp.json()["id"]
    
    # Mark attendance
    mark_data = {
        "student_id": internal_id,
        "status": "late",
        "camera_id": "api_test_cam",
        "confidence_score": 0.99
    }
    response = client.post("/api/attendance/mark", json=mark_data)
    assert response.status_code == 201
    assert response.json()["status"] == "late"
    
    # Check history
    response = client.get("/api/attendance/records", params={"student_id": internal_id})
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_stats():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_students" in data
    assert "attendance_today" in data
    assert "gpu_active" in data
