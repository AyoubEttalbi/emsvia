import pytest
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from database.models import Base, FaceEmbedding
from database.crud import AttendanceDB
from models.embeddings_manager import EmbeddingsManager

TEST_DB_URL = "sqlite:///:memory:"
TEST_CACHE_FILE = "tests/test_cache.pkl"

@pytest.fixture
def db_manager():
    """Fixture for in-memory DB."""
    manager = AttendanceDB(TEST_DB_URL)
    Base.metadata.create_all(manager.engine)
    return manager

@pytest.fixture
def session(db_manager):
    """Fixture for DB session."""
    session = db_manager.get_session()
    yield session
    session.close()

@pytest.fixture
def manager(db_manager):
    """Fixture for EmbeddingsManager with temporary cache file."""
    # Ensure cleanup before start
    if os.path.exists(TEST_CACHE_FILE):
        os.remove(TEST_CACHE_FILE)
        
    em = EmbeddingsManager(db_manager, cache_file=TEST_CACHE_FILE)
    yield em
    
    # Cleanup after test
    if os.path.exists(TEST_CACHE_FILE):
        os.remove(TEST_CACHE_FILE)

def test_add_embedding_updates_cache_and_db(manager, session, db_manager):
    """Test that adding embedding updates both DB and in-memory cache."""
    # Add a dummy student to DB first (FK constraint)
    db_manager.add_student(session, "STU1", "Test", "Student")
    student = db_manager.get_student_by_id(session, "STU1")
    
    vector = np.array([0.1, 0.2, 0.3])
    success = manager.add_embedding(session, student.id, vector)
    
    assert success is True
    
    # Check Cache
    assert student.id in manager.embedding_cache
    assert len(manager.embedding_cache[student.id]) == 1
    np.testing.assert_array_equal(manager.embedding_cache[student.id][0], vector)
    
    # Check DB
    db_embeddings = db_manager.get_student_embeddings(session, student.id)
    assert len(db_embeddings) == 1

def test_load_embeddings_from_db(manager, session, db_manager):
    """Test loading embeddings from DB when cache doesn't exist."""
    # Add data directly to DB
    db_manager.add_student(session, "STU2", "Test2", "Student")
    student = db_manager.get_student_by_id(session, "STU2")
    vector = np.array([0.5, 0.6])
    db_manager.add_face_embedding(session, student.id, vector)
    
    # Load
    manager.load_embeddings(session)
    
    # Verify loaded
    assert manager.cache_loaded is True
    assert student.id in manager.embedding_cache
    np.testing.assert_array_equal(manager.embedding_cache[student.id][0], vector)
    
    # Verify pickle created
    assert os.path.exists(TEST_CACHE_FILE)

def test_load_embeddings_from_pickle(manager, session, db_manager):
    """Test loading from pickle to avoid DB hit."""
    # Setup: Create a pickle file with data
    student_id = 999
    vector = np.array([0.9, 0.9])
    data = {student_id: [vector]}
    
    import pickle
    with open(TEST_CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
        
    # Load (should use pickle)
    manager.load_embeddings(session)
    
    assert manager.cache_loaded is True
    assert student_id in manager.embedding_cache
    np.testing.assert_array_equal(manager.embedding_cache[student_id][0], vector)
