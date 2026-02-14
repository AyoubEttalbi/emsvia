import sys
import os
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.models import Base, Student
from database.crud import AttendanceDB
from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database():
    """
    Initialize the database, create tables, and seed with initial data.
    """
    logger.info(f"Initializing database at: {DATABASE_URL}")
    
    # Ensure database directory exists
    db_path = BASE_DIR / "database"
    if not db_path.exists():
        db_path.mkdir(parents=True)
        logger.info(f"Created database directory: {db_path}")

    # Initialize DB manager
    db_manager = AttendanceDB(DATABASE_URL)
    
    # Create tables
    try:
        Base.metadata.create_all(db_manager.engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return

    # Seed initial data
    session = db_manager.get_session()
    try:
        seed_data(db_manager, session)
        logger.info("Initial seed data added successfully.")
    except Exception as e:
        logger.error(f"Error seeding data: {e}")
        session.rollback()
    finally:
        session.close()

def seed_data(db_manager: AttendanceDB, session: Session):
    """
    Add some initial test students if the table is empty.
    """
    student_count = session.query(Student).count()
    if student_count > 0:
        logger.info("Database already contains students. Skipping seed.")
        return

    # Sample students
    sample_students = [
        ("STU001", "John", "Doe", "john.doe@example.com", "1234567890"),
        ("STU002", "Jane", "Smith", "jane.smith@example.com", "0987654321"),
        ("STU003", "Alice", "Johnson", "alice.j@example.com", "5551234567"),
    ]

    for stu_id, f_name, l_name, email, phone in sample_students:
        db_manager.add_student(session, stu_id, f_name, l_name, email, phone)

if __name__ == "__main__":
    setup_database()
