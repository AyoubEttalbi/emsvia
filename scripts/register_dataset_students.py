import sys
from pathlib import Path
import logging

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.crud import AttendanceDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def register_students():
    db_manager = AttendanceDB(DATABASE_URL)
    session = db_manager.get_session()
    
    images_dir = BASE_DIR / "data" / "student_images"
    if not images_dir.exists():
        logger.error(f"Image directory {images_dir} does not exist.")
        return

    # Get student names from folder names
    student_folders = [d.name for d in images_dir.iterdir() if d.is_dir() and d.name != "STU001"]
    
    logger.info(f"Found {len(student_folders)} potential students to register.")
    
    count = 0
    for s_id in student_folders:
        # Check if student already exists
        existing = db_manager.get_student_by_id(session, s_id)
        if existing:
            logger.info(f"Student {s_id} already exists in database. Skipping.")
            continue
            
        # Add student
        # We use s_id as both student_id and first_name for now
        student = db_manager.add_student(
            session=session,
            student_id=s_id,
            first_name=s_id,
            last_name="Dataset" # Placeholder
        )
        if student:
            count += 1
            
    session.close()
    logger.info(f"Successfully registered {count} new students.")

if __name__ == "__main__":
    register_students()
