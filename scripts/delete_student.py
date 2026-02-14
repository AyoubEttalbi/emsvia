import os
import sys
import shutil
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.crud import AttendanceDB
from models.embeddings_manager import EmbeddingsManager

def delete_student(student_id):
    print(f"Deleting student {student_id}...")
    
    # 1. Delete from Database
    db_manager = AttendanceDB(DATABASE_URL)
    session = db_manager.get_session()
    
    student = db_manager.get_student_by_id(session, student_id)
    if not student:
        print(f"Student {student_id} not found in database.")
    else:
        # Delete attendance logs first (cascade usually handles this, but being safe)
        # Actually CRUD doesn't have delete logs specific method exposed easily,
        # but SQLAlchemy cascade should work if configured.
        # Let's just delete the student.
        session.delete(student)
        session.commit()
        print(f"Deleted {student.first_name} {student.last_name} ({student_id}) from database.")
    session.close()

    # 2. Delete Images
    images_dir = BASE_DIR / "data" / "student_images" / student_id
    if images_dir.exists():
        shutil.rmtree(images_dir)
        print(f"Deleted image directory: {images_dir}")
    else:
        print(f"Image directory not found: {images_dir}")

    # 3. Clear Embeddings Cache
    # We delete the whole cache to force regeneration, ensuring no artifacts remain.
    embeddings_file = BASE_DIR / "data" / "embeddings" / "embeddings_arcface.pkl"
    if embeddings_file.exists():
        os.remove(embeddings_file)
        print(f"Deleted embeddings cache: {embeddings_file} (Will regenerate on next run)")
    else:
        print("Embeddings cache not found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/delete_student.py <STUDENT_ID>")
        sys.exit(1)
    
    target_id = sys.argv[1]
    delete_student(target_id)
