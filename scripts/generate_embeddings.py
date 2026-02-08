import cv2
import os
import sys
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.crud import AttendanceDB
from models.face_recognizer import FaceRecognizer
from models.embeddings_manager import EmbeddingsManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_student_images(session, student, student_dir, recognizer, em_manager):
    """
    Generates embeddings for a single student's images.
    """
    student_id = student_dir.name
    image_files = list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.png"))
    if not image_files:
        print(f"  No images found for {student_id}.")
        return 0, 0

    success_count = 0
    total_embeddings = 0
    for img_path in tqdm(image_files, desc=f"  Generating embeddings for {student_id}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Generate embeddings for ALL active models
        embeddings_dict = recognizer.generate_embeddings(img)
        if embeddings_dict:
            for model_name, vector in embeddings_dict.items():
                # Save to DB and Cache
                if em_manager.add_embedding(session, student.id, vector, model_name=model_name):
                    total_embeddings += 1
            success_count += 1
    
    return success_count, len(image_files)

def generate_embeddings():
    """
    Scans data/student_images/ and generates embeddings for all students.
    Saves results to Database and Cache.
    """
    print("\n" + "="*50)
    print("      EMBEDDING GENERATION TOOL")
    print("="*50 + "\n")

    # 1. Initialize Components
    db_manager = AttendanceDB(DATABASE_URL)
    session = db_manager.get_session()
    recognizer = FaceRecognizer()
    em_manager = EmbeddingsManager(db_manager)

    try:
        images_root = BASE_DIR / "data" / "student_images"
        if not images_root.exists():
            print(f"Error: Directory {images_root} does not exist.")
            return

        student_dirs = [d for d in images_root.iterdir() if d.is_dir()]
        print(f"Found {len(student_dirs)} students in data/student_images/.")

        for student_dir in student_dirs:
            student_id = student_dir.name
            print(f"\nProcessing Student: {student_id}")
            
            student = db_manager.get_student_by_id(session, student_id)
            if not student:
                print(f"  Warning: Student {student_id} not found in database. Skipping.")
                continue

            success, total = process_student_images(session, student, student_dir, recognizer, em_manager)
            print(f"  Successfully generated {success}/{total} embeddings.")

        print("\n" + "="*50)
        print("      PROCESSING COMPLETE")
        print("="*50)

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    generate_embeddings()
