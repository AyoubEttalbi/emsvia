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
from preprocessing.pipeline import preprocess_frame

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_student_images(session, student, student_dir, recognizer, em_manager, db_manager, incremental=True):
    """
    Generates embeddings for a single student's images.

    Args:
        incremental: If True, skip images that already have embeddings in the DB.
                     If False, process all images (full re-generation).
    """
    student_id = student_dir.name
    image_files = list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.png"))
    if not image_files:
        print(f"  No images found for {student_id}.")
        return 0, 0, 0

    # --- Incremental: determine which images to skip ---
    already_processed = set()
    if incremental:
        already_processed = db_manager.get_processed_image_paths(session, student.id)
        skipped = sum(1 for f in image_files if str(f) in already_processed)
        if skipped:
            print(f"  Skipping {skipped}/{len(image_files)} already-processed images.")

    to_process = [f for f in image_files if str(f) not in already_processed]

    if not to_process:
        print(f"  All images already processed for {student_id}. Nothing to do.")
        return 0, 0, len(image_files)

    success_count = 0
    total_embeddings = 0
    for img_path in tqdm(to_process, desc=f"  Generating embeddings for {student_id}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # CRITICAL: Apply same preprocessing as live recognition
        img_preprocessed = preprocess_frame(img)
        if img_preprocessed is None:
            continue

        # Generate embeddings for ALL active models
        embeddings_dict = recognizer.generate_embeddings(img_preprocessed)
        if embeddings_dict:
            for model_name, vector in embeddings_dict.items():
                # Save to DB and Cache — pass image_path so we can track it
                if em_manager.add_embedding(
                    session, student.id, vector,
                    model_name=model_name,
                    image_path=str(img_path)   # <-- track the source image
                ):
                    total_embeddings += 1
            success_count += 1

    return success_count, len(to_process), len(image_files)


def generate_embeddings(incremental=True):
    """
    Scans data/student_images/ and generates embeddings for all students.
    Saves results to Database and Cache.

    Args:
        incremental: If True (default), only process NEW images not yet in the DB.
                     Pass --full as CLI arg to reprocess everything.
    """
    mode_label = "INCREMENTAL" if incremental else "FULL RE-GENERATION"
    print("\n" + "="*55)
    print(f"      EMBEDDING GENERATION TOOL  [{mode_label}]")
    print("="*55 + "\n")

    if incremental:
        print("  Mode: Only NEW images will be processed.")
    else:
        print("  Mode: ALL images will be (re)processed.")
    print()

    # 1. Initialize Components
    db_manager = AttendanceDB(DATABASE_URL)
    session = db_manager.get_session()
    recognizer = FaceRecognizer()
    em_manager = EmbeddingsManager(db_manager)
    em_manager.load_embeddings(session)

    try:
        images_root = BASE_DIR / "data" / "student_images"
        if not images_root.exists():
            print(f"Error: Directory {images_root} does not exist.")
            return

        student_dirs = [d for d in images_root.iterdir() if d.is_dir()]
        print(f"Found {len(student_dirs)} students in data/student_images/.\n")

        total_new = 0
        total_skipped = 0

        for student_dir in student_dirs:
            student_id = student_dir.name
            print(f"Processing Student: {student_id}")

            student = db_manager.get_student_by_id(session, student_id)
            if not student:
                print(f"  Warning: Student {student_id} not found in database. Skipping.\n")
                continue

            success, processed, total_imgs = process_student_images(
                session, student, student_dir, recognizer, em_manager, db_manager,
                incremental=incremental
            )
            skipped = total_imgs - processed
            total_new += success
            total_skipped += skipped

            if processed > 0:
                print(f"  ✓ Generated embeddings from {success}/{processed} new images. ({skipped} skipped)\n")
            else:
                print()

        print("=" * 55)
        print(f"  DONE — {total_new} new embeddings added, {total_skipped} images skipped.")
        print("=" * 55)

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate face embeddings for enrolled students.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-process ALL images, even those already in the database (default: incremental)."
    )
    args = parser.parse_args()
    generate_embeddings(incremental=not args.full)
