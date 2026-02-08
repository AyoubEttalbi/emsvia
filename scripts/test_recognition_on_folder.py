import cv2
import logging
import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.crud import AttendanceDB
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.embeddings_manager import EmbeddingsManager
from core.attendance_manager import AttendanceManager

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_on_folder():
    print("\n" + "="*50)
    print("      RECOGNITION TEST (FOLDER MODE)")
    print("="*50 + "\n")

    # 1. Initialize Components
    db_manager = AttendanceDB(DATABASE_URL)
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    em_manager = EmbeddingsManager(db_manager)
    
    # Load knowledge (embeddings)
    session = db_manager.get_session()
    em_manager.load_embeddings(session)
    known_faces = em_manager.get_all_embeddings()
    session.close()

    if not known_faces:
        print("Error: No known faces found in database. Run generate_embeddings.py first.")
        return

    # 2. Get Folder Path
    folder_path = input("Enter path to the folder you want to TEST (e.g. data/test-faces/c-ronaldo-photos): ").strip()
    source_dir = Path(folder_path)
    if not source_dir.exists():
        print(f"Error: Folder {folder_path} not found.")
        return

    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
    print(f"Found {len(image_files)} images. Analyzing...\n")

    # 3. Process each image
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None: continue

        print(f"Checking: {img_path.name}")
        faces = detector.detect_faces(img)
        
        if not faces:
            print("  -> No face detected.")
            continue

        for face in faces:
            face_crop = detector.extract_face(img, face['box'])
            if face_crop is not None:
                embeddings = recognizer.generate_embeddings(face_crop)
                
                if embeddings:
                    match = recognizer.find_best_match(embeddings, known_faces)
                    if match["match_found"]:
                        print(f"  -> SUCCESS: Recognized as Student ID: {match['student_id']} (Vote Ratio: {match.get('vote_ratio', 0):.2f})")
                    else:
                        print("  -> UNKNOWN: Face detected but not matched to database.")
                else:
                    print("  -> ERROR: Failed to generate embedding for this face.")
        print("-" * 30)

    print("\nTest complete.")

if __name__ == "__main__":
    test_on_folder()
