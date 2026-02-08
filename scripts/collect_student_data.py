import cv2
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.crud import AttendanceDB
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.embeddings_manager import EmbeddingsManager
from preprocessing.pipeline import preprocess_frame
from scripts.generate_embeddings import process_student_images

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_next_student_id(db_manager: AttendanceDB, session) -> str:
    """
    Calculate the next available Student ID in the format STUxxx.
    """
    students = db_manager.get_all_students(session, active_only=False)
    if not students:
        return "STU001"
    
    max_id = 0
    for s in students:
        try:
            if s.student_id.startswith("STU"):
                num = int(s.student_id[3:])
                if num > max_id:
                    max_id = num
            else:
                # Handle cases like 'stu001' case-insensitively if needed
                if s.student_id.upper().startswith("STU"):
                    num = int(s.student_id[3:])
                    if num > max_id:
                        max_id = num
        except (ValueError, IndexError):
            continue
            
    return f"STU{max_id + 1:03d}"

def collect_student_data():
    """
    Script to enroll a new student and collect their facial images.
    Supports either LIVE camera mode or FOLDER mode (for WSL/Remote).
    """
    print("\n" + "="*50)
    print("      STUDENT ENROLLMENT TOOL")
    print("="*50 + "\n")

    # 1. Initialize Components
    db_manager = AttendanceDB(DATABASE_URL)
    session = db_manager.get_session()
    detector = FaceDetector(min_confidence=0.9)

    try:
        # 2. Assign Student ID automatically
        student_id = get_next_student_id(db_manager, session)
        print(f"Assigning Student ID: {student_id}")
        
        # 3. Collect Metadata
        first_name = input("Enter First Name: ").strip()
        last_name = input("Enter Last Name: ").strip()
        email = input("Enter Email (optional): ").strip() or None
        
        # 4. Choose Mode
        print("\nHow would you like to provide images?")
        print("1. Live Camera (Requires USB/IP Bridge)")
        print("2. Folder Mode (Load existing photos from a directory)")
        mode_choice = input("Select mode (1/2): ").strip()

        # 5. Create Student in DB
        student = db_manager.add_student(session, student_id, first_name, last_name, email)
        if not student:
            print("Failed to create student in database.")
            return

        # 6. Prepare Storage
        images_dir = BASE_DIR / "data" / "student_images" / student_id
        if not images_dir.exists():
            images_dir.mkdir(parents=True)

        captured_count = 0
        if mode_choice == "2":
            # --- FOLDER MODE ---
            folder_path = input("Enter path to your image folder: ").strip()
            source_dir = Path(folder_path)
            if not source_dir.exists():
                print(f"Error: Directory {folder_path} not found.")
                return

            image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
            print(f"Found {len(image_files)} images. Validating and processing...")

            for img_path in image_files:
                frame = cv2.imread(str(img_path))
                if frame is None: continue
                
                validation = detector.validate_for_enrollment(frame)
                if validation['passed']:
                    face_crop = validation['face_crop']
                    save_name = f"{student_id}_{captured_count:02d}.jpg"
                    cv2.imwrite(str(images_dir / save_name), face_crop)
                    captured_count += 1
                
            print(f"\nEnrollment complete! {captured_count} valid face images saved.")

        else:
            # --- CAMERA MODE ---
            source = input("Enter camera index (0) or IP URL: ").strip()
            try:
                # If it's a number, convert it
                source = int(source)
            except:
                pass # Keep as string (URL)

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return

            print(f"\nStudent {first_name} registered successfully with ID {student_id}.")
            print("Press 'c' to capture, 'q' to quit.")

            stages = [
                {"name": "FRONTAL", "target": 20, "message": "Look directly at the camera."},
                {"name": "LEFT PROFILE", "target": 15, "message": "Turn your head SLIGHTLY to the LEFT."},
                {"name": "RIGHT PROFILE", "target": 15, "message": "Turn your head SLIGHTLY to the RIGHT."}
            ]
            
            stage_idx = 0
            captured_count = 0

            while stage_idx < len(stages):
                current_stage = stages[stage_idx]
                stage_captured = 0
                
                print(f"\n--- STAGE: {current_stage['name']} ---")
                print(f"Instruction: {current_stage['message']}")

                while stage_captured < current_stage['target']:
                    ret, frame = cap.read()
                    if not ret: break

                    display_frame = cv2.flip(frame, 1) if isinstance(source, int) else frame
                    validation = detector.validate_for_enrollment(frame)
                    
                    status_color = (0, 255, 0) if validation['passed'] else (0, 0, 255)
                    
                    # UI Overlays
                    cv2.putText(display_frame, f"STAGE: {current_stage['name']} ({stage_captured}/{current_stage['target']})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, current_stage['message'], (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(display_frame, validation['message'], (10, display_frame.shape[0] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                    if validation['face_box']:
                        x, y, w, h = validation['face_box']
                        x_disp = display_frame.shape[1] - (x + w) if isinstance(source, int) else x
                        cv2.rectangle(display_frame, (x_disp, y), (x_disp + w, y + h), status_color, 2)

                    cv2.imshow("Student Enrollment", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): 
                        stage_idx = 99 # Break outer
                        break
                    elif key in [ord('c'), 32]: # 'c' or Space
                        if validation['passed']:
                            face_crop = validation['face_crop']
                            
                            # CRITICAL: Apply same preprocessing as live recognition
                            processed_face = preprocess_frame(face_crop)
                            
                            save_name = f"{student_id}_{captured_count:02d}.jpg"
                            cv2.imwrite(str(images_dir / save_name), processed_face)
                            
                            captured_count += 1
                            stage_captured += 1
                            print(f"Captured {stage_captured}/{current_stage['target']} for {current_stage['name']}")

                stage_idx += 1

            cap.release()
            cv2.destroyAllWindows()

        # 7. Post-enrollment Embedding Generation
        if captured_count > 0:
            choice = input(f"\nEnrollment finished. Do you want to auto-generate embeddings for {student_id} now? (y/n): ").strip().lower()
            if choice == 'y':
                print("\nInitializing Face Recognizer and Embedding Manager...")
                recognizer = FaceRecognizer()
                em_manager = EmbeddingsManager(db_manager)
                
                success, total = process_student_images(session, student, images_dir, recognizer, em_manager)
                print(f"Successfully generated {success}/{total} embeddings for {student_id}.")
            else:
                print(f"\nSkipping embedding generation. You can run it manually later using: python scripts/generate_embeddings.py")

    except Exception as e:
        logger.error(f"Error during enrollment: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    collect_student_data()
