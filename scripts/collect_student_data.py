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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # 2. Collect Metadata
        student_id = input("Enter Student ID (e.g., STU001): ").strip()
        
        # Check if student already exists
        existing_student = db_manager.get_student_by_id(session, student_id)
        if existing_student:
            print(f"Error: Student with ID {student_id} already exists ({existing_student.first_name} {existing_student.last_name}).")
            return

        first_name = input("Enter First Name: ").strip()
        last_name = input("Enter Last Name: ").strip()
        email = input("Enter Email (optional): ").strip() or None
        
        # 3. Choose Mode
        print("\nHow would you like to provide images?")
        print("1. Live Camera (Requires USB/IP Bridge)")
        print("2. Folder Mode (Load existing photos from a directory)")
        mode_choice = input("Select mode (1/2): ").strip()

        # 4. Create Student in DB
        student = db_manager.add_student(session, student_id, first_name, last_name, email)
        if not student:
            print("Failed to create student in database.")
            return

        # 5. Prepare Storage
        images_dir = BASE_DIR / "data" / "student_images" / student_id
        if not images_dir.exists():
            images_dir.mkdir(parents=True)

        if mode_choice == "2":
            # --- FOLDER MODE ---
            folder_path = input("Enter path to your image folder: ").strip()
            source_dir = Path(folder_path)
            if not source_dir.exists():
                print(f"Error: Directory {folder_path} not found.")
                return

            image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpeg"))
            print(f"Found {len(image_files)} images. Validating and processing...")

            captured_count = 0
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

            print(f"\nStudent {first_name} registered successfully.")
            print("Press 'c' to capture, 'q' to quit.")

            captured_count = 0
            target_count = 55

            while captured_count < target_count:
                ret, frame = cap.read()
                if not ret: break

                display_frame = cv2.flip(frame, 1) if isinstance(source, int) else frame
                validation = detector.validate_for_enrollment(frame)
                
                status_color = (0, 255, 0) if validation['passed'] else (0, 0, 255)
                cv2.putText(display_frame, validation['message'], (10, display_frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                if validation['face_box']:
                    x, y, w, h = validation['face_box']
                    x_disp = display_frame.shape[1] - (x + w) if isinstance(source, int) else x
                    cv2.rectangle(display_frame, (x_disp, y), (x_disp + w, y + h), status_color, 2)

                cv2.imshow("Student Enrollment", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key in [ord('c'), 32]: # 'c' or Space
                    if validation['passed']:
                        face_crop = validation['face_crop']
                        save_name = f"{student_id}_{captured_count:02d}.jpg"
                        cv2.imwrite(str(images_dir / save_name), face_crop)
                        captured_count += 1
                        print(f"Captured {captured_count}/{target_count}")

            cap.release()
            cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Error during enrollment: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    collect_student_data()
