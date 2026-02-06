import cv2
import logging
import time
import sys
import threading
from pathlib import Path
from sqlalchemy.orm import Session

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL
from database.crud import AttendanceDB
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.embeddings_manager import EmbeddingsManager
from core.camera_handler import CameraHandler
from core.attendance_manager import AttendanceManager
from core.unknown_face_handler import UnknownFaceHandler

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the real-time attendance system.
    """
    logger.info("Starting Attendance System...")
    
    # 1. Initialize Components
    db_manager = AttendanceDB(DATABASE_URL)
    
    # Load detector and recognizer
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Load embedding manager and load cache
    em_manager = EmbeddingsManager(db_manager)
    session = db_manager.get_session()
    em_manager.load_embeddings(session)
    session.close() # Close initial session
    
    # Core handlers
    camera = CameraHandler(source=0).start()
    attend_mgr = AttendanceManager(db_manager)
    unknown_mgr = UnknownFaceHandler(db_manager)
    
    logger.info("All components initialized. Running main loop.")
    
    try:
        while True:
            # 2. Get latest frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue
                
            display_frame = camera.preprocess_frame(frame)
            
            # 3. Process frame (every few frames or based on logic)
            # For simplicity, we process every frame here, but MTCNN is heavy.
            # In a production app, we might use a faster tracker or skip frames.
            
            faces = detector.detect_faces(frame)
            active_student_ids = set()
            
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
                
                # Draw bounding box (mirrored for display)
                x_disp = display_frame.shape[1] - (x + w)
                color = (0, 255, 0)
                
                # 4. Recognition
                face_crop = detector.extract_face(frame, [x, y, w, h])
                if face_crop is not None:
                    # Quality check for recognition
                    quality = detector.check_image_quality(face_crop)
                    
                    if quality['passed']:
                        current_embedding = recognizer.generate_embedding(face_crop)
                        
                        if current_embedding is not None:
                            # Compare with known embeddings
                            all_known = em_manager.get_all_embeddings()
                            match_result = recognizer.find_best_match(current_embedding, all_known)
                            
                            if match_result["match_found"]:
                                student_id = match_result["student_id"]
                                dist = match_result["distance"]
                                
                                # Resolve student name from ID (using internal cache or quick DB lookups)
                                # For now, just show ID
                                label = f"ID: {student_id} (D:{dist:.2f})"
                                color = (0, 255, 0)
                                active_student_ids.add(student_id)
                                
                                # 5. Attendance Logic
                                # confidence = 1.0 - distance (clamped to 0-1 range)
                                confidence = max(0.0, min(1.0, 1.0 - dist))
                                session = db_manager.get_session()
                                attend_mgr.process_recognition(session, student_id, confidence=confidence)
                                session.close()
                                
                            else:
                                label = "Unknown"
                                color = (0, 0, 255)
                                # 6. Unknown Face Logic
                                session = db_manager.get_session()
                                unknown_mgr.handle_unknown_face(session, face_crop, confidence=confidence)
                                session.close()
                        else:
                            label = "Recog Failed"
                    else:
                        label = "Low Quality"
                        color = (0, 255, 255)

                # Draw UI overlays
                cv2.rectangle(display_frame, (x_disp, y), (x_disp + w, y + h), color, 2)
                cv2.putText(display_frame, label, (x_disp, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Reset counts for students who left the frame
            attend_mgr.reset_counts_except(active_student_ids)

            # Dashboard Info
            fps = camera.get_fps()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            stats = attend_mgr.get_session_status()
            cv2.putText(display_frame, f"Marked: {stats['marked_this_session']}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 7. Show frame
            cv2.imshow("Real-time Attendance System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting...")
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
