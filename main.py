import cv2
import logging
import time
import sys
import threading
import numpy as np
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
from core.tracker import FaceTracker
from config.settings import IDENTITY_STABILITY_THRESHOLD, MIN_BUFFER_FOR_RECOG, RECOGNITION_MODELS

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
    tracker = FaceTracker()
    
    logger.info("All components initialized. Running main loop.")
    
    try:
        while True:
            # 2. Get latest frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue
                
            display_frame = camera.preprocess_frame(frame)
            
            # 3. Process frame
            # Optimization: Recognition is expensive. We track every frame, 
            # but only run Recognition/SR every N frames per track.
            RECOGNITION_INTERVAL = 10 # frames
            DEBUG_MODE = True # Set to False for production
            
            faces = detector.detect_faces(frame)
            tracked_faces = tracker.update(faces)
            active_student_ids = set()
            active_track_ids = {f['track_id'] for f in tracked_faces}
            
            for face in tracked_faces:
                x, y, w, h = face['box']
                tid = face['track_id']
                track_data = tracker.tracks[tid]
                track_data['frame_count'] += 1
                
                # UI Defaults
                x_disp = display_frame.shape[1] - (x + w)
                color = (255, 0, 0) # Default tracking color
                label = f"Track: {tid}"
                
                # 4. Extract and Buffer Embeddings (Every frame if quality OK)
                face_crop = detector.extract_face(frame, [x, y, w, h])
                if face_crop is not None:
                    quality = detector.check_image_quality(face_crop)
                    if quality['passed']:
                        current_embeddings = recognizer.generate_embeddings(face_crop)
                        if current_embeddings:
                            for model_name, emb in current_embeddings.items():
                                if model_name in track_data['embeddings_buffer']:
                                    track_data['embeddings_buffer'][model_name].append(emb)

                # 5. Averaging & Recognition
                # Check if we have enough samples in the buffer for the first model
                first_model = RECOGNITION_MODELS[0]
                if len(track_data['embeddings_buffer'].get(first_model, [])) >= MIN_BUFFER_FOR_RECOG:
                    # Average embeddings
                    avg_embeddings = {}
                    for model_name, buffer in track_data['embeddings_buffer'].items():
                        if len(buffer) > 0:
                            avg_embeddings[model_name] = np.mean(list(buffer), axis=0)
                    
                    # Recognize based on averaged embeddings
                    all_known = em_manager.get_all_embeddings()
                    match_result = recognizer.find_best_match(avg_embeddings, all_known)
                    track_data['last_match'] = match_result # For UI diagnostics

                    # 6. Identity Stability Rule
                    new_identity = match_result["student_id"] if match_result["match_found"] else None
                    
                    if new_identity != track_data['current_identity']:
                        track_data['identity_stability_counter'] += 1
                        if track_data['identity_stability_counter'] >= IDENTITY_STABILITY_THRESHOLD:
                            track_data['current_identity'] = new_identity
                            track_data['identity_stability_counter'] = 0
                    else:
                        track_data['identity_stability_counter'] = 0

                # 7. Apply Stable Identity
                stable_id = track_data['current_identity']
                if stable_id is not None:
                    label = f"ID: {stable_id} (T:{tid})"
                    color = (0, 255, 0)
                    active_student_ids.add(stable_id)
                    
                    # Attendance Logic (Temporal evidence already handled by averaging/stability)
                    session = db_manager.get_session()
                    attend_mgr.process_recognition(session, stable_id, track_id=tid, confidence=1.0)
                    session.close()
                else:
                    # If confirmed None and buffer full, it's truly Unknown
                    if len(track_data['embeddings_buffer'].get(first_model, [])) >= MIN_BUFFER_FOR_RECOG:
                        label = f"Unknown (T:{tid})"
                        color = (0, 0, 255)
                        # Periodic unknown handling
                        if track_data['frame_count'] % 30 == 0 and face_crop is not None:
                            session = db_manager.get_session()
                            unknown_mgr.handle_unknown_face(session, face_crop, confidence=0.0)
                            session.close()

                # Visual Debug Info
                if DEBUG_MODE:
                    buffer_len = len(track_data['embeddings_buffer'].get(first_model, []))
                    cv2.putText(display_frame, f"Buf: {buffer_len} | Stab: {track_data['identity_stability_counter']}", 
                                (x_disp, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw UI overlays
                cv2.rectangle(display_frame, (x_disp, y), (x_disp + w, y + h), color, 2)
                cv2.putText(display_frame, label, (x_disp, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Clean up stale tracks in manager
            attend_mgr.clean_stale_tracks(active_track_ids)

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
