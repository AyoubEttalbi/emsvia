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

from config.settings import DATABASE_URL, IDENTITY_STABILITY_THRESHOLD, MIN_BUFFER_FOR_RECOG, RECOGNITION_MODELS
from database.crud import AttendanceDB
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.embeddings_manager import EmbeddingsManager
from core.camera_handler import CameraHandler
from core.attendance_manager import AttendanceManager
from core.unknown_face_handler import UnknownFaceHandler
from core.tracker import FaceTracker

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
    
    # Optimization State
    frame_count = 0
    DETECTION_INTERVAL = 5
    RECOGNITION_INTERVAL = 10
    
    
    # Performance Monitoring
    timings = {
        "detection": [],
        "recognition": [],
        "total": []
    }
    
    # Cache for student names to avoid DB hits on every frame
    student_name_cache = {}
    
    try:
        while True:
            loop_start = time.time()
            frame_count += 1
            
            # 2. Get latest frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue
                
            # Apply preprocessing (CLAHE enhancement) for both display AND processing
            processed_frame = camera.preprocess_frame(frame)
            
            # 3. Detection (Every N frames)
            det_start = time.time()
            if frame_count % DETECTION_INTERVAL == 0:
                faces = detector.detect_faces(processed_frame)
                tracker.update(faces)
            else:
                tracker.predict()
            det_end = time.time()
            if frame_count % DETECTION_INTERVAL == 0:
                timings["detection"].append(det_end - det_start)
            
            # 4. Recognition (Every M frames, for specific tracks)
            rec_start = time.time()
            if frame_count % RECOGNITION_INTERVAL == 0:
                active_tracks = tracker.get_tracks()
                for track_data in active_tracks:
                    tid = track_data['track_id']
                    
                    # Skip if already stable identity (unless we want to re-verify occasionally)
                    if track_data['current_identity'] is not None:
                         continue

                    # Extract face (from enhanced frame)
                    face_crop = detector.extract_face(processed_frame, track_data['box'], target_size=None)
                    
                    if face_crop is not None:
                        # Conditional Enhancement
                        face_size = face_crop.shape[:2]
                        # Check previous confidence (not easily available directly, passing None for now)
                        face_crop = detector.enhancer.enhance_if_needed(face_crop, face_size)
                        
                        # Resize for model
                        face_crop = cv2.resize(face_crop, (160, 160))

                        quality = detector.check_image_quality(face_crop)
                        if quality['passed']:
                            current_embeddings = recognizer.generate_embeddings(face_crop)
                            if current_embeddings:
                                # Update tracker buffer directly
                                real_track = tracker.tracks[tid]
                                for model_name, emb in current_embeddings.items():
                                    if model_name in real_track['embeddings_buffer']:
                                        real_track['embeddings_buffer'][model_name].append(emb)
                                
                                # Perform recognition logic
                                first_model = RECOGNITION_MODELS[0]
                                buffer = real_track['embeddings_buffer'].get(first_model, [])
                                if len(buffer) >= MIN_BUFFER_FOR_RECOG:
                                    # Average
                                    avg_embeddings = {}
                                    for m_name, buf in real_track['embeddings_buffer'].items():
                                        if buf: avg_embeddings[m_name] = np.mean(list(buf), axis=0)
                                    
                                    # Match
                                    all_known = em_manager.get_all_embeddings()
                                    match_result = recognizer.find_best_match(avg_embeddings, all_known)
                                    real_track['last_match'] = match_result
                                    
                                    # Stability
                                    new_identity = match_result["student_id"] if match_result["match_found"] else None
                                    if new_identity != real_track['current_identity']:
                                        real_track['identity_stability_counter'] += 1
                                        if real_track['identity_stability_counter'] >= IDENTITY_STABILITY_THRESHOLD:
                                            real_track['current_identity'] = new_identity
                                            real_track['identity_stability_counter'] = 0
                                            
                                            # Mark Attendance
                                            if new_identity:
                                                 session = db_manager.get_session()
                                                 attend_mgr.process_recognition(session, new_identity, track_id=tid, confidence=1.0)
                                                 session.close()
                                    else:
                                        real_track['identity_stability_counter'] = 0

            rec_end = time.time()
            if frame_count % RECOGNITION_INTERVAL == 0:
                timings["recognition"].append(rec_end - rec_start)

            # 5. Display & UI (Every Frame)
            active_tracks = tracker.get_tracks() # Get latest state
            active_track_ids = set()
            
            for track_data in active_tracks:
                tid = track_data['track_id']
                active_track_ids.add(tid)
                x, y, w, h = track_data['box']
                
                # Colors
                color = (255, 0, 0) # Unknown/Searching
                label = f"Track {tid}"
                
                if track_data['current_identity']:
                    color = (0, 255, 0)
                    student_id = track_data['current_identity']
                    
                    # Check cache first
                    if student_id not in student_name_cache:
                        session = db_manager.get_session()
                        # The recognizer returns the DB Primary Key (int), not the string ID
                        student = db_manager.get_student_by_pk(session, student_id)
                        if student:
                            # Cache the full display name
                            student_name_cache[student_id] = f"{student.first_name} {student.last_name} ({student.student_id})"
                        else:
                            student_name_cache[student_id] = f"ID: {student_id}"
                        session.close()
                    
                    label = student_name_cache.get(student_id, f"ID: {student_id}")
                elif track_data['frame_count'] > 30: # Long time no ID
                    color = (0, 0, 255)
                    label = "Unknown"
                
                # Draw
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(processed_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Stats
            attend_mgr.clean_stale_tracks(active_track_ids)
            
            loop_end = time.time()
            timings["total"].append(loop_end - loop_start)
            
            # FPS Calculation
            if frame_count % 30 == 0:
                total_times = timings["total"]
                if len(total_times) > 30:
                     avg_total = np.mean(total_times[-30:])
                else:
                     avg_total = np.mean(total_times) if total_times else 0

                fps = 1.0 / avg_total if avg_total > 0 else 0
            
            # Use the averaged FPS (calculated every 30 frames)
            # We initialize fps to 0 before the loop or handle the first 30 frames
            current_fps_display = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --"

            cv2.putText(processed_frame, current_fps_display, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            stats = attend_mgr.get_session_status()
            cv2.putText(processed_frame, f"Marked: {stats['marked_this_session']}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Real-time Attendance System", processed_frame)
            
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
