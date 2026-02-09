import os
import sys
import logging
import traceback
from pathlib import Path

# 0. Early GPU & Environment Setup (MUST BE FIRST)
try:
    # Build LD_LIBRARY_PATH from bundled nvidia-* packages if they exist
    # This helps TensorFlow/ONNX find the correct CUDA libs in the venv
    venv_site_packages = Path(__file__).resolve().parent / "venv" / "lib" / "python3.12" / "site-packages"
    nvidia_base = venv_site_packages / "nvidia"
    
    if nvidia_base.exists():
        cuda_libs = []
        for p in nvidia_base.glob("**/lib"):
            if p.is_dir():
                cuda_libs.append(str(p))
        
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(cuda_libs)
        if current_ld:
            new_ld = f"{new_ld}:{current_ld}"
        
        os.environ["LD_LIBRARY_PATH"] = new_ld
        # Some systems also need this for the linker to pick it up later via TF
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={str(nvidia_base)}"
        
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA Initialized: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Torch: CUDA NOT available in this context.")
except Exception as e:
    print(f"Early Environment Setup warning: {e}")

# Critical: Set NVIDIA library paths before any other imports to enable TensorFlow GPU
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/usr/lib/x86_64-linux-gnu'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce TF spam

import cv2
import time
import threading
import numpy as np
from pathlib import Path
from sqlalchemy.orm import Session

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config.settings import DATABASE_URL, IDENTITY_STABILITY_THRESHOLD, MIN_BUFFER_FOR_RECOG, RECOGNITION_MODELS, DEBUG_MODE
from database.crud import AttendanceDB
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.embeddings_manager import EmbeddingsManager
from core.camera_handler import CameraHandler
from core.attendance_manager import AttendanceManager
from core.unknown_face_handler import UnknownFaceHandler
from core.tracker import FaceTracker
from models.gpu_manager import GPUModelManager

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
    from models.gpu_manager import GPUModelManager
    gpu_mgr = GPUModelManager() # Ensure GPU is initialized
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Load embedding manager and load cache
    em_manager = EmbeddingsManager(db_manager)
    session = db_manager.get_session()
    em_manager.load_embeddings(session)
    
    # Load student names for UI display
    students = db_manager.get_all_students(session)
    student_names = {s.id: f"{s.first_name} {s.last_name}" for s in students}
    session.close() # Close initial session
    
    # Core handlers
    camera = CameraHandler(source=0).start()
    attend_mgr = AttendanceManager(db_manager)
    unknown_mgr = UnknownFaceHandler(db_manager)
    tracker = FaceTracker()
    
    # Speed-focused intervals
    DETECTION_INTERVAL = 2
    RECOGNITION_INTERVAL = 3
    frame_idx = 0
    
    logger.info("All components initialized. Running main loop.")
    
    try:
        while True:
            # 2. Get latest frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue
            
            frame_idx += 1
            display_frame = camera.preprocess_frame(frame)
            
            # 3. Process frame
            # Use a single session for all operations in this frame to reduce overhead
            session = db_manager.get_session()
            
            try:
                if frame_idx % DETECTION_INTERVAL == 0:
                    faces = detector.detect_faces(frame)
                    tracked_faces = tracker.update(faces)
                else:
                    # Update tracker with empty detections to maintain existing tracks via prediction
                    tracked_faces = tracker.update([]) 

                active_student_ids = set()
                active_track_ids = {f['track_id'] for f in tracked_faces}
                
                for face in tracked_faces:
                    x, y, w, h = face['box']
                    tid = face['track_id']
                    track_data = tracker.tracks[tid]
                    track_data['frame_count'] += 1
                    
                    # UI Defaults
                    img_w_disp = display_frame.shape[1]
                    x_disp = img_w_disp - (x + w)
                    color = (255, 0, 0) # Default tracking color
                    label = f"Track: {tid}"
                    
                    try:
                        # 4. Extract and Buffer Embeddings (Throttled by RECOGNITION_INTERVAL)
                        if track_data['frame_count'] % RECOGNITION_INTERVAL == 0:
                            face_crop = detector.extract_face(frame, [x, y, w, h])
                            if face_crop is not None:
                                quality = detector.check_image_quality(face_crop)
                                if quality['passed']:
                                    current_embeddings = recognizer.generate_embeddings(face_crop)
                                    if current_embeddings:
                                        for model_name, emb in current_embeddings.items():
                                            if model_name in track_data['embeddings_buffer']:
                                                track_data['embeddings_buffer'][model_name].append(emb)
                                                if DEBUG_MODE:
                                                    logger.debug(f"Added embedding for T:{tid} (Buf size: {len(track_data['embeddings_buffer'][model_name])})")
                                    else:
                                        logger.warning(f"Failed to generate embeddings for T:{tid}")
                                else:
                                    if DEBUG_MODE:
                                        logger.debug(f"Quality check failed for T:{tid}: {quality}")
                            else:
                                if DEBUG_MODE:
                                    logger.debug(f"Failed to extract face for T:{tid}")
                        else:
                            face_crop = None
                    except Exception as face_err:
                        logger.error(f"Error processing face T:{tid}: {face_err}")
                        face_crop = None

                    # 5. Averaging & Recognition
                    first_model = RECOGNITION_MODELS[0]
                    has_min_buffer = len(track_data['embeddings_buffer'].get(first_model, [])) >= 3
                    
                    # Only calculate match occasionally
                    if has_min_buffer and track_data['frame_count'] % RECOGNITION_INTERVAL == 0:
                        # Average embeddings
                        avg_embeddings = {}
                        for model_name, buffer in track_data['embeddings_buffer'].items():
                            if len(buffer) > 0:
                                avg_embeddings[model_name] = np.array(buffer).mean(axis=0)
                        
                        # Recognize based on averaged embeddings
                        all_known = em_manager.get_all_embeddings()
                        match_result = recognizer.find_best_match(avg_embeddings, all_known)
                        track_data['last_match'] = match_result 

                        # 6. Identity Stability Rule (Improved: only switch on positive matches)
                        new_identity = match_result["student_id"] if match_result["match_found"] else None
                        
                        if new_identity is not None:
                            if new_identity != track_data['current_identity']:
                                track_data['identity_stability_counter'] += 1
                                if track_data['identity_stability_counter'] >= IDENTITY_STABILITY_THRESHOLD:
                                    track_data['current_identity'] = new_identity
                                    track_data['identity_stability_counter'] = 0
                            else:
                                track_data['identity_stability_counter'] = 0
                        # Note: We don't reset to None if match is not found. 
                        # This prevents the label from flickering back to 'Analyzing' or 'Unknown'
                        # if the person is still in frame but a single frame fails to match.

                    # 7. Apply Stable Identity
                    stable_id = track_data['current_identity']
                    if stable_id is not None:
                        name = student_names.get(stable_id, f"ID: {stable_id}")
                        label = f"{name} (T:{tid})"
                        color = (0, 255, 0)
                        active_student_ids.add(stable_id)
                        
                        # Attendance Logic: process_recognition handles its own cooldowns
                        attend_mgr.process_recognition(session, stable_id, track_id=tid, confidence=1.0)
                    else:
                        buffer_len = len(track_data['embeddings_buffer'].get(first_model, []))
                        if buffer_len > 0:
                            label = f"Analyzing... ({buffer_len}/{MIN_BUFFER_FOR_RECOG})"
                            color = (0, 255, 255) # Yellow for processing
                        
                        if buffer_len >= MIN_BUFFER_FOR_RECOG:
                            label = f"Unknown (T:{tid})"
                            color = (0, 0, 255)
                            if track_data['frame_count'] % 60 == 0 and face_crop is not None:
                                unknown_mgr.handle_unknown_face(session, face_crop, confidence=0.0)

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
                
            finally:
                session.close()

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
        logger.error(f"Unexpected error in main loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
