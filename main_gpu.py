"""
GPU-Optimized Main Pipeline for Face Recognition Attendance System.
Use this instead of main.py when GPU is available for 10-20x speedup.

Features:
- Batch face processing
- Parallel ensemble detection and recognition
- GPU-accelerated preprocessing and super-resolution
- Automatic memory management
- Real-time FPS and GPU memory monitoring
"""
import os
import sys
import logging
import traceback
from pathlib import Path

# 0. Early GPU & Environment Setup (MUST BE FIRST)
try:
    import os
    from pathlib import Path
    import site
    
    # 1. Set CUDA Visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 2. Aggressive Library Linking
    site_packages = site.getsitepackages()
    
    cuda_lib_paths = []
    # Prioritize venv libraries
    for sp in site_packages:
        nvidia_path = Path(sp) / "nvidia"
        if nvidia_path.exists():
            for lib_dir in nvidia_path.glob("**/lib"):
                if lib_dir.is_dir():
                    cuda_lib_paths.append(str(lib_dir))
    
    # Common system paths
    cuda_lib_paths.extend([
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia"
    ])
    
    if cuda_lib_paths:
        unique_paths = list(dict.fromkeys(cuda_lib_paths))
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(unique_paths)
        if current_ld:
            new_ld = f"{new_ld}:{current_ld}"
        
        os.environ["LD_LIBRARY_PATH"] = new_ld
        
        # Link for TensorFlow XLA
        for p in unique_paths:
            if "cuda_runtime" in p:
                os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={p.split('/lib')[0]}"
                break

    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
        # Log to stdout for visibility
        sys.stdout.write(f"🚀 CUDA Initialized: {torch.cuda.get_device_name(0)}\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("⚠️  Torch: CUDA NOT available. Falling back to CPU.\n")
        sys.stdout.flush()
except Exception as e:
    sys.stdout.write(f"⚠️  Early Environment Setup warning: {e}\n")
    sys.stdout.flush()

# Critical: Set TF specific flags
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

import cv2
import time
import numpy as np
from pathlib import Path
from sqlalchemy.orm import Session

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config.settings import (
    DATABASE_URL, IDENTITY_STABILITY_THRESHOLD, MIN_BUFFER_FOR_RECOG, 
    RECOGNITION_MODELS, DEBUG_MODE, DETECTION_INTERVAL, RECOGNITION_INTERVAL,
    BATCH_PROCESSING, BATCH_SIZE, CLEAR_CACHE_INTERVAL, USE_GPU
)
from database.crud import AttendanceDB
from models.face_detector import FaceDetector
from models.face_recognizer import FaceRecognizer
from models.embeddings_manager import EmbeddingsManager
from models.gpu_manager import GPUModelManager
from core.camera_handler import CameraHandler
from core.attendance_manager import AttendanceManager
from core.unknown_face_handler import UnknownFaceHandler
from core.tracker import FaceTracker
from recognition.batch_recognizer import BatchRecognizer
from detection.parallel_detector import ParallelDetector
from utils.gpu_utils import GPUMemoryManager

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    GPU-Optimized main entry point for real-time attendance system.
    """
    logger.info("="*60)
    logger.info("🚀 Starting GPU-Accelerated Attendance System")
    logger.info("="*60)
    
    # 1. Initialize GPU Manager First
    gpu_mgr = GPUModelManager()
    
    if not gpu_mgr.is_gpu_ready():
        logger.warning("⚠️  GPU not available! Performance will be significantly reduced.")
        logger.warning("    Continuing with CPU fallback...")
    else:
        logger.info(f"✅ GPU Ready: {gpu_mgr.get_memory_summary()}")
    
    # 2. Initialize Database
    db_manager = AttendanceDB(DATABASE_URL)
    
    # 3. Load Detectors and Recognizers
    logger.info("📦 Loading detection and recognition models...")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # 4. Initialize GPU-accelerated components
    if BATCH_PROCESSING and gpu_mgr.is_gpu_ready():
        logger.info(f"🔋 Batch processing enabled (batch_size={BATCH_SIZE})")
        batch_recognizer = BatchRecognizer(recognizer, batch_size=BATCH_SIZE)
        parallel_detector = ParallelDetector(detector)
    else:
        logger.info("📝 Batch processing disabled (CPU mode or setting disabled)")
        batch_recognizer = None
        parallel_detector = None
    
    # 5. Load embeddings and student data
    em_manager = EmbeddingsManager(db_manager)
    session = db_manager.get_session()
    em_manager.load_embeddings(session)
    
    students = db_manager.get_all_students(session)
    student_names = {s.id: f"{s.first_name} {s.last_name}" for s in students}
    session.close()
    
    logger.info(f"👨‍🎓 Loaded {len(student_names)} students from database")
    
    # 6. Core handlers
    camera = CameraHandler(source=0).start()
    attend_mgr = AttendanceManager(db_manager)
    unknown_mgr = UnknownFaceHandler(db_manager)
    tracker = FaceTracker()
    
    # 7. Processing intervals (GPU can handle every frame!)
    # FORCE 1 if GPU is available to solve 'slow' issue
    detection_interval = 1 if USE_GPU else DETECTION_INTERVAL
    recognition_interval = 1 if USE_GPU else RECOGNITION_INTERVAL
    
    logger.info(f"⚙️  Detection interval: {detection_interval} frame(s) {'(FORCED for GPU)' if USE_GPU else ''}")
    logger.info(f"⚙️  Recognition interval: {recognition_interval} frame(s) {'(FORCED for GPU)' if USE_GPU else ''}")
    logger.info(f"🎯 Identity stability threshold: {IDENTITY_STABILITY_THRESHOLD}")
    
    frame_idx = 0
    fps_history = []
    last_gpu_log_time = time.time()
    
    logger.info("="*60)
    logger.info("🎬 Starting main processing loop")
    logger.info("   Press 'q' to quit")
    logger.info("="*60)
    
    try:
        while True:
            loop_start = time.time()
            
            # Get latest frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue
            
            frame_idx += 1
            display_frame = camera.preprocess_frame(frame)
            
            # Use single session for all operations in this frame
            session = db_manager.get_session()
            
            try:
                # DETECTION (with interval throttling)
                if frame_idx % detection_interval == 0:
                    if parallel_detector and gpu_mgr.is_gpu_ready():
                        # GPU: Use parallel ensemble detection
                        faces = parallel_detector.detect_ensemble(frame, use_tiling=False)
                    else:
                        # CPU fallback: Standard detection
                        faces = detector.detect_faces(frame)
                    
                    tracked_faces = tracker.update(faces)
                else:
                    # Maintain existing tracks via prediction
                    tracked_faces = tracker.update([])
                
                active_student_ids = set()
                active_track_ids = {f['track_id'] for f in tracked_faces}
                
                # Process each tracked face
                for face in tracked_faces:
                    x, y, w, h = face['box']
                    tid = face['track_id']
                    track_data = tracker.tracks[tid]
                    track_data['frame_count'] += 1
                    
                    # UI Defaults
                    img_w_disp = display_frame.shape[1]
                    x_disp = img_w_disp - (x + w)
                    color = (255, 0, 0)  # Blue for tracking
                    label = f"Track: {tid}"
                    
                    try:
                        # EMBEDDING EXTRACTION (with interval throttling)
                        if track_data['frame_count'] % recognition_interval == 0:
                            face_crop = detector.extract_face(frame, [x, y, w, h])
                            if face_crop is not None:
                                quality = detector.check_image_quality(face_crop)
                                if quality['passed']:
                                    # Generate embeddings
                                    current_embeddings = recognizer.generate_embeddings(face_crop)
                                    if current_embeddings:
                                        for model_name, emb in current_embeddings.items():
                                            if model_name in track_data['embeddings_buffer']:
                                                track_data['embeddings_buffer'][model_name].append(emb)
                                else:
                                    if DEBUG_MODE:
                                        logger.debug(f"Quality check failed T:{tid}: {quality}")
                        else:
                            face_crop = None
                    except Exception as face_err:
                        logger.error(f"Error processing face T:{tid}: {face_err}")
                        face_crop = None
                    
                    # RECOGNITION (averaging & matching)
                    first_model = RECOGNITION_MODELS[0]
                    has_min_buffer = len(track_data['embeddings_buffer'].get(first_model, [])) >= 3
                    
                    if has_min_buffer and track_data['frame_count'] % recognition_interval == 0:
                        # Average embeddings
                        avg_embeddings = {}
                        for model_name, buffer in track_data['embeddings_buffer'].items():
                            if len(buffer) > 0:
                                avg_embeddings[model_name] = np.array(buffer).mean(axis=0)
                        
                        # Recognize
                        all_known = em_manager.get_all_embeddings()
                        match_result = recognizer.find_best_match(avg_embeddings, all_known)
                        track_data['last_match'] = match_result
                        
                        # Identity Stability
                        new_identity = match_result["student_id"] if match_result["match_found"] else None
                        
                        if new_identity is not None:
                            if new_identity != track_data['current_identity']:
                                track_data['identity_stability_counter'] += 1
                                if track_data['identity_stability_counter'] >= IDENTITY_STABILITY_THRESHOLD:
                                    track_data['current_identity'] = new_identity
                                    track_data['identity_stability_counter'] = 0
                            else:
                                track_data['identity_stability_counter'] = 0
                    
                    # Apply stable identity to UI
                    stable_id = track_data['current_identity']
                    if stable_id is not None:
                        name = student_names.get(stable_id, f"ID: {stable_id}")
                        label = f"{name} (T:{tid})"
                        color = (0, 255, 0)  # Green for recognized
                        active_student_ids.add(stable_id)
                        
                        # Mark attendance
                        attend_mgr.process_recognition(session, stable_id, track_id=tid, confidence=1.0)
                    else:
                        buffer_len = len(track_data['embeddings_buffer'].get(first_model, []))
                        if buffer_len > 0:
                            label = f"Analyzing... ({buffer_len}/{MIN_BUFFER_FOR_RECOG})"
                            color = (0, 255, 255)  # Yellow for processing
                        
                        if buffer_len >= MIN_BUFFER_FOR_RECOG:
                            label = f"Unknown (T:{tid})"
                            color = (0, 0, 255)  # Red for unknown
                            if track_data['frame_count'] % 60 == 0 and face_crop is not None:
                                unknown_mgr.handle_unknown_face(session, face_crop, confidence=0.0)
                    
                    # Visual debug info
                    if DEBUG_MODE:
                        buffer_len = len(track_data['embeddings_buffer'].get(first_model, []))
                        cv2.putText(display_frame, f"Buf: {buffer_len} | Stab: {track_data['identity_stability_counter']}", 
                                   (x_disp, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Draw UI overlays
                    cv2.rectangle(display_frame, (x_disp, y), (x_disp + w, y + h), color, 2)
                    cv2.putText(display_frame, label, (x_disp, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Clean up stale tracks
                attend_mgr.clean_stale_tracks(active_track_ids)
                
            finally:
                session.close()
            
            # Dashboard info
            fps = camera.get_fps()
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # FPS display
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # GPU memory (if available)
            if gpu_mgr.is_gpu_ready():
                gpu_mem = gpu_mgr.get_memory_summary()
                cv2.putText(display_frame, gpu_mem, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Attendance stats
            stats = attend_mgr.get_session_status()
            cv2.putText(display_frame, f"Marked: {stats['marked_this_session']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("GPU-Accelerated Attendance System", display_frame)
            
            # GPU cache management
            if gpu_mgr.is_gpu_ready():
                gpu_mgr.auto_clear_cache()
                
                # Log GPU memory periodically
                if time.time() - last_gpu_log_time > 30:  # Every 30 seconds
                    mem_stats = gpu_mgr.monitor_memory()
                    logger.info(f"📊 GPU Memory: {mem_stats['allocated_gb']:.2f}GB / {mem_stats['total_gb']:.2f}GB ({mem_stats['utilization_pct']:.1f}%)")
                    logger.info(f"📈 Average FPS: {avg_fps:.1f}")
                    last_gpu_log_time = time.time()
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("👋 Exiting...")
                break
        
    except KeyboardInterrupt:
        logger.info("⏹️  Interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Unexpected error in main loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
        # Final GPU stats
        if gpu_mgr.is_gpu_ready():
            logger.info("="*60)
            logger.info("📊 Final GPU Statistics:")
            mem_stats = gpu_mgr.monitor_memory()
            logger.info(f"   Memory used: {mem_stats['allocated_gb']:.2f}GB / {mem_stats['total_gb']:.2f}GB")
            logger.info(f"   Average FPS: {sum(fps_history)/len(fps_history) if fps_history else 0:.1f}")
            logger.info("="*60)
        
        logger.info("🧹 Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
