"""
GPU-Optimized Main Pipeline for Face Recognition Attendance System.
Use this instead of main.py when GPU is available for 10-20x speedup.

Features:
- Batch face processing
- Single-model optimization (RetinaFace + ArcFace)
- Face Tracking (Detect once, track N frames)
- Event-based recognition (Identity stability filter)
- GPU-accelerated preprocessing
- Real-time FPS and GPU memory monitoring
"""
import os
import sys
import logging
import traceback
import time
import cv2
import numpy as np
import argparse
from pathlib import Path
from sqlalchemy.orm import Session

# 0. Early GPU & Environment Setup
try:
    import site
    
    # 1. Force Legacy Keras for compatibility with DeepFace and Keras 3 issues
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    
    # 2. Set CUDA Visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 2. Aggressive Library Linking
    site_packages = site.getsitepackages()
    cuda_lib_paths = []
    # Prioritize venv libraries
    for sp in site_packages:
        nvidia_path = Path(sp) / "nvidia"
        if nvidia_path.exists():
            # Add all lib directories under nvidia/
            for lib_dir in nvidia_path.glob("**/lib"):
                if lib_dir.is_dir():
                    cuda_lib_paths.append(str(lib_dir))
    
    # Common system paths
    cuda_lib_paths.extend([
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia",
        "/usr/local/cuda/targets/x86_64-linux/lib"
    ])
    
    if cuda_lib_paths:
        unique_paths = list(dict.fromkeys(cuda_lib_paths))
        # Add side-loaded cuDNN 8 for TensorFlow
        tf_cudnn8_path = "/home/ayoub/projects/Ai-Ml/emsvia/libs/cudnn8"
        if os.path.exists(tf_cudnn8_path):
            unique_paths.insert(0, tf_cudnn8_path)
            
            # PRE-LOAD LIBRARIES to ensure they are available to TensorFlow
            import ctypes
            for lib_name in [
                "libcudnn.so.8", "libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8",
                "libcudnn_adv_infer.so.8"
            ]:
                try:
                    ctypes.CDLL(os.path.join(tf_cudnn8_path, lib_name), mode=ctypes.RTLD_GLOBAL)
                    sys.stdout.write(f"  📦 Pre-loaded {lib_name}\n")
                except Exception as le:
                    sys.stdout.write(f"  ⚠️  Failed to pre-load {lib_name}: {le}\n")
            
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
        sys.stdout.write(f"🚀 PyTorch CUDA Initialized: {torch.cuda.get_device_name(0)}\n")
    else:
        sys.stdout.write("⚠️  PyTorch: CUDA NOT available.\n")

    # Important: Check TensorFlow GPU availability
    import tensorflow as tf
    try:
        # Prevent TF from taking all VRAM
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            sys.stdout.write(f"✅ TensorFlow GPU(s) found: {len(tf_gpus)}\n")
            for gpu in tf_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            sys.stdout.write("❌ TensorFlow: GPU NOT FOUND. Inference will be slow (CPU).\n")
    except Exception as tfe:
        sys.stdout.write(f"⚠️  TensorFlow init warning: {tfe}\n")
    
    sys.stdout.flush()
except Exception as e:
    sys.stdout.write(f"⚠️  Early Environment Setup warning: {e}\n")
    sys.stdout.flush()

# Set TF specific flags
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit'

# Try to limit TF VRAM early
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Set limit to 60% of GPU to avoid competition with Torch/OS
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.getenv('TF_GPU_MEMORY_LIMIT', '3500')))]
        )
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config.settings import (
    DATABASE_URL, IDENTITY_STABILITY_THRESHOLD, MIN_BUFFER_FOR_RECOG, 
    RECOGNITION_MODELS, DEBUG_MODE, DETECTION_SKIP_FRAMES, RECOGNITION_INTERVAL_FRAMES,
    BATCH_PROCESSING, BATCH_SIZE, CLEAR_CACHE_INTERVAL, USE_GPU, DETECTION_CONFIDENCE,
    CAMERA_INDEX, CAMERA_AUTO_SELECT
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
from detection.async_detector import AsyncDetector
from recognition.async_recognizer import AsyncRecognizer
from core.streaming import FrameBridge

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
    parser = argparse.ArgumentParser(description="EMSVIA Recognition Engine")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--camera", type=int, help="Camera index to use (overrides settings)")
    parser.add_argument("--role", type=str, choices=["entry", "exit"], default="entry", help="Engine role: entry or exit")
    parser.add_argument("--bridge", type=str, help="Unique bridge name for shared memory")
    args = parser.parse_args()
    
    # Calculate unique bridge name if not provided
    bridge_name = args.bridge if args.bridge else f"emsvia_cam_{args.camera if args.camera is not None else 0}"
    
    logger.info("="*60)
    logger.info("🚀 Starting Optimized Industry-Grade Attendance System")
    logger.info("="*60)
    
    # 0. Initialize GPU Manager
    gpu_mgr = GPUModelManager()
    if not gpu_mgr.is_gpu_ready():
        logger.warning("⚠️  GPU not detected! Performance will be affected.")
    else:
        logger.info(f"✅ GPU Ready: {gpu_mgr.get_memory_summary()}")

    # 1. Initialize Models
    logger.info("📦 Loading models...")
    detector = FaceDetector(min_confidence=DETECTION_CONFIDENCE)
    recognizer = FaceRecognizer()
    
    # 2. Initialize Database and Embeddings
    db_manager = AttendanceDB(DATABASE_URL)
    em_manager = EmbeddingsManager(db_manager)
    session = db_manager.get_session()
    em_manager.load_embeddings(session)
    students = db_manager.get_all_students(session)
    student_names = {s.id: f"{s.first_name} {s.last_name}" for s in students}
    session.close()
    logger.info(f"👨‍🎓 Loaded {len(student_names)} students from database")
    
    # 3. Async detection (runs in background thread for smooth video)
    async_detector = AsyncDetector(detector)
    async_detector.start()
    logger.info("🔄 Async detection thread started")
    
    # 3b. Async recognition (runs in background thread, no UI blocking)
    async_recognizer = AsyncRecognizer(detector, recognizer, em_manager)
    async_recognizer.start()
    logger.info("🔄 Async recognition thread started")
    
    # 4. Core Handlers
    cam_source = args.camera if args.camera is not None else CAMERA_INDEX
    camera = CameraHandler(
        source=cam_source, 
        auto_select=CAMERA_AUTO_SELECT if args.camera is None else False,
        interactive=not args.headless
    ).start()
    attend_mgr = AttendanceManager(db_manager)
    unknown_mgr = UnknownFaceHandler(db_manager)
    tracker = FaceTracker()
    bridge = FrameBridge(bridge_name)
    
    # 5. Processing Settings
    detection_skip = DETECTION_SKIP_FRAMES
    recognition_interval = RECOGNITION_INTERVAL_FRAMES
    logger.info(f"⚙️  Detection Skip: {detection_skip} frames")
    logger.info(f"⚙️  Recognition Interval: {recognition_interval} frames")
    
    frame_idx = 0
    fps = 0.0
    fps_history = []
    last_gpu_log_time = time.time()
    
    logger.info("🎬 Processing loop started. Press 'q' to quit.")
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = camera.read()
            if not ret or frame is None: 
                time.sleep(0.005)
                continue
            
            frame_idx += 1
            display_frame = camera.preprocess_frame(frame)
            session = db_manager.get_session()
            
            t_det = 0
            t_rec = 0
            
            try:
                # --- ASYNC DETECTION & TRACKING ---
                # Submit frame to background thread (non-blocking)
                # Returns latest available detections (may be from previous frame)
                t_det_start = time.time()
                faces = async_detector.detect_async(frame)
                t_det = (time.time() - t_det_start) * 1000
                
                # Tracker smooths between detection updates
                tracked_faces = tracker.update(faces)
                
                active_track_ids = {f['track_id'] for f in tracked_faces}
                
                # --- PROCESS TRACKS ---
                for face in tracked_faces:
                    x, y, w, h = face['box']
                    tid = face['track_id']
                    track_data = tracker.tracks.get(tid)
                    if not track_data: continue
                    
                    track_data['frame_count'] += 1
                    
                    # UI Mirroring logic
                    img_w_disp = display_frame.shape[1]
                    x_disp = img_w_disp - (x + w)
                    color = (255, 128, 0) # Orange for tracking
                    label = f"Scanning... (T:{tid})"
                    
                    # --- RECOGNITION (Async, non-blocking) ---
                    stable_id = track_data['current_identity']
                    needs_recognition = False
                    
                    if stable_id is None:
                        if track_data['frame_count'] % 5 == 0: needs_recognition = True
                    else:
                        if track_data['frame_count'] % recognition_interval == 0: needs_recognition = True

                    # Submit recognition job to background thread (non-blocking)
                    if needs_recognition:
                        async_recognizer.submit(tid, frame, [x, y, w, h])
                    
                    # Poll for recognition results (non-blocking)
                    rec_result = async_recognizer.get_result(tid)
                    if rec_result is not None:
                        track_data['last_match'] = rec_result
                        new_id = rec_result["student_id"] if rec_result["match_found"] else None
                        
                        if new_id:
                            if new_id == track_data['current_identity']:
                                track_data['identity_stability_counter'] = 0
                            else:
                                track_data['identity_stability_counter'] += 1
                                if track_data['identity_stability_counter'] >= IDENTITY_STABILITY_THRESHOLD:
                                    track_data['current_identity'] = new_id
                                    track_data['identity_stability_counter'] = 0
                                    logger.info(f"✨ Track {tid} identified as: {student_names.get(new_id, new_id)}")

                    # --- MARK ATTENDANCE & DRAW UI ---
                    stable_id = track_data['current_identity']
                    if stable_id:
                        name = student_names.get(stable_id, f"ID: {stable_id}")
                        label = f"{name}"
                        color = (0, 255, 0) # Green
                        attend_mgr.process_recognition(session, stable_id, track_id=tid, confidence=1.0, entry_type=args.role)
                    elif track_data.get('last_match') and not track_data['last_match']['match_found']:
                        label = f"Unknown (T:{tid})"
                        color = (0, 0, 255) # Red
                    
                    cv2.rectangle(display_frame, (x_disp, y), (x_disp + w, y + h), color, 2)
                    cv2.putText(display_frame, label, (x_disp, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                attend_mgr.clean_stale_tracks(active_track_ids)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
            finally:
                session.close()

            # --- BROADCAST & DISPLAY ---
            # Push processed frame to shared memory for API/Dashboard
            bridge.push(display_frame, meta={
                'fps': round(fps, 1),
                'gpu': gpu_mgr.get_memory_summary() if gpu_mgr.is_gpu_ready() else "CPU Mode",
                'active_tracks': len(active_track_ids),
                'camera_id': cam_source,
                'role': args.role
            })

            if not args.headless:
                cv2.imshow("Industry-Grade Attendance System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # FPS Calculation
            loop_time = time.time() - loop_start
            fps_history.append(loop_time)
            if len(fps_history) > 30: fps_history.pop(0)
            fps = 1.0 / (sum(fps_history) / len(fps_history)) if fps_history else 0
            
            # Periodic GPU stats log (every 30 seconds)
            if gpu_mgr.is_gpu_ready() and time.time() - last_gpu_log_time > 30:
                gpu_mgr.auto_clear_cache()
                logger.info(f"📊 Stats - FPS: {fps:.1f} | Det: {t_det:.1f}ms | Rec: {t_rec:.1f}ms | GPU: {gpu_mgr.get_memory_summary()}")
                last_gpu_log_time = time.time()

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
    finally:
        async_detector.stop()
        async_recognizer.stop()
        camera.stop()
        bridge.clear()
        if not args.headless:
            cv2.destroyAllWindows()
        logger.info("🧹 Cleanup complete.")

if __name__ == "__main__":
    main()

