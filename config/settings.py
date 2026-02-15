"""
Configuration settings for Face Recognition Attendance System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STUDENT_IMAGES_DIR = DATA_DIR / "student_images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
UNKNOWN_FACES_DIR = DATA_DIR / "unknown_faces"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, STUDENT_IMAGES_DIR, EMBEDDINGS_DIR, UNKNOWN_FACES_DIR, DATABASE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model settings
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "opencv")
FACE_RECOGNITION_MODEL = os.getenv("FACE_RECOGNITION_MODEL", "Facenet512")
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.9"))
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.6"))
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "cosine")

# Camera settings
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
FPS = int(os.getenv("CAMERA_FPS", "30"))

# Attendance settings
MIN_IMAGES_PER_STUDENT = int(os.getenv("MIN_IMAGES_PER_STUDENT", "10"))
ATTENDANCE_COOLDOWN = int(os.getenv("ATTENDANCE_COOLDOWN", "3600"))  # seconds
SAVE_UNKNOWN_FACES = os.getenv("SAVE_UNKNOWN_FACES", "True").lower() == "true"

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_DIR}/attendance.db")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "attendance.log"

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Accuracy Improvement & Performance Settings ---
# Hardware Configuration
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
DEVICE = os.getenv("DEVICE", "cuda")
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
ALLOW_GROWTH = os.getenv("ALLOW_GROWTH", "True").lower() == "true"
USE_MIXED_PRECISION = os.getenv("USE_MIXED_PRECISION", "True").lower() == "true"

# Model Toggles
DETECTOR_RETINAFACE = os.getenv("DETECTOR_RETINAFACE", "True").lower() == "true"
DETECTOR_MTCNN = os.getenv("DETECTOR_MTCNN", "False").lower() == "true"

RECOGNIZER_ARCFACE = os.getenv("RECOGNIZER_ARCFACE", "True").lower() == "true"
RECOGNIZER_FACENET512 = os.getenv("RECOGNIZER_FACENET512", "False").lower() == "true"
RECOGNIZER_VGGFACE = os.getenv("RECOGNIZER_VGGFACE", "False").lower() == "true"

# Dynamic Model Lists
def get_active_detectors():
    detectors = []
    if DETECTOR_RETINAFACE: detectors.append("retinaface")
    if DETECTOR_MTCNN: detectors.append("mtcnn")
    return detectors if detectors else ["retinaface"] # Default

def get_active_recognizers():
    recognizers = []
    if RECOGNIZER_ARCFACE: recognizers.append("ArcFace")
    if RECOGNIZER_FACENET512: recognizers.append("Facenet512")
    if RECOGNIZER_VGGFACE: recognizers.append("VGG-Face")
    return recognizers if recognizers else ["ArcFace"] # Default

# Tracking & Efficiency
TRACKING_ENABLED = os.getenv("TRACKING_ENABLED", "True").lower() == "true"
TRACKING_SENSITIVITY = float(os.getenv("TRACKING_SENSITIVITY", "0.5"))
DETECTION_SKIP_FRAMES = int(os.getenv("DETECTION_SKIP_FRAMES", "10"))
RECOGNITION_INTERVAL_FRAMES = int(os.getenv("RECOGNITION_INTERVAL_FRAMES", "30"))

# Phase 1: Preprocessing
CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
CLAHE_TILE_GRID_SIZE = (8, 8)
DARK_THRESHOLD = int(os.getenv("DARK_THRESHOLD", "50"))
GAMMA_CORRECTION = float(os.getenv("GAMMA_CORRECTION", "2.2"))
ENABLE_ZERO_DCE = os.getenv("ENABLE_ZERO_DCE", "True").lower() == "true"

# Phase 2: Multi-Scale Detection
USE_TILING = os.getenv("USE_TILING", "False").lower() == "true"
TILE_SIZE = (1080, 1080)
TILE_OVERLAP = float(os.getenv("TILE_OVERLAP", "0.2"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.4"))
MIN_FACE_SIZE_DETECTION = int(os.getenv("MIN_FACE_SIZE_DETECTION", "20"))
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", "1"))
RECOGNITION_INTERVAL = int(os.getenv("RECOGNITION_INTERVAL", "1"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"

# Phase 3: Ensemble Detection
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "True").lower() == "true"
ENSEMBLE_DETECTORS = get_active_detectors()

# Phase 4: Super-Resolution
USE_SUPER_RESOLUTION = os.getenv("USE_SUPER_RESOLUTION", "False").lower() == "true"
SR_MODEL = os.getenv("SR_MODEL", "FSRCNN")
SR_SCALE = int(os.getenv("SR_SCALE", "4"))
SR_MIN_SIZE = int(os.getenv("SR_MIN_SIZE", "64"))

# Phase 5: Ensemble Recognition
USE_RECOGNITION_ENSEMBLE = os.getenv("USE_RECOGNITION_ENSEMBLE", "True").lower() == "true"
RECOGNITION_MODELS = get_active_recognizers()
ENSEMBLE_VOTING_THRESHOLD = float(os.getenv("ENSEMBLE_VOTING_THRESHOLD", "0.5"))

# Phase 6: Temporal Smoothing
TRACKING_MAX_DISAPPEARED = int(os.getenv("TRACKING_MAX_DISAPPEARED", "15"))
ATTENDANCE_MIN_CONSISTENCY = int(os.getenv("ATTENDANCE_MIN_CONSISTENCY", "5"))
TRACKER_IOU_THRESHOLD = float(os.getenv("TRACKER_IOU_THRESHOLD", "0.3"))
IDENTITY_STABILITY_THRESHOLD = int(os.getenv("IDENTITY_STABILITY_THRESHOLD", "3"))
EMBEDDING_BUFFER_SIZE = 10
MIN_BUFFER_FOR_RECOG = 3

# Batch Processing (GPU)
BATCH_PROCESSING = os.getenv("BATCH_PROCESSING", "True").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
AUTO_BATCH_SIZE = os.getenv("AUTO_BATCH_SIZE", "True").lower() == "true"

# GPU Memory Management
CLEAR_CACHE_INTERVAL = int(os.getenv("CLEAR_CACHE_INTERVAL", "100"))

# Async Processing
ASYNC_PREPROCESSING = os.getenv("ASYNC_PREPROCESSING", "False").lower() == "true"
ASYNC_DISPLAY = os.getenv("ASYNC_DISPLAY", "False").lower() == "true"
