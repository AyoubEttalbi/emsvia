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
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "mtcnn")
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
