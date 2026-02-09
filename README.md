# Face Recognition Attendance System

A real-time face recognition system for automatic student attendance marking using classroom camera feeds. Built with Python, OpenCV, DeepFace, and TensorFlow.

## Features

- **GPU Accelerated Pipeline**: Uses NVIDIA CUDA for high-performance detection and recognition
- **Real-time Face Detection**: Uses MTCNN & RetinaFace ensemble for superior accuracy
- **Face Recognition**: DeepFace with multi-model ensemble (Facenet512, ArcFace, VGG-Face)
- **Super-Resolution**: Enhancement of small faces using deep learning models
- **Automatic Attendance**: Mark attendance automatically with deduplication

## System Requirements

### Hardware
- Camera (USB webcam or built-in laptop camera)
- Minimum 8GB RAM
- CPU: Intel i5 / AMD Ryzen 5 or better
- GPU: **NVIDIA GeForce GTX 1050 (Verified & Enabled)**

### Software
- Python 3.9 or higher
- Windows, Linux, or macOS
- WSL2 (for Windows users)

## Installation

### 1. Clone or Download the Project

```bash
cd attendance-face-recognition
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your settings (use notepad, nano, or any text editor)
```

### 5. Test Camera

```bash
python scripts/test_camera.py
```

This will:
- List available cameras
- Display live camera feed
- Test basic face detection
- Verify your camera is working correctly

## Quick Start

### Phase 1 Complete ✅

You have successfully set up the project foundation! 

**What's been created:**
- ✅ Complete folder structure
- ✅ Core dependencies (`requirements.txt`)
- ✅ Configuration files (`.env.example`, `settings.py`)
- ✅ Utility functions
- ✅ Camera test script

**Next Steps:**
1. Run the camera test to verify your hardware
2. Proceed to Phase 2: Database Layer
3. Then Phase 3: Face Recognition Components

## Project Structure

```
attendance-face-recognition/
├── config/              # Configuration files
├── data/               # Data storage
│   ├── student_images/ # Student photos
│   ├── embeddings/     # Face embeddings
│   └── unknown_faces/  # Unrecognized faces
├── database/           # Database models and operations
├── models/             # Face detection & recognition
├── core/               # Core business logic
├── api/                # FastAPI application
├── ui/                 # Streamlit dashboard
├── scripts/            # Utility scripts
├── tests/              # Test suite
└── logs/               # Application logs
```

## Configuration

Key settings in `.env`:

- `CAMERA_INDEX`: Camera device (default: 0)
- `DETECTION_CONFIDENCE`: Face detection threshold (0.9)
- `RECOGNITION_THRESHOLD`: Face recognition threshold (0.6)
- `ATTENDANCE_COOLDOWN`: Prevent duplicate marking (3600 seconds)

## Development Status

- ✅ Phase 1: Project Setup & Foundation (COMPLETE)
- ⏳ Phase 2: Database Layer (Next)
- ⏳ Phase 3: Core Face Recognition
- ⏳ Phase 4: Data Collection
- ⏳ Phase 5: Attendance Management
- ⏳ Phases 6-12: Advanced Features

## Troubleshooting

### Camera not detected
- Ensure camera is connected
- Check camera permissions
- Close other applications using the camera
- Try different camera index (0, 1, 2...)

### Import errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.9+)

## License

This project is for educational purposes.

## Support

For issues and questions, please check the troubleshooting section or review the code documentation.
