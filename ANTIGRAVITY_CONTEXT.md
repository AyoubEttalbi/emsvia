# Antigravity EMSVIA Project Context

## 1. Project Identity & Goal
- **Project Name:** EMSVIA (Face Recognition Attendance System)
- **Goal:** Industrial-grade, real-time classroom attendance marking using face recognition.
- **Key Constraints:** 40+ FPS on consumer hardware (GTX 1050), zero false positives, handle variable lighting/distances.

## 2. Core Architecture & Stack
- **Languages/Frameworks:** Python 3.12+, OpenCV, DeepFace, TensorFlow, PyTorch, SQLite (SQLAlchemy), FastAPI (planned), Streamlit (planned).
- **Concurrency:** Asynchronous, multi-threaded design. Main loop (UI) runs at 40-50 FPS while AI workers (detection/recognition) run asynchronously in the background.
- **Event-Based Execution:** Recognition runs only every `RECOGNITION_INTERVAL` frames or on new tracks, saving 90% GPU load.
- **Hardware Hack (Version Bridge):** Isolates cuDNN 8 (`/libs/cudnn8`) via `ctypes` pre-loading so TensorFlow (needs cuDNN 8) and PyTorch (needs cuDNN 9) can share the GPU concurrently.

## 3. The "Accuracy Pipeline"
1. **Preprocessing (The Cleaners):**
   - Applies CLAHE (to LAB L-channel) before detection to boost local contrast.
   - Applies Gamma exposure correction if the mean frame intensity < 50.
2. **Multi-Scale Tiled Detection:**
   - Slices 1080p frame into 4 overlapping (20%) tiles + original frame. Runs detection, maps coordinates back, and applies NMS (IoU 0.4). Catches small/distant faces.
3. **Ensemble Detection:**
   - Runs RetinaFace (primary) + SCRFD (small face specialist). Uses NMS and a voting system for confidence.
4. **Selective Super-Resolution:**
   - Checks face size (< 80px) and blur (Laplacian variance < 100). Applies OpenCV DNN FSRCNN (2x or 4x upscale) *before* embedding extraction.
5. **Ensemble Recognition (The Brains):**
   - Uses ArcFace + Facenet512. Attendance is marked only if both models agree.
6. **Temporal Tracking (The Smoothing):**
   - IOU-based face tracking. An ID must be stable over multiple frames (`IDENTITY_STABILITY_THRESHOLD`) before locking.

## 4. Smart Matching Algorithm
- **Replaced absolute distance thresholds** with a **Relative Distance Comparison**.
- Calculates the gap between the *best match* and *second-best match*.
- Confidence = `(second_best - best) / threshold`.
- Requires a `MIN_CONFIDENCE_GAP` (e.g., 0.12 or 12%) and `RECOGNITION_QUALITY_MULTIPLIER` (e.g., 0.85). If the gap is too small, it rejects the face as an ambiguous false positive.

## 5. Directory Structure Reference
- `api/`: FastAPI backend (Routes, Schemas, Dependencies).
- `config/`: Settings (`settings.py`), database configs.
- `core/`: Business logic, trackers (`tracker.py`), attendance logic.
- `detection/`: Tiling, ensemble detection logic.
- `frontend/`: Premium React/Vite dashboard (Tailwind 4, Glassmorphism).
- `models/`: ML wrappers (`face_detector.py`, `face_recognizer.py`, `embeddings_manager.py`, `gpu_manager.py`).
- `preprocessing/`: Lighting fixes (`clahe.py`, `exposure.py`).
- `scripts/`: Enrollment automation, embedding generation, testing.

## 6. Premium Admin Dashboard (Phase 8-13)
The system now features a production-grade administrative command center:
- **📊 Real-Time Analytics**: Live attendance metrics, trend charts, and student directory.
- **🛡️ Security Audit Hub**: Unknown face review system with Review/Dismiss/Enroll workflows.
- **🎥 Surveillance Engine**: Low-latency MJPEG camera streaming with dynamic source detection and manual toggle controls.
- **📉 Hardware & Software Telemetry**: Live GPU VRAM monitoring + Real-time engine log stream (tailing `attendance.log`).
- **👨‍🎓 Student Enrollment**: Premium modal with multi-photo upload support; automatically triggers background embedding vectorization.

## 7. Current Project Status
- **Completed:** 
  - Phases 1-8 (Accuracy, GPU Bridge, Database Core).
  - Phase 9-11 (FastAPI Backend, Surveillance Streaming, Secure Reviews).
  - Phase 12 (Hardware/Software Telemetry, Log Streaming).
  - Phase 13 (Automated AI Pipeline / Rebuild on Enrollment).
- **In Progress / Pending:**
  - Phase 14: AI Precision Tuning (Ensemble Voting, NMS optimization).
  - Phase 15: Face Anti-Spoofing (Liveness detection).
  - Phase 16: Selective Super-Resolution implementation in live pipeline.
  - Phase 17: Enterprise Security (JWT Authentication & RBAC).

## AI Reminders
- **Engine Control**: The Recognition Engine (`main_gpu.py`) must be running separately from the API for full system functionality.
- **Vectorization**: Enrollment *automatically* triggers `scripts/generate_embeddings.py` via background process in the API.
- **Camera Access**: If streaming fails, check if the camera index is locked or if `/dev/video*` permissions are set.
- **HW Telemetry**: GPU stats depend on `pynvml` and `torch.cuda`.
- **Styling**: All dashboard UI uses **Tailwind CSS 4** and Lucide React icons.
