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
- `config/`: Settings (`settings.py`), database configs.
- `core/`: Business logic, trackers (`tracker.py`), attendance logic.
- `detection/`: Tiling, ensemble detection logic.
- `models/`: ML wrappers (`face_detector.py`, `face_recognizer.py`, `embeddings_manager.py`, `gpu_manager.py`).
- `preprocessing/`: Lighting fixes (`clahe.py`, `exposure.py`).
- `scripts/`: Enrollment, embedding generation, testing.

## 6. Current Project Status
- **Completed:** Phases 1-4 (Foundations, DB, Recognition) and Phase 8 (GPU/Accuracy Integrations).
- **In Progress / Pending:**
  - Phase 5: Advanced attendance logic (manual overrides, stats).
  - Phase 6: Mask/Glasses logic & Periocular recognition.
  - Phase 7: FastAPI endpoints and JWT auth.
  - Phase 8: Streamlit Admin / Analytics Dashboard.

## AI Reminders
- Do NOT alter the async/threaded architecture without considering FPS drops.
- Always check `config/settings.py` for thresholds before hardcoding values.
- Respect the cuDNN 8/9 Version Bridge when modifying deep learning imports.
