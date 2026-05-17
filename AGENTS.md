# AGENTS.md — EMSVIA Face Recognition Attendance System

## Project Overview
Real-time face recognition attendance system. Python 3.12, OpenCV, DeepFace, TensorFlow + PyTorch (CUDA 12.1). Runs on Linux/WSL2 with NVIDIA GPU.

## Critical Setup
- **Virtual env**: `venv/` (Python 3.12). Always activate before any command.
- **Env file**: `cp .env.example .env` — required for config to load via `python-dotenv`.
- **GPU libs**: `libs/cudnn8/` contains side-loaded cuDNN 8 for TensorFlow compatibility. This dir is `.gitignore`d — if missing, GPU init in `main_gpu.py` will fail silently.
- **Verified deps**: `requirements_gpu_verified.txt` is a known-good frozen set; `requirements.txt` is the working copy.
- **No package install**: The project uses `sys.path.append(BASE_DIR)` everywhere. Never `pip install -e .`.

## Entry Points & How to Run

| Component | Command | Notes |
|---|---|---|
| GPU recognition engine | `python main_gpu.py` | Primary entry point. Supports `--headless`, `--camera <idx>`, `--role entry\|exit`, `--bridge <name>` |
| CPU recognition engine | `python main.py` | Fallback, no async threads |
| FastAPI backend | `python -m api.main` or `uvicorn api.main:app --reload` | Port 8000. Spawns engine subprocesses via `/api/engine/start` |
| Streamlit dashboard | `streamlit run ui/admin_dashboard.py` | Admin UI for students, attendance, unknown face review |
| React frontend | `cd frontend && npm run dev` | Vite + React 19 + Tailwind 4. Connects to API on port 8000 |
| Camera test | `python scripts/test_camera.py` | Verify hardware before running engine |
| GPU check | `python scripts/check_gpu.py` | Verify CUDA availability |

## Architecture

```
main_gpu.py ──(spawns)──> engine subprocess (headless, per-camera)
                              │
                              │ FrameBridge (/dev/shm shared memory)
                              ▼
api/main.py (FastAPI:8000) ── reads bridge ──> MJPEG stream + WS stream ──> React frontend
```

- **Recognition engines** run as **separate OS processes** (via `subprocess.Popen` with `os.setsid`). The API manages their lifecycle. Killing the API should kill engines, but stale engines can hold `/dev/video*` — use `pkill -f main_gpu.py` if needed.
- **FrameBridge** (`core/streaming.py`) uses `/dev/shm` (POSIX shared memory) for zero-copy frame sharing between engine and API. **Linux only**. Double-buffered (A/B `.jpg` files + `.json` metadata heartbeat).
- **Async detection + recognition**: `detection/async_detector.py` and `recognition/async_recognizer.py` run background threads so the main loop never blocks on model inference.
- **Face tracker** (`core/tracker.py`): IOU-based tracking with identity stability counter (needs 3 consistent matches before switching identity).

## Directory Boundaries

| Dir | Purpose |
|---|---|
| `config/` | Settings, env loading, DB config |
| `database/` | SQLAlchemy models + CRUD. `attendance.db` is the SQLite file |
| `models/` | FaceDetector, FaceRecognizer, EmbeddingsManager, GPUModelManager |
| `core/` | CameraHandler, AttendanceManager, UnknownFaceHandler, FaceTracker, FrameBridge |
| `detection/` | AsyncDetector, RetinaFace ONNX, ensemble, tiling |
| `recognition/` | AsyncRecognizer, batch recognizer |
| `preprocessing/` | CLAHE, exposure correction (Zero-DCE), pipeline |
| `enhancement/` | Super-resolution (FSRCNN/RealESRGAN) |
| `api/` | FastAPI app, routes (students, attendance), schemas |
| `ui/` | Streamlit admin dashboard |
| `frontend/` | React + Vite + Tailwind web dashboard |
| `scripts/` | One-off utilities (registration, embedding generation, diagnostics) |
| `data/` | Runtime data: `student_images/`, `embeddings/`, `unknown_faces/` |
| `libs/` | Side-loaded cuDNN 8 (gitignored) |

## Database Schema (SQLite)

### `students`
- `id` (int PK), `student_id` (string unique), `first_name`, `last_name`, `email` (unique), `phone`, `enrollment_date`, `is_active`

### `face_embeddings`
- `id` (int PK), `student_id` (FK → students), `embedding_vector` (JSON text), `model_name`, `image_path`, `created_at`

### `attendance_records`
- `id` (int PK), `student_id` (FK → students), `timestamp`, `confidence_score`, `camera_id`, `image_path`, `status` (present/late/excused/absent/departed), `entry_type` (entry/exit)

### `unknown_faces`
- `id` (int PK), `timestamp`, `image_path`, `reviewed` (bool), `confidence`

## API Endpoints (FastAPI :8000)

### Students (`/api/students`)
- `GET /` — List all students (`?active_only=true`)
- `GET /{student_id_str}` — Get student by string ID
- `POST /` — Enroll student (multipart form: `student_id`, `first_name`, `last_name`, `files[]`). Saves photos to `data/student_images/{student_id}/`, auto-triggers embedding generation.

### Attendance (`/api/attendance`)
- `GET /recent` — Last N records (`?limit=10`)
- `GET /records` — Filtered history (`?student_id=`, `?start_date=`, `?end_date=`)
- `POST /mark` — Manual attendance mark
- `PATCH /{record_id}` — Update record status
- `DELETE /{record_id}` — Delete record

### Engine Management (`/api/engine`)
- `GET /status` — Check running engines (scans bridges for cams 0–9)
- `POST /start` — Launch engine (`?cam_id=0&role=entry`). Spawns `main_gpu.py --headless` via `subprocess.Popen`.
- `POST /stop` — Kill engine by `cam_id`. Kills process group + clears bridge.

### Streams (`/api/cameras`)
- `GET /stream` — MJPEG feed (`?cam_id=0`). Reads from FrameBridge, falls back to raw OpenCV if engine offline.
- `GET /detect` — Scan for working cameras (indices 0–9)
- `WS /ws/cameras/stream` — WebSocket binary JPEG stream (`?cam_id=0&fps=15`). No raw fallback.

### Other
- `GET /api/stats` — System stats (students, attendance today, unknown pending, GPU status)
- `GET /api/health` — GPU memory stats + version info
- `GET /api/system/logs` — Last 50 lines of `attendance.log`
- `GET /api/students/presence` — Daily presence analysis (absent/in_school/present/under_time based on 90min threshold)
- `GET /api/unknown/pending` — Unreviewed unknown faces
- `PATCH /api/unknown/{id}/review` — Mark as reviewed
- `DELETE /api/unknown/{id}` — Delete record
- `POST /api/embeddings/rebuild` — Trigger `scripts/generate_embeddings.py` in background subprocess

## Recognition Pipeline (GPU Engine)

```
CameraHandler (threaded capture)
  → AsyncDetector (background thread, non-blocking)
    → FaceDetector.detect_faces() → IOU tracker
      → AsyncRecognizer (background thread, non-blocking)
        → extract_face + preprocess_frame + quality_check
        → generate_embeddings (DeepFace)
        → find_best_match (consensus voting)
      → AttendanceManager (evidence accumulation + cooldown)
  → FrameBridge.push() → /dev/shm shared memory
```

### Detection

**Architecture**: `FaceDetector` (`models/face_detector.py`) orchestrates detection with support for single backend, ensemble fusion, and tiled detection.

**Backends (configurable via env vars)**:
- **RetinaFace ONNX** (`detection/retinaface_onnx.py`): Primary detector on GPU. Uses `~/.deepface/weights/retinaface.onnx` with ONNX Runtime (`CUDAExecutionProvider`). Preprocessing: resize to 640×640, subtract mean `[104, 117, 123]`, NCHW transpose.
- **DeepFace RetinaFace**: Fallback when ONNX model unavailable. Uses `DeepFace.extract_faces(detector_backend="retinaface")`.
- **MTCNN**: Alternative DeepFace backend (`DETECTOR_MTCNN=True`).
- **OpenCV Haar Cascade**: Fast fallback (`haarcascade_frontalface_default.xml`), no confidence scores (always 1.0).

**Ensemble Fusion** (`detection/ensemble_detection.py`):
- Runs all active detectors on the same frame, collects all detections
- Applies NMS across all detections with `NMS_IOU_THRESHOLD=0.4` (`detection/tiling.py:apply_nms`)
- Detections agreed on by ≥2 models get confidence boosted by 1.1x and flagged with `agreement_score`
- Env: `USE_ENSEMBLE=True` (default), `ENSEMBLE_DETECTORS` (from `DETECTOR_RETINAFACE`, `DETECTOR_MTCNN`)

**Tiled Detection** (`detection/tiling.py`):
- Splits frame into overlapping tiles (`TILE_SIZE=1080×1080`, `TILE_OVERLAP=0.2`)
- Each tile is independently detected, then boxes mapped back to original coordinates via `map_to_original()`
- NMS applied to remove duplicates across tile boundaries
- Env: `USE_TILING=False` (default, too slow for real-time), `MIN_FACE_SIZE_DETECTION=20`
- Auto-disabled on CPU for performance

**Auto-scaling**: When `GPUModelManager.is_gpu_ready()` returns false, tiling and ensemble are forcibly disabled, and only the first active detector runs.

**Output format**: Each detection is `{"box": [x, y, w, h], "confidence": float, "model": str}` — box is pixel coords in original frame (or tile origin if tiled).

### Recognition

**Architecture**: `FaceRecognizer` (`models/face_recognizer.py`) generates embeddings via DeepFace and matches using consensus voting.

**Models & Thresholds** (cosine distance, lower = better match; configurable via `.env`):

| Model | Default `.env` | Notes |
|---|---|---|
| ArcFace | 0.32 (`ARCFACE_THRESHOLD`) | Stricter, default on GPU |
| Facenet512 | 0.28 (`FACENET512_THRESHOLD`) | Balanced, default on CPU |
| VGG-Face | 0.30 (`VGGFACE_THRESHOLD`) | Legacy, highest false-positive rate — keep disabled |
| Facenet | 0.40 (hardcoded) | Smaller variant |

Env: `RECOGNIZER_ARCFACE=True` (default), `RECOGNIZER_FACENET512=False`, `RECOGNIZER_VGGFACE=False`

**Embedding Generation** (`generate_embeddings`):
- Calls `DeepFace.represent(img_path=image, model_name=model, detector_backend="skip", enforce_detection=False)`
- `detector_backend="skip"` because faces are already cropped by `FaceDetector.extract_face()`
- Explicit L2 normalization: `embedding = embedding / np.linalg.norm(embedding)`
- Returns `Dict[str, np.ndarray]` keyed by model name

**Consensus Voting** (`find_best_match`):
- For each active model, finds the best-matching student using per-model threshold
- **Per-student consensus rule**: A student only qualifies as a candidate if ≥`MIN_VECTOR_PASS_RATIO` (default 25%) of their stored embedding vectors pass the threshold (minimum 2 vectors). This filters out "bad apple" embeddings that match everyone
- Winner per model = student with highest match ratio (passing_vectors / total_vectors), tie-broken by minimum cosine distance
- **Final majority check**: Winning student must be voted by ≥`ENSEMBLE_VOTING_THRESHOLD` (60%) of active models
- **Confidence floor**: Even if a student wins the vote, the match is rejected if final confidence < `RECOGNITION_CONFIDENCE_FLOOR` (default 0.15). Typical real-camera confidence ranges 0.15–0.40 — don't set this too high or legitimate matches will be rejected.
- Confidence = `max(0.0, 1.0 - (avg_distance / avg_threshold))` where `avg_threshold` is mean of all active model thresholds
- Output: `{"match_found": bool, "student_id": int|None, "confidence": float, "vote_ratio": float, "best_distance": float, "model_results": dict}`

**Embedding Storage**:
- Vectors stored as JSON text in `face_embeddings.embedding_vector` column
- Each vector tracks `model_name` and `image_path` for incremental rebuilds
- `EmbeddingsManager` loads all into memory as `Dict[int, Dict[str, List[np.ndarray]]]` (student_id → model_name → [vectors])
- Cached in `data/embeddings/cache.pkl` for fast startup; falls back to DB if cache missing or `force_refresh=True`

### Preprocessing

**Pipeline** (`preprocessing/pipeline.py`): Two-stage, applied to every face crop before embedding generation in BOTH enrollment and live recognition:

1. **Exposure Normalization** (`preprocessing/exposure.py`):
   - Checks if mean grayscale intensity < `DARK_THRESHOLD` (default 50)
   - If dark, applies gamma correction: `adjusted = ((img/255)^(1/gamma)) * 255`
   - Default `GAMMA_CORRECTION=2.2` (brightens dark images)
   - Uses lookup table (`cv2.LUT`) for speed
   - Skipped entirely if image is already bright enough

2. **CLAHE Contrast Boost** (`preprocessing/clahe.py`):
   - Converts BGR → LAB color space
   - Applies CLAHE only to Luminance (L) channel to avoid color distortion
   - `CLAHE_CLIP_LIMIT=2.0`, `CLAHE_TILE_GRID_SIZE=(8, 8)`
   - Merges back and converts LAB → BGR
   - Always applied (not conditional on darkness)

**Zero-DCE**: `ENABLE_ZERO_DCE=True` in settings but not wired into the pipeline yet (placeholder).

**Critical**: The same pipeline must be used for both enrollment (`scripts/generate_embeddings.py:preprocess_frame()`) and live recognition (`recognition/async_recognizer.py:preprocess_frame()`). Mismatch causes recognition failures.

**Super-Resolution** (`enhancement/super_resolution.py`):
- Uses OpenCV DNN SuperRes (`cv2.dnn_superres.DnnSuperResImpl_create()`)
- Models: FSRCNN (default) or EDSR, scale 4x
- Weights expected at `data/models/{model}_x{scale}.pb` (gitignored, must be downloaded separately)
- Only triggers if face crop is < `SR_MIN_SIZE=64` pixels in width or height
- Applied in `FaceDetector.extract_face()` AFTER padding but BEFORE resize to 160×160
- Only runs when GPU is available (`gpu_mgr.is_gpu_ready()`)
- Env: `USE_SUPER_RESOLUTION=False` (default, disabled for performance)

### Quality Gate

**Checks** (`FaceDetector.check_image_quality`):

| Check | Method | Threshold | Failure |
|---|---|---|---|
| Pose angle | Landmark asymmetry (eyes/nose) or crop aspect ratio | ≤ `MAX_POSE_ANGLE` (30°) | `is_extreme_pose=True` |
| Blur | Laplacian variance on grayscale | ≥ 40 | `is_blurry=True` |
| Too dark | Mean HSV V channel | ≥ 30 | `is_too_dark=True` |
| Too bright | Mean HSV V channel | ≤ 235 | `is_too_bright=True` |

- All checks must pass for `quality["passed"] = True`
- If any check fails, the face crop is skipped (no embedding generated, no recognition attempted)
- Pose rejection uses RetinaFace landmarks when available (left/right eye + nose), falls back to crop aspect ratio heuristic
- Applied in two places:
  1. **Live recognition** (`async_recognizer.py`): After preprocessing, before `generate_embeddings()`
  2. **Enrollment** (`generate_embeddings.py`): Not explicitly checked — enrollment assumes input images are pre-vetted

**Enrollment Validation** (`FaceDetector.validate_for_enrollment`):
- Ensures exactly ONE face in frame
- Runs quality check on the extracted face crop
- Returns `{"passed": bool, "message": str, "face_box": list|None, "face_crop": ndarray|None}`
- Not currently called by the API enrollment endpoint (saves photos directly, relies on `generate_embeddings.py` to process)

### Attendance Logic
- Evidence accumulation: track must match same student `ATTENDANCE_MIN_CONSISTENCY` (default 5) times before marking
- Cooldown: `ATTENDANCE_COOLDOWN` (default 3600s) between marks per student
- Entry/exit: engine role (`--role entry|exit`) determines `entry_type` on attendance records
- **Tracker hardening**: Identity switch requires `IDENTITY_STABILITY_THRESHOLD` (default 3) consecutive matches AND rolling average confidence ≥ 0.5

## Testing
- **Framework**: pytest + httpx (for API). No pytest config file — uses defaults.
- **Run all**: `python -m pytest tests/`
- **Run single**: `python -m pytest tests/test_api.py -v`
- **Test DB**: API tests use `test_api.db` (file-based SQLite, cleaned up after). Not in-memory for cross-thread reliability.
- **No conftest.py**: Each test file sets up its own DB override.
- Test files in `tests/` include diagnostic scripts (`check_distances.py`, `diagnose_cache.py`, `verify_fix.py`) — these are NOT tests, they are one-off analysis scripts.

## Key Scripts
- `scripts/generate_embeddings.py` — Rebuild embeddings from `data/student_images/`. Supports `--full` flag to reprocess all images (default: incremental, skips already-processed). Also triggerable via `POST /api/embeddings/rebuild`.
- `scripts/setup_database.py` — Initialize DB schema + seed 3 test students.
- `scripts/register_dataset_students.py` — Bulk register students from dataset.
- `scripts/collect_student_data.py` — Capture student face images from camera.
- `scripts/reset_system.py` — Wipe DB and all data directories. Use `--force` to skip confirmation. **Destructive**.
- `migrate.py` — One-off migration: adds `entry_type` column to `attendance_records`.

## GPU & CUDA Quirks
- `main_gpu.py` does extensive env setup **at module level** before any imports: sets `LD_LIBRARY_PATH`, pre-loads cuDNN 8 libs via `ctypes.CDLL(..., RTLD_GLOBAL)`, configures TensorFlow memory growth, and caps TF VRAM (configurable via `TF_GPU_MEMORY_LIMIT`, default 3500MB).
- **Order matters**: GPU env vars must be set before `import tensorflow` or `import torch`. The early setup block at the top of `main_gpu.py` handles this.
- `TF_USE_LEGACY_KERAS=1` and `KERAS_BACKEND=tensorflow` are forced for DeepFace compatibility.
- PyTorch uses CUDA 12.1 (`torch==2.5.1+cu121`), TensorFlow uses cuDNN 8 side-loaded in `libs/cudnn8/`. Both frameworks share the same GPU — TF is limited to avoid OOM.
- `GPUModelManager` is a singleton. Enables TF32 matmul, cuDNN benchmarking, mixed precision (FP16), and auto-cache clearing every `CLEAR_CACHE_INTERVAL` frames.

## Frontend (React + Vite)
- Stack: React 19, Tailwind 4, Vite 8, axios, recharts, framer-motion, lucide-react
- Pages: Overview, StudentDirectory, AttendanceLogs, FaceReview, LiveCameras, SystemHealth, Settings
- API client: `src/api/client.js` (axios-based, connects to `localhost:8000`)
- Dev: `cd frontend && npm run dev` (no proxy config — uses direct URLs to API)

## Conventions
- **Imports**: Every module that needs project root adds `sys.path.append(str(BASE_DIR))`. No `__init__.py` package exports — flat imports like `from config.settings import ...`.
- **Database sessions**: Always `session.close()` in `finally` blocks. The CRUD layer (`database/crud.py`) wraps SQLAlchemy with `AttendanceDB` helper. API uses `get_db()` dependency with `try/finally`.
- **Logging**: Standard `logging` module. Logs go to `logs/attendance.log` and per-engine `logs/engine_cam_<N>.log`.
- **Config**: All settings from `config/settings.py` via `os.getenv()` with defaults. `DEBUG_MODE` defaults to `True`.
- **Debug logging**: Several files write to `.cursor/debug.log` via `_debug_log()` helper with JSON entries containing hypothesis IDs.

## Gotchas
- **Engine process groups**: Engines spawn with `preexec_fn=os.setsid` — always kill the **process group**, not just the PID. The API does this via `os.killpg(os.getpgid(pid), signal.SIGTERM)`.
- **Shared memory cleanup**: `/dev/shm/emsvia_cam_*.a.jpg`, `*.b.jpg`, `*.json` persist if the engine crashes. Call `FrameBridge.clear()` before restarting.
- **Camera device conflicts**: Engine and API must not both open the same `/dev/video*`. API falls back to raw camera only when bridge is stale (>3s heartbeat). On Linux, the API checks `/dev/video{N}` existence before opening.
- **Embedding cache**: `EmbeddingsManager` caches in memory AND `data/embeddings/cache.pkl`. After adding new student images, call `POST /api/embeddings/rebuild` or restart the engine.
- **Student image directory structure**: `data/student_images/{student_id}/` — one subdirectory per student, containing `.jpg`/`.png` face images. The student ID in the directory name must match the `student_id` string in the database.
- **Enrollment auto-embeddings**: `POST /api/students/` automatically spawns `generate_embeddings.py` as a background subprocess after saving photos.
- **Camera auto-flip**: `CameraHandler.preprocess_frame()` flips horizontally by default (selfie mode). The engine compensates by mirroring x-coordinates for display: `x_disp = img_w - (x + w)`.
- **`.env` is gitignored**: Must manually copy from `.env.example`. Without it, `python-dotenv` loads nothing and settings fall back to defaults.
