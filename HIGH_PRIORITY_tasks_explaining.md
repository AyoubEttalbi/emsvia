# PROMPT FOR CODE AGENT

You are an expert AI engineer tasked with implementing a comprehensive face recognition accuracy improvement system. I will provide you with my current codebase context through the following files: `explaining.md`, `README.md`, and `tasks.md`.

## YOUR MISSION
Implement ALL of the following improvements to maximize face recognition accuracy for a classroom attendance system (40 students max, variable lighting conditions, all distances from camera).

---

## PHASE 1: PREPROCESSING & LIGHTING ENHANCEMENT
**Goal:** Handle dark and variable lighting conditions

### [x] Task 1.1: Implement CLAHE Preprocessing
**Command to Test:**
```bash
./venv/bin/python3 scripts/test_preprocessing_viz.py
```
- [x] Apply Contrast Limited Adaptive Histogram Equalization to every frame BEFORE face detection
- [x] Use OpenCV's `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`
- [x] Convert to LAB color space, apply CLAHE to L channel only, then convert back to BGR
- **Why:** Boosts local contrast, recovers faces from shadows with ~5ms overhead

### [x] Task 1.2: Add Exposure Normalization
- [x] Check if frame is too dark: calculate mean pixel intensity
- [x] If mean < 50 (on 0-255 scale), apply gamma correction: `adjusted = ((img/255) ** (1/2.2)) * 255`
- **Why:** Prevents model confusion during sudden lighting changes

### Task 1.3: Optional - Zero-DCE Integration (if CLAHE insufficient)
- Integrate Zero-DCE or SCI (Self-Calibrated Illumination) model for extreme low-light
- Apply only when mean intensity < 30 (very dark frames)
- **Why:** Deep learning-based enhancement for severe darkness (~50-100ms overhead)

**Deliverable:** Preprocessing pipeline that normalizes lighting before detection

---

## PHASE 2: MULTI-SCALE DETECTION WITH TILING
**Goal:** Detect faces at all distances (small faces in back rows)

### [x] Task 2.1: Implement Image Tiling
**Command to Test:**
```bash
./venv/bin/python3 scripts/test_tiled_detection.py
```
Create a tiling function that:
1. [x] Divides 1080p frame into 4 overlapping tiles (1080x1080 each)
   - Top-left: (0, 0) to (1080, 1080)
   - Top-right: (840, 0) to (1920, 1080)  
   - Bottom-left: (0, 0) to (1080, 1080)
   - Bottom-right: (840, 0) to (1920, 1080)
   - 20% overlap to avoid cutting faces at edges

2. [x] Run face detection on:
   - Full original frame
   - Each of 4 tiles separately

3. [x] Coordinate mapping:
   - For faces found in tiles, convert coordinates back to original frame reference
   - Example: face at (100, 50) in top-right tile → (940, 50) in original

4. [x] Apply Non-Maximum Suppression (NMS) to remove duplicate detections
   - Use IoU threshold of 0.4
   - Keep detection with highest confidence when duplicates overlap

**Why:** Makes distant faces appear 2-4x larger to the detector, improving small face detection by 20-30%

### [x] Task 2.2: Upgrade Detection Model
- [x] Replace MTCNN with **RetinaFace** (primary) or **SCRFD** (alternative)
- [x] Set minimum detection size to 20 pixels
- [x] Configure detection threshold: 0.6 for balance of precision/recall
- **Why:** RetinaFace uses FPN (Feature Pyramid Network) that handles multi-scale faces far better than MTCNN's pyramid approach

**Deliverable:** Tiled multi-scale detection pipeline that catches faces from 20-200+ pixels

---

## PHASE 3: ENSEMBLE DETECTION
**Goal:** Reduce false positives through model agreement

### [x] Task 3.1: Implement Multi-Model Detection
**Command to Test:**
```bash
./venv/bin/python3 scripts/test_ensemble_viz.py
```
Run 2-3 detection models in parallel:
1. [x] **RetinaFace** (primary - balanced)
2. [x] **SCRFD** (small face specialist) - *Note: Used DeepFace backends 'retinaface' and 'mtcnn' for initial ensemble implementation*
3. [ ] **YuNet** (optional - speed/accuracy balance)

### [x] Task 3.2: Detection Fusion Strategy
For each frame:
1. [x] Collect all detections from all models
2. [x] Apply NMS across all detections (IoU threshold 0.4)
3. [x] Assign confidence scores based on model agreement:
   - [x] Detected by 2+ models: confidence bonus
   - [x] Track agreement score in detection metadata

**Why:** Multiple detection models reduce single-model blind spots and false positives by 10-15%

**Deliverable:** Ensemble detection system with confidence scoring

---

## PHASE 4: SUPER-RESOLUTION ENHANCEMENT
**Goal:** Improve quality of small/blurry faces before recognition

### [x] Task 4.1: Implement Selective Super-Resolution
**Command to Test:**
```bash
./venv/bin/python3 scripts/test_sr_viz.py
```
For each detected face:
1. [x] Measure face bounding box size (width × height)
2. [x] Check if blur detection needed: calculate Laplacian variance
   - If variance < 100: face is blurry

3. [x] Apply enhancement if ANY of these conditions:
   - Face width < 80 pixels OR height < 80 pixels
   - Face is blurry (variance < 100)
   - Detection confidence < 0.7 (uncertain detection)

4. [x] Enhancement options (choose ONE to implement):
   - **OpenCV DNN SuperRes** (FSRCNN x4) - *Chosen for speed and ease of integration*

5. [x] Upscale small faces by 2x or 4x depending on original size:
   - <40 pixels: 4x upscale
   - 40-80 pixels: 2x upscale
   - >80 pixels: no upscale

### [x] Task 4.2: Integration Point
- [x] Apply SR AFTER detection, BEFORE embedding extraction
- [x] Cache enhanced faces to avoid re-processing same face across frames

**Why:** Reconstructs facial details lost in small/distant faces, improving embedding quality by 15-25% for faces <60 pixels

**Deliverable:** Selective super-resolution pipeline integrated into recognition flow

---

## PHASE 5: ENSEMBLE RECOGNITION
**Goal:** Maximum recognition accuracy through model voting

### [x] Task 5.1: Implement Multi-Model Recognition
**Command to Test:**
```bash
./venv/bin/python3 scripts/test_recognition_on_folder.py
```
For each enhanced face:
1. [x] Generate embeddings using multiple models:
   - **Facenet512** (Standard)
   - **ArcFace** (High-accuracy variant)

### [x] Task 5.2: Embedding Extraction & Voting
1. [x] Search database for match for EACH model independently
2. [x] Apply Voting Rule:
   - Match is only valid if both models agree on the student ID.
   - If models disagree: mark as "Uncertain" or "Unknown" to avoid false positives.

### [x] Task 5.3: Database Structure Support
1. [x] Store model names in the embedding tables
2. [x] Filter embeddings by model name during retrieval to avoid "mixing" vector

4. Voting logic:
```
   If 3 models agree on same student ID:
       confidence = "VERY HIGH"
       return student_id
   
   Else if 2 models agree on same student ID:
       confidence = "HIGH"  
       return student_id (if both scores > 0.6)
   
   Else (all disagree or tie):
       confidence = "UNCERTAIN"
       return "UNKNOWN" or "NEEDS_MANUAL_REVIEW"
```

5. Threshold logic:
   - Accept if: 2+ models agree AND average similarity > 0.6
   - Flag for review if: models disagree OR average similarity 0.4-0.6
   - Reject if: average similarity < 0.4

**Why:** Ensemble reduces false positives by 10-20% and increases robustness to edge cases

**Deliverable:** Multi-model recognition system with confidence-based voting

---

## PHASE 6: TEMPORAL SMOOTHING
**Goal:** Eliminate single-frame errors through tracking

### [x] Task 6.1: Implement Face Tracking
**Command to Run Main (Tracking + Recognition):**
```bash
./venv/bin/python3 main.py
```
1. [x] Use IOU Tracker to maintain student identity between frames.
2. [x] Each face in the video gets a `Track ID` (e.g., Track #1, Track #2).

### [x] Task 6.2: Embedding Averaging
1. [x] For each Track ID, maintain a history of recent recognition results.

### [x] Task 6.3: Temporal Consistency
1. [x] Minimum consistency rule: student_id = 5 must be recognized in **5 out of 10 consecutive frames** for that Track ID.
2. [x] Attendance is only marked once this consistency threshold is met.an([frame1_emb, frame2_emb, ..., frame10_emb])
   avg_vggface_emb = mean([frame1_emb, frame2_emb, ..., frame10_emb])

3. Perform recognition on AVERAGED embeddings (not single-frame)
4. Update identity only when confidence is high (ensemble agreement + high similarity)

### Task 6.3: Temporal Consistency
- If track switches identity (Student A → Student B), require 3 consecutive frames of agreement before accepting change
- **Why:** Prevents flickering misidentifications from blinks, shadows, temporary occlusions

**Deliverable:** Tracking system that smooths recognition across time

---

## PHASE 7: ENHANCED ENROLLMENT SYSTEM
**Goal:** Build robust student database with variation coverage

### [x] Task 7.1: Multi-Angle Enrollment
**Command to Enroll:**
```bash
./venv/bin/python3 scripts/collect_student_data.py
```
1. [x] Update `collect_student_data.py` to prompt user for:
   - Front-on view
   - 45-degree left/right views
   - Up/down tilt (optional - implemented 3 main angles)

### [x] Task 7.2: Reference Image Enhancement
1. [x] Run the Phase 1 Preprocessing (CLAHE + Exposure) on all enrollment images before saving.
2. [x] This ensures the "known" face and the "live" face look exactly the same to the AI.

### [x] Task 7.3: Matching Strategy Update
1. [x] When a face is detected in the classroom, compared it against ALL enrolled images (front, side, etc.).
2. [x] Success if ANY of the reference angles match with high confidence.g

### Task 7.4: Embedding Database Structure
Store embeddings as:
```python
{
  "student_id": "12345",
  "name": "John Doe",
  "embeddings": {
    "arcface": [emb1, emb2, emb3, ..., emb10],
    "facenet": [emb1, emb2, emb3, ..., emb10],
    "vggface": [emb1, emb2, emb3, ..., emb10]
  },
  "conditions": ["normal_light", "dim_light", "angle_left", ...]
}
```

### Task 7.3: Matching Strategy
During recognition:
- Compare incoming embedding against ALL stored embeddings for a student
- Take the BEST match (highest similarity) across all enrollment conditions
- **Why:** Handles lighting/pose variations that weren't in training data

**Deliverable:** Enrollment system capturing 8-12 embeddings per student across conditions

---

## [x] PHASE 8: COMPLETE PIPELINE INTEGRATION
**Goal:** Final optimization and system hardening

### [x] Task 8.1: End-to-End Testing
1. [x] Verify that ALL 7 phases work together without crashing.
2. [x] Test with multiple people moving at once.

### [x] Task 8.2: Performance Optimization
1. [x] Optimize `main.py` loop:
   - Run **Detection** every 1 frame (cheap).
   - Run **Recognition** only every 10 frames for each tracked face (expensive).
   - Use the "Last Known Identity" for the 9 frames in between.

### [x] Task 8.3: Logging & Debugging
1. [x] Add clear UI indicators for "Tracking" vs "Recognizing".
2. [x] Log performance metrics (FPS) to ensure the system is usable.greement
    ↓
[3] ENHANCEMENT (Selective SR)
    - For each face: check size and blur
    - If small/blurry: apply GFPGAN/Real-ESRGAN
    ↓
[4] RECOGNITION (Ensemble)
    - Extract embeddings: ArcFace + Facenet + VGGFace
    - Compare against database (all enrollment embeddings)
    - Vote on identity
    ↓
[5] TRACKING (Temporal Smoothing)
    - Update track with new embedding
    - Average last 10 frames
    - Re-run recognition on averaged embedding
    ↓
[6] OUTPUT
    - Student ID + Confidence Score
    - Flag for manual review if uncertain
```

### Task 8.2: Performance Optimization
Implement these optimizations:
1. **Frame skipping**: Process every 3-5 frames (not every frame)
2. **Parallel processing**: Run detection and previous frame's recognition in parallel
3. **GPU acceleration**: Use GPU for SR and embedding extraction if available
4. **Caching**: 
   - Cache enhanced faces (SR results)
   - Cache detection results for 2-3 frames if no movement
5. **Smart SR**: Only apply to faces that FAILED recognition on first attempt

### Task 8.3: Logging & Debugging
Add comprehensive logging:
- Per-frame processing time breakdown
- Detection confidence distribution
- Recognition confidence distribution  
- Model agreement statistics
- False positive/negative tracking (if ground truth available)

**Deliverable:** Production-ready integrated system with performance monitoring

---

## IMPLEMENTATION GUIDELINES

### Code Organization
```
project/
├── preprocessing/
│   ├── clahe.py
│   ├── exposure_normalization.py
│   └── zero_dce.py
├── detection/
│   ├── retinaface_detector.py
│   ├── scrfd_detector.py
│   ├── tiling.py
│   └── ensemble_detection.py
├── enhancement/
│   ├── super_resolution.py (GFPGAN/Real-ESRGAN)
│   └── quality_check.py (blur detection)
├── recognition/
│   ├── arcface_model.py
│   ├── facenet_model.py
│   ├── vggface_model.py
│   └── ensemble_recognition.py
├── tracking/
│   ├── face_tracker.py
│   └── temporal_smoother.py
├── enrollment/
│   ├── capture_system.py
│   └── database_manager.py
└── pipeline/
    ├── main_pipeline.py
    └── config.py
```

### Error Handling
- Wrap each model inference in try-except blocks
- Fallback gracefully: if ensemble fails, use single best model
- Log all errors with frame number and timestamp
- Continue processing even if one component fails

### Configuration Management
Create a config file with all tunable parameters:
```yaml
preprocessing:
  clahe_clip_limit: 2.0
  clahe_tile_size: 8
  dark_threshold: 50
  gamma_correction: 2.2

detection:
  use_tiling: true
  tile_overlap: 0.2
  min_face_size: 20
  detection_threshold: 0.6
  nms_iou_threshold: 0.4

enhancement:
  sr_threshold_size: 80
  sr_blur_threshold: 100
  sr_scale_factor: 2
  sr_model: "GFPGAN"  # or "Real-ESRGAN" or "CodeFormer"

recognition:
  ensemble_models: ["arcface", "facenet", "vggface"]
  similarity_threshold: 0.6
  consensus_required: 2  # out of 3 models

tracking:
  temporal_window: 10
  track_timeout: 15
  min_track_confidence: 0.7

performance:
  process_every_n_frames: 3
  use_gpu: true
  parallel_processing: true
```

### Testing Strategy
1. **Unit tests**: Test each component independently
2. **Integration tests**: Test pipeline end-to-end
3. **Performance benchmarks**: Measure FPS and latency
4. **Accuracy tests**: Use labeled test set if available

### Documentation Requirements
For each implemented component, provide:
1. Function-level docstrings explaining inputs/outputs
2. Code comments explaining WHY (not just WHAT)
3. Performance characteristics (latency, memory usage)
4. Example usage
5. Known limitations

---

## SUCCESS CRITERIA
After implementation, the system should achieve:
- ✅ Detect faces as small as 20-30 pixels
- ✅ Maintain >85% accuracy in dim lighting
- ✅ <5% false positive rate
- ✅ <10% false negative rate in normal conditions
- ✅ Process at >2 FPS on CPU, >10 FPS on GPU
- ✅ Graceful degradation when components fail

---

## DELIVERABLES CHECKLIST
- [ ] All code files organized as specified
- [ ] Configuration file with all parameters
- [ ] README with setup instructions
- [ ] Performance benchmark results
- [ ] Example output showing detection/recognition results
- [ ] Logging output showing pipeline stages

---

## IMPORTANT NOTES
1. **Explain as you code**: Add detailed comments explaining the mathematical/algorithmic reasoning
2. **Incremental implementation**: Build and test each phase before moving to next
3. **Performance profiling**: Use time.time() or cProfile to measure each component
4. **Model weights**: Provide links or instructions to download pretrained models
5. **Dependencies**: List all required packages with versions

**Start with PHASE 1 and proceed sequentially. After each phase, verify it works before continuing.**

Do you have access to the files I mentioned (explaining.md, README.md, tasks.md)? If yes, read them first to understand the current system architecture, then begin implementation starting with Phase 1.