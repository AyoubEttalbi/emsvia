# Strategy: Phase 6 Mask & Glasses Handling

## Overview
This document outlines the proposed "Hybrid Async" strategy for handling facial occlusions (masks/glasses) in the EMSVIA system.

## Proposed Strategy: Hybrid Async Pipeline

We will implement a "Detect-Classify-Route" architecture to handle masks without sacrificing the current 40+ FPS performance.

### 1. Mask Detection Layer
- **Approach:** Use a lightweight MobileNetV2-based ONNX model.
- **Function:** Classify face crops into `no_mask`, `mask`, or `improper_mask`.
- **Integration:** Runs as part of the `AsyncDetector` workflow, adding < 5ms overahead.

### 2. Hybrid Recognition Logic
- **If `no_mask`**: Use the standard **Facenet512 + ArcFace** ensemble for maximum accuracy.
- **If `mask`**: Switch to **Periocular-Focused Recognition**. 
  - Instead of a new model, use an **ArcFace** configuration focused on the upper face region (landmarks: bridge of nose to eyebrows). 
  - Apply slightly more lenient distance thresholds to account for reduced feature availability.

### 3. Data Logging
- Update `AttendanceRecord` and `AttendanceManager` to store `mask_status`.
- Enables compliance reporting (e.g., "95% of students wore masks correctly").

## Expected Implementation Steps (Future)
1. Implement `models/mask_detector.py` (ONNX baseline).
2. Refactor `models/face_recognizer.py` to support region-of-interest (ROI) switching.
3. Update `database/models.py` for mask status storage.
4. Update `scripts/collect_student_data.py` for guided masked enrollment.
