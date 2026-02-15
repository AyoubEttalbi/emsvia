import cv2
import numpy as np
from deepface import DeepFace
from typing import List, Tuple, Optional, Dict, Any
import logging
from config.settings import (
    FACE_DETECTION_MODEL, 
    DETECTION_CONFIDENCE, 
    USE_TILING, 
    TILE_SIZE, 
    TILE_OVERLAP, 
    NMS_IOU_THRESHOLD,
    MIN_FACE_SIZE_DETECTION,
    USE_ENSEMBLE,
    ENSEMBLE_DETECTORS,
    USE_GPU, 
    DEVICE
)
from detection.tiling import get_tiles, map_to_original, apply_nms
from detection.ensemble_detection import fuse_detections
from enhancement.super_resolution import FaceEnhancer
from detection.retinaface_onnx import RetinaFaceONNX

# Configure logging
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Handles face detection using multiple backends (RetinaFace, MTCNN, etc.).
    Includes support for image tiling to detect small faces.
    """
    
    def __init__(self, backend: str = None, min_confidence: float = None):
        """
        Initialize the Face Detector.
        """
        from config.settings import get_active_detectors
        self.active_detectors = get_active_detectors()
        self.backend = backend or (self.active_detectors[0] if self.active_detectors else FACE_DETECTION_MODEL)
        self.min_confidence = min_confidence or DETECTION_CONFIDENCE
        self.min_face_size = (MIN_FACE_SIZE_DETECTION, MIN_FACE_SIZE_DETECTION)
        
        # Initialize Enhancer & raw OpenCV Cascade
        self.enhancer = FaceEnhancer()
        self.face_cascade = None
        if "opencv" in self.active_detectors or self.backend == "opencv":
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        # Initialize ONNX detector if GPU is enabled
        self.onnx_detector = None
        if USE_GPU and "retinaface" in self.active_detectors:
            self.onnx_detector = RetinaFaceONNX(device=DEVICE)
            if not self.onnx_detector.is_ready():
                logger.info("Falling back to DeepFace RetinaFace (ONNX not ready)")
        
        logger.info(f"FaceDetector initialized with active detectors: {self.active_detectors}")

    def detect_only_model(self, image: np.ndarray, backend: str) -> List[Dict[str, Any]]:
        """
        Internal method to run a specific detection model.
        """
        # RAW OPENCV PATH (High Speed)
        if backend == "opencv" and self.face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=self.min_face_size
            )
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'box': [int(x), int(y), int(w), int(h)],
                    'confidence': 1.0, # Haar doesn't provide easy confidence
                    'model': 'opencv_raw'
                })
            return detections

        # Use ONNX if available and backend matches
        if backend == "retinaface" and self.onnx_detector and self.onnx_detector.is_ready():
            try:
                # This is faster than DeepFace.extract_faces for batch processing
                detections = self.onnx_detector.detect(image, threshold=self.min_confidence)
                if detections:
                    # Add model name to detections from ONNX
                    for det in detections:
                        det['model'] = 'retinaface_onnx'
                    return detections
            except Exception as e:
                logger.error(f"ONNX detection error, falling back: {e}")

        # Standard DeepFace fallback
        try:
            results = DeepFace.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=False,
                align=True
            )
            
            detections = []
            for res in results:
                if res['confidence'] >= self.min_confidence:
                    detections.append({
                        'box': [res['facial_area']['x'], res['facial_area']['y'], 
                               res['facial_area']['w'], res['facial_area']['h']],
                        'confidence': res['confidence'],
                        'model': backend
                    })
            return detections
        except Exception as e:
            logger.error(f"DeepFace detection error: {e}")
            return []

    def detect_faces(self, image: np.ndarray, use_tiling: bool = None, use_ensemble: bool = None) -> List[Dict[str, Any]]:
        """
        Detect faces in an image with optional tiling and ensemble support.
        """
        if image is None or image.size == 0:
            return []

        from models.gpu_manager import GPUModelManager
        gpu_mgr = GPUModelManager()
        
        should_tile = use_tiling if use_tiling is not None else USE_TILING
        
        # Determine which models to run
        models_to_run = self.active_detectors
        should_ensemble = len(models_to_run) > 1
        
        # AUTO-SCALE: Disable heavy features on CPU
        if not gpu_mgr.is_gpu_ready():
            if should_tile or should_ensemble:
                 should_tile = False
                 should_ensemble = False
                 models_to_run = [models_to_run[0]] if models_to_run else ["opencv"]

        all_detections = []
        
        # 1. Run detectors
        for backend in models_to_run:
            # Full frame detection
            det_results = self.detect_only_model(image, backend)
            all_detections.extend(det_results)
            
            # Tiled detection
            if should_tile:
                from detection.tiling import get_tiles, map_to_original
                tiles = get_tiles(image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
                for tile_data in tiles:
                    tile_orig = tile_data['origin']
                    tile_detections = self.detect_only_model(tile_data['tile'], backend)
                    
                    for det in tile_detections:
                        det['box'] = map_to_original(det['box'], tile_orig)
                        all_detections.append(det)
        
        # 2. Apply Fusion/NMS
        if not all_detections:
            return []
            
        if should_ensemble:
            from detection.ensemble_detection import fuse_detections
            final_detections = fuse_detections(all_detections, iou_threshold=NMS_IOU_THRESHOLD)
        else:
            from detection.tiling import apply_nms
            final_detections = apply_nms(all_detections, iou_threshold=NMS_IOU_THRESHOLD)
        
        # Filter by minimum size
        valid_detections = []
        for det in final_detections:
            w, h = det['box'][2], det['box'][3]
            if w >= self.min_face_size[0] and h >= self.min_face_size[1]:
                valid_detections.append(det)
                
        return valid_detections

    def extract_face(self, image: np.ndarray, box: List[int], target_size: Tuple[int, int] = (160, 160), padding_pct: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract a face from the image with padding and resize it.
        """
        if image is None: return None
        x, y, w, h = box
        img_h, img_w = image.shape[:2]
        pad_w, pad_h = int(w * padding_pct), int(h * padding_pct)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)
        
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0: return None
        
        # Resize to target size (standard behavior)
        if target_size:
            # Check if we should ENHANCE before resizing
            # If face is small (< 64px) and we are resizing up to 160px, use SR
            # OPTIMIZATION: Only use SR if GPU is available to prevent UI freezing
            h, w = face_crop.shape[:2]
            from models.gpu_manager import GPUModelManager
            gpu_mgr = GPUModelManager()
            if (w < 64 or h < 64) and gpu_mgr.is_gpu_ready():
                 face_crop = self.enhancer.enhance_face(face_crop)
            
            return cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
            
        return face_crop

    def check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform basic quality checks on a face image.
        """
        results = {"passed": True, "blur_score": 0.0, "is_blurry": False, "brightness": 0.0, "is_too_dark": False, "is_too_bright": False}
        if image is None: return {"passed": False}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        results["blur_score"] = blur_score
        
        # Extremely loose for low-end webcams
        if blur_score < 5: 
            results["is_blurry"] = True
            results["passed"] = False
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:, :, 2])
        results["brightness"] = avg_brightness
        
        if avg_brightness < 30:
            results["is_too_dark"] = True
            results["passed"] = False
        elif avg_brightness > 235:
            results["is_too_bright"] = True
            results["passed"] = False
        return results

    def validate_for_enrollment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Special validation for student enrollment.
        """
        detections = self.detect_faces(image, use_tiling=False) # Disable tiling for enrollment for speed
        if len(detections) != 1:
            return {"passed": False, "message": "Ensure exactly ONE face is in frame.", "face_box": None}
        face_box = detections[0]['box']
        face_crop = self.extract_face(image, face_box)
        quality = self.check_image_quality(face_crop)
        if not quality["passed"]:
            return {"passed": False, "message": "Low quality detected.", "face_box": face_box}
        return {"passed": True, "message": "Quality OK", "face_box": face_box, "face_crop": face_crop}
