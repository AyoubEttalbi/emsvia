import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Tuple, Optional, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Handles face detection using MTCNN.
    Includes utilities for face extraction, alignment, and quality checks.
    """
    
    def __init__(self, min_confidence: float = 0.9, min_face_size: Tuple[int, int] = (60, 60)):
        """
        Initialize the Face Detector.
        
        Args:
            min_confidence: Minimum confidence threshold for detections (0.0 to 1.0)
            min_face_size: Minimum face size (width, height) to be considered valid
        """
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        try:
            self.detector = MTCNN()
            logger.info("MTCNN Face Detector initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            raise e

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image in BGR format (OpenCV standard)
            
        Returns:
            List of dictionaries containing detection details:
            {
                'box': [x, y, width, height],
                'confidence': float,
                'keypoints': {
                    'left_eye': (x, y),
                    'right_eye': (x, y),
                    'nose': (x, y),
                    'mouth_left': (x, y),
                    'mouth_right': (x, y)
                }
            }
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_faces")
            return []

        # Convert BGR to RGB (MTCNN expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            detections = self.detector.detect_faces(image_rgb)
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []

        valid_detections = []
        for detection in detections:
            confidence = detection.get('confidence', 0.0)
            box = detection.get('box', [0, 0, 0, 0])
            width, height = box[2], box[3]
            
            # 1. Confidence Check
            if confidence < self.min_confidence:
                continue
                
            # 2. Size Check
            if width < self.min_face_size[0] or height < self.min_face_size[1]:
                continue
            
            valid_detections.append(detection)
            
        return valid_detections

    def extract_face(self, image: np.ndarray, box: List[int], target_size: Tuple[int, int] = (160, 160), padding_pct: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract a face from the image with padding and resize it.
        
        Args:
            image: Source image (BGR)
            box: Bounding box [x, y, width, height]
            target_size: Output size (width, height)
            padding_pct: Percentage of padding to add around the face (default 10%)
            
        Returns:
            Resized face image (BGR) or None if extraction fails.
        """
        if image is None: 
            return None
            
        x, y, w, h = box
        img_h, img_w = image.shape[:2]
        
        # Calculate padding
        pad_w = int(w * padding_pct)
        pad_h = int(h * padding_pct)
        
        # Apply padding with boundary checks
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        # Extract face crop
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
            
        # Resize to target size
        try:
            face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
            return face_resized
        except Exception as e:
            logger.error(f"Error resizing face: {e}")
            return None

    def check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform basic quality checks on a face image.
        
        Args:
            image: Face image (BGR)
            
        Returns:
            Dictionary with quality metrics and pass/fail status.
        """
        results = {
            "passed": True,
            "blur_score": 0.0,
            "is_blurry": False,
            "brightness": 0.0,
            "is_too_dark": False,
            "is_too_bright": False
        }
        
        if image is None:
            results["passed"] = False
            return results

        # 1. Blur Detection (Variance of Laplacian)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        results["blur_score"] = blur_score
        
        # Threshold can be adjusted (e.g., < 100 often considered blurry)
        if blur_score < 100: 
            results["is_blurry"] = True
            results["passed"] = False

        # 2. Brightness Check
        # Convert to HSV, extract V channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        avg_brightness = np.mean(v_channel)
        results["brightness"] = avg_brightness
        
        if avg_brightness < 40: # Too dark
            results["is_too_dark"] = True
            results["passed"] = False
        elif avg_brightness > 220: # Too bright
            results["is_too_bright"] = True
            results["passed"] = False
            
        return results

    def validate_for_enrollment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Special validation for student enrollment.
        Ensures exactly one high-quality face is present.
        """
        results = {"passed": False, "message": "", "face_box": None}
        
        # 1. Detect all faces
        detections = self.detect_faces(image)
        
        if len(detections) == 0:
            results["message"] = "No face detected."
            return results
        
        if len(detections) > 1:
            results["message"] = "Multiple faces detected! Please ensure only one person is in frame."
            return results
            
        # 2. Extract and check quality of the single face
        face_box = detections[0]['box']
        face_crop = self.extract_face(image, face_box)
        
        quality = self.check_image_quality(face_crop)
        if not quality["passed"]:
            msg = "Low quality: "
            if quality.get("is_blurry"): msg += "Blurry "
            if quality.get("is_too_dark"): msg += "Too dark "
            if quality.get("is_too_bright"): msg += "Too bright "
            results["message"] = msg.strip()
            return results
            
        results["passed"] = True
        results["message"] = "Quality OK"
        results["face_box"] = face_box
        results["face_crop"] = face_crop
        return results
