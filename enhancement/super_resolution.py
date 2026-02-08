import cv2
import numpy as np
import logging
from pathlib import Path
from config.settings import USE_SUPER_RESOLUTION, SR_MODEL, SR_SCALE, SR_MIN_SIZE, DATA_DIR

logger = logging.getLogger(__name__)

class FaceEnhancer:
    """
    Handles Face Super-Resolution using OpenCV's DNN SuperRes module.
    Supports FSRCNN and EDSR models.
    """
    
    def __init__(self):
        self.enabled = USE_SUPER_RESOLUTION
        self.model_name = SR_MODEL
        self.scale = SR_SCALE
        self.min_size = SR_MIN_SIZE
        self.sr = None
        
        if self.enabled:
            self._load_model()
            
    def _load_model(self):
        try:
            from cv2 import dnn_superres
            self.sr = dnn_superres.DnnSuperResImpl_create()
            
            model_path = DATA_DIR / "models" / f"{self.model_name}_x{self.scale}.pb"
            
            if not model_path.exists():
                logger.warning(f"SR Model weights not found at {model_path}. Disabling enhancement.")
                self.enabled = False
                return

            self.sr.readModel(str(model_path))
            # Set the model name and scale
            # Note: For FSRCNN, the model name in opencv is 'fsrcnn', mostly lowercase.
            # Convert generic name to opencv expected name
            cv_model_name = self.model_name.lower()
            self.sr.setModel(cv_model_name, self.scale)
            
            logger.info(f"FaceEnhancer initialized with {self.model_name} x{self.scale}")
            
        except ImportError:
            logger.error("opencv-contrib-python is required for dnn_superres. Please install it.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize FaceEnhancer: {e}")
            self.enabled = False

    def enhance_face(self, face_img: np.ndarray, confidence: float = 1.0) -> np.ndarray:
        """
        Apply Super-Resolution to a face crop if it meets criteria.
        
        Args:
            face_img: BGR face crop
            confidence: Detection confidence (optional filter)
            
        Returns:
            Enhanced face image (or original if not enhanced)
        """
        if not self.enabled or face_img is None or face_img.size == 0:
            return face_img
            
        h, w = face_img.shape[:2]
        
        # Criteria: Only enhance if face is small
        if w < self.min_size or h < self.min_size:
            try:
                # Upscale
                enhanced_img = self.sr.upsample(face_img)
                # logger.debug(f"Enhanced face from {w}x{h} to {enhanced_img.shape[1]}x{enhanced_img.shape[0]}")
                return enhanced_img
            except Exception as e:
                logger.error(f"Error during face enhancement: {e}")
                return face_img
                
        return face_img
