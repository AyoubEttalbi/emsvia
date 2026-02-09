import torch
import logging
import gc
from config.settings import (
    USE_GPU, 
    DEVICE, 
    USE_MIXED_PRECISION, 
    GPU_MEMORY_FRACTION, 
    ALLOW_GROWTH,
    CLEAR_CACHE_INTERVAL
)

logger = logging.getLogger(__name__)

class GPUModelManager:
    """
    Manages GPU memory and model loading for face detection and recognition.
    Ensures models are loaded once and shared across components.
    Handles mixed precision, memory optimization, and cache management.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device(DEVICE if torch.cuda.is_available() and USE_GPU else "cpu")
        self.use_mixed_precision = USE_MIXED_PRECISION and self.device.type == "cuda"
        self.models = {}
        self.frame_count = 0
        
        if self.device.type == "cuda":
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # GPU Optimizations
            if ALLOW_GROWTH:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
                
            # Enable TF32 for faster matrix operations (Ampere+ GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN benchmarking for faster convolutions
            torch.backends.cudnn.benchmark = True
            
            # Disable deterministic mode for better performance
            torch.backends.cudnn.deterministic = False
            
            if self.use_mixed_precision:
                logger.info("⚡ Mixed precision (FP16) enabled for 2x speedup")
        else:
            logger.warning("⚠️  No GPU detected or USE_GPU=False. Falling back to CPU.")
            logger.warning("    Performance will be significantly reduced. GPU acceleration recommended.")
            
        self._initialized = True

    def clear_cache(self, force=False):
        """Clear GPU cache to free up VRAM."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            if force:
                logger.debug("🧹 GPU cache cleared")
    
    def auto_clear_cache(self):
        """Automatically clear cache at intervals."""
        self.frame_count += 1
        if self.frame_count % CLEAR_CACHE_INTERVAL == 0:
            self.clear_cache(force=False)

    def get_device(self):
        """Get the current device (cuda or cpu)."""
        return self.device

    def is_gpu_ready(self):
        """Simple check for GPU readiness."""
        return self.device.type == "cuda"
    
    def monitor_memory(self):
        """Monitor and return GPU memory usage stats."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization_pct": (allocated / total) * 100
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "utilization_pct": 0}
    
    def get_memory_summary(self):
        """Get formatted memory summary string."""
        if self.device.type == "cuda":
            stats = self.monitor_memory()
            return f"GPU: {stats['allocated_gb']:.2f}/{stats['total_gb']:.2f}GB ({stats['utilization_pct']:.1f}%)"
        return "CPU Mode"
    
    def preload_model(self, model_name, model):
        """Preload a model to GPU memory (not implemented yet - placeholder)."""
        # This is a placeholder for future enhancement where we can
        # load PyTorch versions of detection/recognition models
        if model_name not in self.models:
            if self.device.type == "cuda":
                logger.info(f"Loading {model_name} to GPU...")
                # Future: model = model.to(self.device).eval()
                self.models[model_name] = model
            else:
                self.models[model_name] = model
    
    def get_scaler(self):
        """Get GradScaler for mixed precision training/inference."""
        if self.use_mixed_precision:
            return torch.cuda.amp.GradScaler()
        return None
