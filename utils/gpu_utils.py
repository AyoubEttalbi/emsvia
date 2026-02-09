"""
GPU utilities for memory management, monitoring, and optimization.
"""
import torch
import gc
import logging
from typing import Dict, Any
from config.settings import GPU_MEMORY_FRACTION

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache between batches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def monitor_memory() -> Dict[str, float]:
        """Monitor GPU memory usage."""
        if torch.cuda.is_available():
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
    
    @staticmethod
    def print_memory_stats():
        """Print detailed GPU memory statistics."""
        if torch.cuda.is_available():
            stats = GPUMemoryManager.monitor_memory()
            logger.info(f"GPU Memory: {stats['allocated_gb']:.2f}GB allocated, {stats['reserved_gb']:.2f}GB reserved, {stats['total_gb']:.2f}GB total")
            logger.info(f"GPU Utilization: {stats['utilization_pct']:.1f}%")
        else:
            logger.info("No GPU available")
    
    @staticmethod
    def optimize_memory():
        """Optimize memory usage with PyTorch settings."""
        if torch.cuda.is_available():
            # Enable memory efficient operations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels
            torch.backends.cudnn.deterministic = False  # Faster, less reproducible
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
            
            logger.info("GPU memory optimizations applied")


def get_optimal_batch_size(base_batch_size: int = 8, available_memory_gb: float = None) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        base_batch_size: Default batch size
        available_memory_gb: Available GPU memory in GB
    
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 1  # CPU mode, process one at a time
    
    if available_memory_gb is None:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        available_memory_gb = total_memory - allocated
    
    # Heuristic: ~0.5GB per face for deep models
    memory_per_face = 0.5
    calculated_batch_size = int(available_memory_gb / memory_per_face)
    
    # Limit to reasonable range
    optimal_batch_size = max(1, min(calculated_batch_size, base_batch_size))
    
    logger.debug(f"Optimal batch size: {optimal_batch_size} (available memory: {available_memory_gb:.2f}GB)")
    return optimal_batch_size
