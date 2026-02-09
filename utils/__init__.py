"""GPU utilities package."""
from .gpu_utils import GPUMemoryManager, get_optimal_batch_size

__all__ = ['GPUMemoryManager', 'get_optimal_batch_size']
