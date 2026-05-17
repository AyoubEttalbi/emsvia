"""
GPU environment bootstrap for TensorFlow / DeepFace.

Call setup_gpu_environment() BEFORE importing TensorFlow, DeepFace, or any
model wrapper that transitively imports them. TF reads device configuration
at import time, so the order matters.
"""
import os
import sys
import ctypes
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent


def setup_gpu_environment() -> bool:
    """
    Pre-load cuDNN 8 and configure TensorFlow to use the GPU.

    Returns True if a GPU was successfully configured, False on CPU fallback.
    """
    # ── 1. Environment flags (must be set before any TF import) ──────────────
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ORT_LOGGING_LEVEL"] = "3"
    # Disable XLA auto-jit (causes issues on GTX 10-series)
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit"

    # ── 2. Pre-load cuDNN 8 via ctypes RTLD_GLOBAL ───────────────────────────
    # TF links against cuDNN 8 but Ubuntu ships cuDNN 9 by default.
    # Loading the local cuDNN 8 copies with RTLD_GLOBAL makes their symbols
    # globally visible so TF can find them without changing LD_LIBRARY_PATH.
    cudnn8_dir = _BASE_DIR / "libs" / "cudnn8"
    if cudnn8_dir.exists():
        for lib in [
            "libcudnn.so.8",
            "libcudnn_ops_infer.so.8",
            "libcudnn_cnn_infer.so.8",
            "libcudnn_adv_infer.so.8",
        ]:
            lib_path = cudnn8_dir / lib
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    sys.stdout.write(f"  [gpu_init] Warning: could not preload {lib}: {e}\n")

        # Also expose for any child processes spawned later
        _prepend_ld(str(cudnn8_dir))

    # Add common CUDA system paths
    for cuda_path in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]:
        if os.path.isdir(cuda_path):
            _prepend_ld(cuda_path)

    # ── 3. Configure TF GPU memory limit ─────────────────────────────────────
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            sys.stdout.write("⚠️  [gpu_init] TensorFlow sees no GPU — using CPU\n")
            sys.stdout.flush()
            return False

        try:
            mem_mb = int(os.getenv("TF_GPU_MEMORY_LIMIT", "3500"))
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=mem_mb)],
            )
        except RuntimeError:
            # Already initialized (called twice in the same process) — harmless
            pass

        sys.stdout.write(
            f"✅ [gpu_init] TensorFlow GPU ready: {len(gpus)} device(s), "
            f"limit={os.getenv('TF_GPU_MEMORY_LIMIT', '3500')} MB\n"
        )
        sys.stdout.flush()
        return True

    except Exception as e:
        sys.stdout.write(f"⚠️  [gpu_init] TF GPU setup failed: {e}\n")
        sys.stdout.flush()
        return False


def _prepend_ld(path: str):
    """Prepend a path to LD_LIBRARY_PATH if not already present."""
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if path not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{current}" if current else path
