import os
import sys
from pathlib import Path

# Build LD_LIBRARY_PATH exactly as in main.py
venv_site_packages = Path(__file__).resolve().parent / "venv" / "lib" / "python3.12" / "site-packages"
nvidia_base = venv_site_packages / "nvidia"

if nvidia_base.exists():
    cuda_libs = []
    for p in nvidia_base.glob("**/lib"):
        if p.is_dir():
            cuda_libs.append(str(p))
    
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(cuda_libs)
    if current_ld:
        new_ld = f"{new_ld}:{current_ld}"
    os.environ["LD_LIBRARY_PATH"] = new_ld
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={str(nvidia_base)}"

print("Checking Library Visibility...")

import torch
print(f"Torch Version: {torch.__version__}")
print(f"Torch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Torch Device: {torch.cuda.get_device_name(0)}")

print("\n--- Testing TensorFlow ---")
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPUs available: {gpus}")
    if not gpus:
        print("TF: NO GPU DETECTED. Check logs above for 'Could not load dynamic library' errors.")
except Exception as e:
    print(f"TF Error: {e}")

print("\n--- Testing ONNX Runtime ---")
try:
    import onnxruntime as ort
    print(f"ONNX Providers: {ort.get_available_providers()}")
except Exception as e:
    print(f"ONNX Error: {e}")
